# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix, upfirdn2d
from training.DiffAugment_pytorch import DiffAugment
from torch.nn import functional as F

from training.distillation import lpips
from training.distillation.Util.content_aware_pruning import Get_Parsing_Net, Batch_Img_Parsing, Get_Masked_Tensor



def Downsample_Image_256(im_tensor):
    im_tensor = F.interpolate(im_tensor, size=(256,256), mode='bilinear', align_corners=False)
    return im_tensor


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, T=None,augment_pipe=None, r1_gamma=10,
                 kd_l1_lambda=0, kd_lpips_lambda=0, kd_mode='Output_Only', content_aware_KD=True, LPIPS_IMAGE_SIZE=256,
                 style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
                 G_top_k = False, G_top_k_gamma = 0.9, G_top_k_frac = 0.5,diffaugment=''):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.T                  = T
        self.augment_pipe       = augment_pipe
        self.diffaugment = diffaugment
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.G_top_k = G_top_k
        self.G_top_k_gamma = G_top_k_gamma
        self.G_top_k_frac = G_top_k_frac
        self.kd_l1_lambda = kd_l1_lambda
        self.kd_lpips_lambda = kd_lpips_lambda
        self.kd_mode = kd_mode
        self.content_aware_KD = content_aware_KD
        self.LPIPS_IMAGE_SIZE = LPIPS_IMAGE_SIZE

        if (self.T is not None) and (self.kd_lpips_lambda > 0):
            self.percept_loss = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
            print("USING LPIPS LOSS", self.kd_lpips_lambda)
        else:
            self.percept_loss = None

            # Content aware KD
        if (self.T is not None) and (self.content_aware_KD):
            self.parsing_net, _ = Get_Parsing_Net(device)
            print("USING CONTENT AWARE KD", self.kd_l1_lambda)
        else:
            self.parsing_net = None

    def KD_Loss(self, z, c, fake_img, fake_img_list,update_emas=False):
        fake_img_teacher, fake_img_teacher_list = self.T(z, c, update_emas=update_emas, get_rgb_list=True)
        if self.parsing_net is not None:
            teacher_img_parsing = Batch_Img_Parsing(fake_img_teacher, self.parsing_net, self.device)
            fake_img_teacher = Get_Masked_Tensor(fake_img_teacher, teacher_img_parsing, self.device, mask_grad=False)
            fake_img = Get_Masked_Tensor(fake_img, teacher_img_parsing, self.device, mask_grad=True)

        fake_img_teacher.requires_grad = True

        # kd_l1_loss
        if self.kd_mode == 'Output_Only':
            kd_l1_loss = self.kd_l1_lambda * torch.mean(torch.abs(fake_img_teacher - fake_img))
        elif self.kd_mode == 'Intermediate':
            for fake_img_teacher in fake_img_teacher_list:
                fake_img_teacher.requires_grad = True
            loss_list = [torch.mean(torch.abs(fake_img_teacher - fake_img)) for (fake_img_teacher, fake_img) in
                         zip(fake_img_teacher_list, fake_img_list)]
            kd_l1_loss = self.kd_l1_lambda * sum(loss_list)
        else:
            kd_l1_loss = 0

        # kd_lpips_loss
        if self.percept_loss is None:
            kd_lpips_loss = torch.tensor(0.0, device=self.device)
        else:
            if self.G.img_resolution > self.LPIPS_IMAGE_SIZE:  # pooled the image for LPIPS for memory saving
                pooled_fake_img = Downsample_Image_256(fake_img)
                pooled_fake_img_teacher = Downsample_Image_256(fake_img_teacher)
                kd_lpips_loss = self.kd_lpips_lambda * torch.mean(
                    self.percept_loss(pooled_fake_img, pooled_fake_img_teacher))

            else:
                kd_lpips_loss = self.kd_lpips_lambda * torch.mean(self.percept_loss(fake_img, fake_img_teacher))

        return kd_l1_loss, kd_lpips_loss

    def run_G(self, z, c, update_emas=False, get_rgb_list=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img, rgb = self.G.synthesis(ws, update_emas=update_emas, get_rgb_list=get_rgb_list)
        return img, rgb, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        if self.diffaugment:
            img = DiffAugment(img, policy=self.diffaugment)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                minibatch_size = gen_z.shape[0]
                gen_img, rgb, _gen_ws = self.run_G(gen_z, gen_c, get_rgb_list=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                if self.G_top_k:
                    D_fake_scores = gen_logits
                    k_frac = np.maximum(self.G_top_k_gamma ** self.G.epochs, self.G_top_k_frac)
                    k = int(np.ceil(minibatch_size * k_frac))
                    lowest_k_scores, _ = torch.topk(-torch.squeeze(D_fake_scores),
                                                    k=k)  # want smallest probabilities not largest
                    gen_logits = torch.unsqueeze(-lowest_k_scores, axis=1)
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                if self.T is not None:
                    kd_l1_loss, kd_lpips_loss = self.KD_Loss(gen_z, gen_c, gen_img, rgb)
                    training_stats.report('Loss/G/kd_l1_loss', kd_l1_loss)
                    training_stats.report('Loss/G/kd_lpips_loss', kd_lpips_loss)
            if self.T is not None:
                with torch.autograd.profiler.record_function('Gmain_backward'):
                    (loss_Gmain.mean().mul(gain) + kd_l1_loss + kd_lpips_loss).backward()
            else:
                with torch.autograd.profiler.record_function('Gmain_backward'):
                    loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, _, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
