# Tools

The Tools screen gathers four utilities in a single grid: [Projection](#projection), [Feature Extractor](#feature-extractor), [Super Resolution](#super-resolution), and [Model Mixing](#model-mixing).

## Projection

The Projection tool allows loading a target image, a text prompt, or a combination of both, to find the closest corresponding latent vector of a trained model. Different methods are provided to search for the corresponding latent vector that you can choose from. The output is the calculated vector plus a process video showing the search process.

![](assets/offline-modules-projection-module-01.png)

The following options are available for projection:

- Models: Select the .pkl file of the model used for projection.
- Target Image: Choose a single image that should be found in the model.
- Target Text: Provide a description of an image that should be found in the model.
- Save Path: Specify the path where the closest match should be saved. The saved result includes the position in the model and the closest match.
- Save Video: Select this option to save a video of the process of finding the closest match.
- Seed: Select a seed to ensure consistent results for the closest match.
- Learning Rate: Adjust the learning rate of the projection to control convergence speed. Higher values result in faster projections but may be less accurate.
- Steps: Determine the number of steps for the projection. Higher values increase accuracy but require more time.
- Use VGG: Enable this option to use the VGG16 network for calculating image distance based on general features rather than pixel-by-pixel comparisons.
- Use CLIP: Enable this option to use the CLIP network for calculating image and text distance based on general features rather than pixel-by-pixel comparisons.
- Use Pixel: Enable this option to use pixel distance for image comparison. This puts more weight on pixel similarity between the match and the target.
- Use Penalty: Enable this option to penalize large update steps, resulting in a smoother projection and avoiding local minimums.
- Use Center: Enable this option to use an additional center crop as the target image. This can improve matching accuracy but may reduce overall accuracy.

## Feature Extractor

This tool identifies and extracts interpretable feature directions in the latent space of a trained model based on Principal Component Analysis (PCA), using the GANSpace method. These extracted feature directions can then be used on the Perform screen (see [Adjust Input](perform.md#adjust-input)).

![](assets/offline-modules-ganspace-01.png)

The tool provides the following options:

- Models: Select the .pkl file of the model for feature extraction.
- PCA Estimator: Choose an algorithm to find salient directions in the latent space. Different algorithms result in slightly different extracted directions. Experiment with each option to find the best fit for your model.
- Features: Specify the number of features to extract. More features result in more adjustable directions on the Perform screen but also increase file size. Often, extracting around 10 features is recommended.
- Sparsity: This parameter adjusts the sparsity of the extracted directions. Leave it at the default if you are unsure about its impact.
- Save Path: Choose the path to save the results, which include the extracted directions. These directions can be loaded in the "[Adjust Input](perform.md#adjust-input)" widget on the Perform screen.

For more technical information on GANSpace, please refer to the following resources:

[GANSpace Paper](https://www.google.com/url?q=https://arxiv.org/abs/2004.02546&sa=D&source=editors&ust=1769724592296454&usg=AOvVaw1GSkgPNlwUg633Ht9d3zn6)

[GANSpace Repository on GitHub](https://www.google.com/url?q=https://github.com/harskish/ganspace&sa=D&source=editors&ust=1769724592296788&usg=AOvVaw3Xsw5-fahRCc_ZXrq6FtT5)

## Super Resolution

The Super Resolution tool provides the feature to upscale images and videos for higher resolution distribution of your work. There are three different models available: Quality, Balance, and Fast. While the Processing speeds of the models differ, Quality being the slowest and Fast being the fastest, the slower models tend to preserve more details from the original image/video. However, the quality of the results also depends on the type of visual content and is better to be tested with all three models to find the desired option.

![](assets/offline-modules-super-resolution-01.png)

The following options are available:

- Input Files: Select the path to the images or videos that should be upscaled. Multiple files can be selected at once.
- Result Path: Specify the path where the upscaled images or videos should be saved.
- Model: Choose the upscaling model to be used: Fast, Balanced, or Quality. Fast is the quickest but provides lower quality, while Quality is the slowest but provides the highest quality.
- Scale Mode: Choose between defining the exact Height and Width of the output or using a Scale Factor. If the scale factor is selected, the height and width will be automatically calculated based on the scale factor.
- Sharpening: Adjust the sharpness of the upscaled image. Sharpening=1 means no added sharpness. Higher values result in sharper images but may introduce artifacts.

## Model Mixing

This tool provides the ability to mix two trained models to make a new model. It works based on selecting parts of one model and parts of a different model, thereby mixing the features of the two models.

![](assets/offline-modules-model-mixing-01.png)

To mix two models, specify the following:

- Model 1: Select the .pkl file of the first model to be mixed.
- Model 2: Select the .pkl file of the second model to be mixed.

After pressing "Combine," you can select the layers to be used from each model. The layers are listed in order, from early to late layers. While lower-resolution layers (e.g. 4x4, 8x8, 16x16) correspond to coarse and higher-level features, higher-resolution layers correspond to fine features and textures. This can be used to determine what type of features the mixed model inherits from each source. Each layer can also be expanded to select detailed components for advanced mixing. Additionally, features can be removed from the mix. This can be useful when one of the models generates images of higher resolution, and you want to align the resolution with the lower-resolution model.

![](assets/offline-modules-model-mixing-02.png)

Finally, you have the option to save the mixed model as a new .pkl file, which can be used anywhere in Autolume.

For real-time model mixing on the Perform screen, see [Model Mixing (real-time)](perform.md#model-mixing-real-time).
