# import os
# import imgui
# from utils.gui_utils import imgui_utils
# from widgets.browse_widget import BrowseWidget



# import numpy as np

# class PresetWidget:
#     def __init__(self, viz):
#         self.viz = viz
#         self.num_presets = 12
#         self.c_inactive = imgui.get_style().colors[imgui.COLOR_MENUBAR_BACKGROUND]
#         self.active = np.asarray([False] * self.num_presets)
#         self.dir_name = np.asarray([f"{i}" for i in range(self.num_presets)], dtype=object)

#         self.path = "presets"
#         if not os.path.isdir(self.path):
#             os.mkdir(self.path)
#         if len(os.listdir(self.path)) == 0:
#             for i in range(self.num_presets):
#                 os.makedirs(self.path + '/' + str(i))
#         else:
#             print(self.path)
#             for i, dir in enumerate(os.listdir(self.path)):
#                 self.dir_name[i] = dir
#         self.tmp_path = self.path
#         self.paths = np.asarray([f"{self.path}/{i}" for i in range(self.num_presets)], dtype=object)
#         print(self.dir_name)
#         self.check_presets()
#         self.recent_paths = [self.path]
#         self.use_osc = False
#         self.osc_addresses = ""
#         self.browser = BrowseWidget(self.viz, "Preset Path##presetwidget", os.path.abspath(os.getcwd()), [""], width=viz.app.font_size * 10, multiple=False, traverse_folders=False)



#     def check_presets(self):
#         if not os.path.exists(self.path):
#             os.makedirs(self.path)
#         if len(os.listdir(self.path)) == 0:
#             for i in range(self.num_presets):
#                 os.makedirs(self.path + '/' + str(i))
#         self.assigned = np.ones(self.num_presets)
#         print(self.assigned)
#         for p in os.listdir(self.path):
#             #get index of directory
#             print(p)
#             print(self.dir_name)
#             i = np.where(self.dir_name==p)[0]
#             self.assigned[i] = 1

#     def open_path(self, path):
#         try:
#             if not os.path.exists(path):
#                 os.makedirs(path)
#             print(path, os.listdir(path))
#             if len(os.listdir(path)) == 0:
#                 for i in range(self.num_presets):
#                     self.dir_name = np.asarray([f"{i}" for i in range(self.num_presets)], dtype=object)
#                     os.makedirs(path + '/' + self.dir_name[i])
#             else:
#                 for i, dir in enumerate(os.listdir(path)):
#                     self.dir_name[i] = dir
#             self.paths = np.asarray([f"{path}/{self.dir_name[i]}" for i in range(self.num_presets)], dtype=object)
#             print('paden: ' + str(self.paths))
#             self.path = self.tmp_path
#             self.check_presets()
#         except Exception as e:
#             print(e)

#     def save(self, path):
#         try:
#             if not os.path.exists(path):
#                 os.makedirs(path)
#             # self.inter.terminal_widget.cached_text.append(f"Saving Preset at {path}")
#             # self.viz.latent_widget.save(f"{path}/latent.pkl")
#             # self.viz.trunc_noise_widget.save(f"{path}/trunc.pkl")
#             # self.viz.layer_widget.save(f"{path}/layer.pkl")
#             # self.viz.adjuster_widget.save(f"{path}/adjuster.pkl")
#             # self.viz.looping_widget.save(f"{path}/looper.pkl")
#             # self.viz.pickle_widget.save(f"{path}/pickle.pkl")
#             # self.viz.collapsed_widget.save(f"{path}/collap.pkl")
#             # self.viz.mixing_widget.save(f"{path}/mix.pkl")
#             self.viz.latent_widget.save(os.path.join(path, "latent.pkl"))
#             self.viz.trunc_noise_widget.save(os.path.join(path, "trunc.pkl"))
#             self.viz.layer_widget.save(os.path.join(path, "layer.pkl"))
#             self.viz.adjuster_widget.save(os.path.join(path, "adjuster.pkl"))
#             self.viz.looping_widget.save(os.path.join(path, "looper.pkl"))
#             self.viz.pickle_widget.save(os.path.join(path, "pickle.pkl"))
#             self.viz.collapsed_widget.save(os.path.join(path, "collap.pkl"))
#             self.viz.mixing_widget.save(os.path.join(path, "mix.pkl"))
#             self.assigned[np.where(self.active)] = 0
#         except Exception as e:
#             print(e)

#     def load(self, path):
#         try:
#             # self.viz.terminal_widget.cached_text.append(f"Loading Preset from {path}")
#             # self.viz.latent_widget.load(f"{path}/latent.pkl")
#             # self.viz.trunc_noise_widget.load(f"{path}/trunc.pkl")
#             # self.viz.layer_widget.load(f"{path}/layer.pkl")
#             # self.viz.adjuster_widget.load(f"{path}/adjuster.pkl")
#             # self.viz.looping_widget.load(f"{path}/looper.pkl")
#             # self.viz.pickle_widget.load(f"{path}/pickle.pkl")
#             # self.viz.collapsed_widget.load(f"{path}/collap.pkl")
#             # self.viz.mixing_widget.load(f"{path}/mix.pkl")
#             self.viz.latent_widget.load(os.path.join(path, "latent.pkl"))
#             self.viz.trunc_noise_widget.load(os.path.join(path, "trunc.pkl"))
#             self.viz.layer_widget.load(os.path.join(path, "layer.pkl"))
#             self.viz.adjuster_widget.load(os.path.join(path, "adjuster.pkl"))
#             self.viz.looping_widget.load(os.path.join(path, "looper.pkl"))
#             try: 
#                 self.viz.pickle_widget.load(os.path.join(path, "pickle.pkl"))
#             except Exception as e:
#                 print(f"Ignored error while loading pickle.pkl: {e}")
#             self.viz.collapsed_widget.load(f"{path}/collap.pkl")
#             self.viz.mixing_widget.load(os.path.join(path, "mix.pkl"))
#             self.viz.app.skip_frame()
#         except Exception as e:
#             print(e)

#     @imgui_utils.scoped_by_object_id
#     def preset_checkbox(self, i):
#         with imgui_utils.grayed_out(self.assigned[i]):
#             clicked, _ = imgui.checkbox(f"##{i}", self.active[i])
#             if clicked:
#                 self.active *= False
#                 self.active[i] = True

#     @imgui_utils.scoped_by_object_id
#     def __call__(self, show=True):
#         viz = self.viz

#         if show:
#             for i in range(len(self.active)):
#                 imgui.begin_group()
#                 self.preset_checkbox(i)
#                 imgui.same_line()
#                 with imgui_utils.item_width(viz.app.button_w * 4):
#                     dir_name_copy = self.dir_name[i]
#                     _changed, dir_name = imgui_utils.input_text(f"##preset name {i}", dir_name_copy, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE, help_text="Set name for preset "+str(i))
#                     if _changed:
#                         print(dir_name_copy)
#                         print(dir_name)
#                         self.dir_name[i] = str(dir_name)
#                         self.paths[i] = str(self.tmp_path + '/' + self.dir_name[i])
#                         print(self.paths)
#                         if dir_name_copy in os.listdir(self.path):
#                             os.rename(self.tmp_path + '/' + dir_name_copy,  self.paths[i])
#                 imgui.end_group()
#             imgui.same_line()
#             if imgui_utils.button('Load##presets', width=viz.app.button_w):
#                 self.load(self.paths[np.where(self.active)].item())

#             imgui.same_line()
#             if imgui_utils.button("Save##presets", width=viz.app.button_w):
#                 if self.active.any():
#                     self.save(self.paths[np.where(self.active)].item())
#                     self.assigned[np.where(self.active)] = 0
#             with imgui_utils.item_width(viz.app.button_w * 2), imgui_utils.grayed_out(True):
#                 imgui.input_text("##preset_path", self.tmp_path, 256, imgui.INPUT_TEXT_READ_ONLY)
#             imgui.same_line()
#             _clicked, file_out = self.browser(self.viz.app.button_w)

#             if _clicked:
#                 self.tmp_path = file_out[0]

#             imgui.same_line()
#             if imgui_utils.button('Recent...##presets', width=viz.app.button_w, enabled=(len(self.recent_paths) != 0)):
#                 imgui.open_popup('recent_preset_popup')
#             if imgui.begin_popup('recent_preset_popup'):
#                 for pth in self.recent_paths:
#                     clicked, _state = imgui.menu_item(pth)
#                     if clicked:
#                         self.open_path(pth)
#                 imgui.end_popup()
#             imgui.same_line()

#             if imgui_utils.button('Open##presets', width=viz.app.button_w):
#                 self.open_path(self.tmp_path)
#                 if not self.tmp_path in self.recent_paths:
#                     self.recent_paths.append(self.tmp_path)
#             imgui.same_line()
#             _, self.use_osc = imgui.checkbox(f"Use OSC##load", self.use_osc)
#             imgui.same_line()
#             with imgui_utils.grayed_out(not (self.use_osc)):

#                 changed, osc_address = imgui_utils.input_text(f"##OSC_load",
#                                                               self.osc_addresses,
#                                                               256,
#                                                               imgui.INPUT_TEXT_CHARS_NO_BLANK |
#                                                               (
#                                                                   imgui.INPUT_TEXT_READ_ONLY) * (
#                                                                   not self.use_osc),
#                                                               width=viz.app.font_size * 5,
#                                                               help_text="Osc Address")
#                 if changed:
#                     try:
#                         viz.osc_dispatcher.unmap(f"/{self.osc_addresses}",
#                                                    self.osc_handler)
#                         self.osc_addresses = osc_address
#                     except:
#                         print(f"{self.osc_addresses} is not mapped")
#                     viz.osc_dispatcher.map(f"/{self.osc_addresses}",
#                                              self.osc_handler)

#     def osc_handler(self, address, *args):
#         value = str(args[-1])
#         index = np.where(self.dir_name == value)
#         self.active *= False
#         self.active[index]=True
#         self.load(self.paths[np.where(self.active)].item())


import os
import imgui
import numpy as np
from utils.gui_utils import imgui_utils
from widgets.browse_widget import BrowseWidget

class PresetWidget:
    def __init__(self, viz):
        self.viz = viz
        self.num_presets = 12  # 初始预设数量
        self.c_inactive = imgui.get_style().colors[imgui.COLOR_MENUBAR_BACKGROUND]
        self.active = np.asarray([False] * self.num_presets)
        self.dir_name = np.asarray([f"{i}" for i in range(self.num_presets)], dtype=object)
        self.scroll_index = 0  # 添加滚动索引

        self.path = "presets"
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        if len(os.listdir(self.path)) == 0:
            for i in range(self.num_presets):
                os.makedirs(self.path + '/' + str(i))
        else:
            print(self.path)
            for i, dir in enumerate(os.listdir(self.path)):
                if i < self.num_presets:
                    self.dir_name[i] = dir
                else:
                    self.num_presets = i + 1
                    self.dir_name = np.resize(self.dir_name, self.num_presets)
                    self.dir_name[i] = dir
                    self.active = np.resize(self.active, self.num_presets)
                    self.active[i] = False
        self.tmp_path = self.path
        self.paths = np.asarray([f"{self.path}/{i}" for i in range(self.num_presets)], dtype=object)
        print(self.dir_name)
        self.check_presets()
        self.recent_paths = [self.path]
        self.use_osc = False
        self.osc_addresses = ""
        self.browser = BrowseWidget(self.viz, "Preset Path##presetwidget", os.path.abspath(os.getcwd()), [""], width=viz.app.font_size * 10, multiple=False, traverse_folders=False)

    def check_presets(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        dirs = os.listdir(self.path)
        self.num_presets = max(12, len(dirs))
        self.active = np.resize(self.active, self.num_presets)
        self.dir_name = np.resize(self.dir_name, self.num_presets)
        self.assigned = np.ones(self.num_presets)
        
        for i, p in enumerate(dirs):
            if i < self.num_presets:
                self.dir_name[i] = p
                self.assigned[i] = 1
            else:
                break
        self.paths = np.asarray([f"{self.path}/{self.dir_name[i]}" for i in range(self.num_presets)], dtype=object)

    
    def create_new_folder(self):
        try:
            # 确保 self.num_presets 至少等于当前文件夹数量
            current_folders = len([name for name in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, name))])
            self.num_presets = max(self.num_presets, current_folders)

            new_folder_index = self.num_presets
            new_folder_name = str(new_folder_index)
            new_folder_path = os.path.join(self.path, new_folder_name)
            
            # 检查是否存在同名文件夹，如果存在就递增数字
            while os.path.exists(new_folder_path):
                new_folder_index += 1
                new_folder_name = str(new_folder_index)
                new_folder_path = os.path.join(self.path, new_folder_name)
            
            os.makedirs(new_folder_path, exist_ok=True)
            self.num_presets += 1
            self.dir_name = np.append(self.dir_name, new_folder_name)
            self.active = np.append(self.active, False)
            self.paths = np.append(self.paths, new_folder_path)
            self.assigned = np.append(self.assigned, 1)
            print(f"创建新文件夹: {new_folder_path}")
            self.check_presets()
        except Exception as e:
            print(f"创建新文件夹时出错: {e}")

    # def open_path(self, path):
    #     try:
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         dirs = os.listdir(path)
    #         self.num_presets = max(12, len(dirs))
    #         self.active = np.resize(self.active, self.num_presets)
    #         self.dir_name = np.resize(self.dir_name, self.num_presets)
            
    #         if len(dirs) == 0:
    #             for i in range(self.num_presets):
    #                 self.dir_name[i] = f"{i}"
    #                 os.makedirs(path + '/' + self.dir_name[i])
    #         else:
    #             for i, dir in enumerate(dirs):
    #                 if i < self.num_presets:
    #                     self.dir_name[i] = dir
    #                 else:
    #                     break
            
    #         self.paths = np.asarray([f"{path}/{self.dir_name[i]}" for i in range(self.num_presets)], dtype=object)
    #         print('paden: ' + str(self.paths))
    #         self.path = self.tmp_path
    #         self.check_presets()
    #     except Exception as e:
    #         print(e)
    def open_path(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            dirs = os.listdir(path)
            self.num_presets = max(12, len(dirs))
            self.active = np.resize(self.active, self.num_presets)
            self.dir_name = np.resize(self.dir_name, self.num_presets)
            
            if len(dirs) == 0:
                for i in range(self.num_presets):
                    new_dir_name = f"{i}"
                    self.dir_name[i] = new_dir_name
                    os.makedirs(os.path.join(path, new_dir_name), exist_ok=True)
            else:
                for i in range(self.num_presets):
                    if i < len(dirs):
                        self.dir_name[i] = dirs[i]
                    else:
                        self.dir_name[i] = f"{i}"
                        os.makedirs(os.path.join(path, self.dir_name[i]), exist_ok=True)
            
            self.paths = np.asarray([os.path.join(path, self.dir_name[i]) for i in range(self.num_presets)], dtype=object)
            print('paths: ' + str(self.paths))
            self.path = path
            self.tmp_path = path
            self.check_presets()
            self.scroll_index = 0  # 重置滚动索引
        except Exception as e:
            print(f"Error in open_path: {e}")


    def save(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            # self.inter.terminal_widget.cached_text.append(f"Saving Preset at {path}")
            # self.viz.latent_widget.save(f"{path}/latent.pkl")
            # self.viz.trunc_noise_widget.save(f"{path}/trunc.pkl")
            # self.viz.layer_widget.save(f"{path}/layer.pkl")
            # self.viz.adjuster_widget.save(f"{path}/adjuster.pkl")
            # self.viz.looping_widget.save(f"{path}/looper.pkl")
            # self.viz.pickle_widget.save(f"{path}/pickle.pkl")
            # self.viz.collapsed_widget.save(f"{path}/collap.pkl")
            # self.viz.mixing_widget.save(f"{path}/mix.pkl")
            self.viz.latent_widget.save(os.path.join(path, "latent.pkl"))
            self.viz.trunc_noise_widget.save(os.path.join(path, "trunc.pkl"))
            self.viz.layer_widget.save(os.path.join(path, "layer.pkl"))
            self.viz.adjuster_widget.save(os.path.join(path, "adjuster.pkl"))
            self.viz.looping_widget.save(os.path.join(path, "looper.pkl"))
            self.viz.pickle_widget.save(os.path.join(path, "pickle.pkl"))
            self.viz.collapsed_widget.save(os.path.join(path, "collap.pkl"))
            self.viz.mixing_widget.save(os.path.join(path, "mix.pkl"))
            self.assigned[np.where(self.active)] = 0
        except Exception as e:
            print(e)

    def load(self, path):
        try:
            # self.viz.terminal_widget.cached_text.append(f"Loading Preset from {path}")
            # self.viz.latent_widget.load(f"{path}/latent.pkl")
            # self.viz.trunc_noise_widget.load(f"{path}/trunc.pkl")
            # self.viz.layer_widget.load(f"{path}/layer.pkl")
            # self.viz.adjuster_widget.load(f"{path}/adjuster.pkl")
            # self.viz.looping_widget.load(f"{path}/looper.pkl")
            # self.viz.pickle_widget.load(f"{path}/pickle.pkl")
            # self.viz.collapsed_widget.load(f"{path}/collap.pkl")
            # self.viz.mixing_widget.load(f"{path}/mix.pkl")
            self.viz.latent_widget.load(os.path.join(path, "latent.pkl"))
            self.viz.trunc_noise_widget.load(os.path.join(path, "trunc.pkl"))
            self.viz.layer_widget.load(os.path.join(path, "layer.pkl"))
            self.viz.adjuster_widget.load(os.path.join(path, "adjuster.pkl"))
            self.viz.looping_widget.load(os.path.join(path, "looper.pkl"))
            try: 
                self.viz.pickle_widget.load(os.path.join(path, "pickle.pkl"))
            except Exception as e:
                print(f"Ignored error while loading pickle.pkl: {e}")
            self.viz.collapsed_widget.load(f"{path}/collap.pkl")
            self.viz.mixing_widget.load(os.path.join(path, "mix.pkl"))
            self.viz.app.skip_frame()
        except Exception as e:
            print(e)

    @imgui_utils.scoped_by_object_id
    def preset_checkbox(self, i):
        with imgui_utils.grayed_out(self.assigned[i]):
            clicked, _ = imgui.checkbox(f"##{i}", self.active[i])
            if clicked:
                self.active *= False
                self.active[i] = True

    # @imgui_utils.scoped_by_object_id
    # def __call__(self, show=True):
    #     viz = self.viz

    #     if show:
    #         # 检查是否需要更新预设列表
    #         current_dirs = set(os.listdir(self.path))
    #         if set(self.dir_name) != current_dirs:
    #             self.open_path(self.path)  # 重新加载当前路径
    #         # 创建一个水平布局
    #         imgui.begin_group()

    #         # 左侧：预设列表和滚动条
    #         imgui.begin_child("##preset_list", width=viz.app.button_w * 5, height=viz.app.font_size * 14)
            
    #         visible_presets = min(12, self.num_presets)
    #         for i in range(self.scroll_index, self.scroll_index + visible_presets):
    #             if i >= self.num_presets:
    #                 break
    #             imgui.begin_group()
    #             self.preset_checkbox(i)
    #             imgui.same_line()
    #             with imgui_utils.item_width(viz.app.button_w * 3.5):
    #                 dir_name_copy = self.dir_name[i]
    #                 _changed, dir_name = imgui_utils.input_text(f"##preset name {i}", dir_name_copy, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE, help_text="Set name for preset "+str(i))
    #                 if _changed:
    #                     self.dir_name[i] = str(dir_name)
    #                     self.paths[i] = str(self.tmp_path + '/' + self.dir_name[i])
    #                     if dir_name_copy in os.listdir(self.path):
    #                         os.rename(self.tmp_path + '/' + dir_name_copy,  self.paths[i])
    #             imgui.end_group()
            
    #         imgui.end_child()

    #         # 右侧：滚动按钮（如果需要）
    #         if self.num_presets > 12:
    #             imgui.same_line()
    #             imgui.begin_group()
    #             if imgui_utils.button("▲", width=viz.app.button_w):
    #                 self.scroll_index = max(0, self.scroll_index - 1)
    #             if imgui_utils.button("▼", width=viz.app.button_w):
    #                 self.scroll_index = min(self.num_presets - 12, self.scroll_index + 1)
    #             imgui.end_group()

    #         imgui.end_group()

    #         # 加载和保存按钮
    #         imgui.same_line()
    #         imgui.begin_group()
    #         if imgui_utils.button('Load##presets', width=viz.app.button_w):
    #             self.load(self.paths[np.where(self.active)].item())
    #         imgui.same_line()
    #         if imgui_utils.button("Save##presets", width=viz.app.button_w):
    #             if self.active.any():
    #                 self.save(self.paths[np.where(self.active)].item())
    #                 self.assigned[np.where(self.active)] = 0
    #         imgui.end_group()

    #         # 其余的UI元素保持不变
    #         with imgui_utils.item_width(viz.app.button_w * 2), imgui_utils.grayed_out(True):
    #             imgui.input_text("##preset_path", self.tmp_path, 256, imgui.INPUT_TEXT_READ_ONLY)
    #         imgui.same_line()
    #         _clicked, file_out = self.browser(self.viz.app.button_w)

    #         if _clicked:
    #             self.tmp_path = file_out[0]

    #         imgui.same_line()
    #         if imgui_utils.button('Recent...##presets', width=viz.app.button_w, enabled=(len(self.recent_paths) != 0)):
    #             imgui.open_popup('recent_preset_popup')
    #         if imgui.begin_popup('recent_preset_popup'):
    #             for pth in self.recent_paths:
    #                 clicked, _state = imgui.menu_item(pth)
    #                 if clicked:
    #                     self.open_path(pth)
    #             imgui.end_popup()
    #         imgui.same_line()

    #         if imgui_utils.button('Open##presets', width=viz.app.button_w):
    #             self.open_path(self.tmp_path)
    #             if not self.tmp_path in self.recent_paths:
    #                 self.recent_paths.append(self.tmp_path)
    #         imgui.same_line()
    #         _, self.use_osc = imgui.checkbox(f"Use OSC##load", self.use_osc)
    #         imgui.same_line()
    #         with imgui_utils.grayed_out(not (self.use_osc)):
    #             changed, osc_address = imgui_utils.input_text(f"##OSC_load",
    #                                                         self.osc_addresses,
    #                                                         256,
    #                                                         imgui.INPUT_TEXT_CHARS_NO_BLANK |
    #                                                         (imgui.INPUT_TEXT_READ_ONLY) * (not self.use_osc),
    #                                                         width=viz.app.font_size * 5,
    #                                                         help_text="Osc Address")
    #             if changed:
    #                 try:
    #                     viz.osc_dispatcher.unmap(f"/{self.osc_addresses}", self.osc_handler)
    #                     self.osc_addresses = osc_address
    #                 except:
    #                     print(f"{self.osc_addresses} is not mapped")
    #                 viz.osc_dispatcher.map(f"/{self.osc_addresses}", self.osc_handler)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            # 检查是否需要更新预设列表
            current_dirs = set(os.listdir(self.path))
            if set(self.dir_name) != current_dirs:
                self.open_path(self.path)  # 重新加载当前路径
            
            # 创建一个水平布局
            imgui.begin_group()

            # 预设列表（现在包含所有预设）
            imgui.begin_child("##preset_list", width=viz.app.button_w * 5, height=viz.app.font_size * 14, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
            
            for i in range(self.num_presets):
                imgui.begin_group()
                self.preset_checkbox(i)
                imgui.same_line()
                with imgui_utils.item_width(viz.app.button_w * 3.5):
                    dir_name_copy = self.dir_name[i]
                    _changed, dir_name = imgui_utils.input_text(f"##preset name {i}", dir_name_copy, 256, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE, help_text="Set name for preset "+str(i))
                    if _changed:
                        self.dir_name[i] = str(dir_name)
                        self.paths[i] = str(self.tmp_path + '/' + self.dir_name[i])
                        if dir_name_copy in os.listdir(self.path):
                            os.rename(self.tmp_path + '/' + dir_name_copy,  self.paths[i])
                imgui.end_group()
            
            imgui.end_child()

            imgui.end_group()

            # 加载和保存按钮
            imgui.same_line()
            imgui.begin_group()
            if imgui_utils.button('Load##presets', width=viz.app.button_w):
                self.load(self.paths[np.where(self.active)].item())
            imgui.same_line()
            if imgui_utils.button("Save##presets", width=viz.app.button_w):
                if self.active.any():
                    self.save(self.paths[np.where(self.active)].item())
                    self.assigned[np.where(self.active)] = 0
            imgui.same_line()
            if imgui_utils.button("New Folder##presets", width=viz.app.button_w):
                self.create_new_folder()
            imgui.end_group()
            # 其余的UI元素保持不变
            with imgui_utils.item_width(viz.app.button_w * 2), imgui_utils.grayed_out(True):
                imgui.input_text("##preset_path", self.tmp_path, 256, imgui.INPUT_TEXT_READ_ONLY)
            imgui.same_line()
            _clicked, file_out = self.browser(self.viz.app.button_w)

            if _clicked:
                self.tmp_path = file_out[0]

            imgui.same_line()
            if imgui_utils.button('Recent...##presets', width=viz.app.button_w, enabled=(len(self.recent_paths) != 0)):
                imgui.open_popup('recent_preset_popup')
            if imgui.begin_popup('recent_preset_popup'):
                for pth in self.recent_paths:
                    clicked, _state = imgui.menu_item(pth)
                    if clicked:
                        self.open_path(pth)
                imgui.end_popup()
            imgui.same_line()

            if imgui_utils.button('Open##presets', width=viz.app.button_w):
                self.open_path(self.tmp_path)
                if not self.tmp_path in self.recent_paths:
                    self.recent_paths.append(self.tmp_path)
            imgui.same_line()
            _, self.use_osc = imgui.checkbox(f"Use OSC##load", self.use_osc)
            imgui.same_line()
            with imgui_utils.grayed_out(not (self.use_osc)):
                changed, osc_address = imgui_utils.input_text(f"##OSC_load",
                                                            self.osc_addresses,
                                                            256,
                                                            imgui.INPUT_TEXT_CHARS_NO_BLANK |
                                                            (imgui.INPUT_TEXT_READ_ONLY) * (not self.use_osc),
                                                            width=viz.app.font_size * 5,
                                                            help_text="Osc Address")
                if changed:
                    try:
                        viz.osc_dispatcher.unmap(f"/{self.osc_addresses}", self.osc_handler)
                        self.osc_addresses = osc_address
                    except:
                        print(f"{self.osc_addresses} is not mapped")
                    viz.osc_dispatcher.map(f"/{self.osc_addresses}", self.osc_handler)

    def osc_handler(self, address, *args):
        value = str(args[-1])
        index = np.where(self.dir_name == value)
        self.active *= False
        self.active[index]=True
        self.load(self.paths[np.where(self.active)].item())