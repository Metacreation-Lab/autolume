from pathlib import Path
import pandas as pd
import cv2
import imgui
import webbrowser
from utils.gui_utils import gl_utils

DOCS_BASE_URL = "https://metacreation-lab.github.io/autolume"

class HelpIconWidget:
    """Reusable widget for displaying a help icon next to labels"""
    
    def __init__(self, icon_path="assets/help_icon.png"):
        self.icon_path = icon_path # currently unavailable
        self.help_icon_texture = None
        self._open_popup_id = None
        self._popup_position = None  
        self._load_icon()

    def load_help_texts(self, module_name):
        help_texts = {}
        help_urls = {}
        try:
            csv_path = Path("modules/help_texts.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if 'module' in df.columns:
                    df = df[df['module'] == module_name]
                for _, row in df.iterrows():
                    if row.get('key') and row.get('text'):
                        key = str(row['key']).strip()
                        text = str(row['text']).strip()
                        text = text.replace('\\n', '\n')
                        help_texts[key] = text
                        if pd.notna(row.get('url')) and str(row['url']).strip():
                            raw_url = str(row['url']).strip()
                            help_urls[key] = self._resolve_docs_url(raw_url)
        except Exception as e:
            print(f"Error loading help texts: {e}")

        return help_texts, help_urls

    @staticmethod
    def _resolve_docs_url(url_or_path):
        """Append DOCS_BASE_URL to the url_or_path"""
        if not url_or_path:
            return url_or_path
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return url_or_path
        base = DOCS_BASE_URL
        path = url_or_path if url_or_path.startswith("/") else "/" + url_or_path
        return base + path

    def _load_icon(self):
        """Load the help icon image as a texture"""
        try:
            if Path(self.icon_path).exists():
                help_img = cv2.imread(Path(self.icon_path), cv2.IMREAD_UNCHANGED)
                if help_img is not None:
                    self.help_icon_texture = gl_utils.Texture(
                        image=help_img,
                        width=help_img.shape[1],
                        height=help_img.shape[0],
                        channels=help_img.shape[2]
                    )
        except Exception as e:
            print(f"Error loading help icon: {e}")
            self.help_icon_texture = None
    
    def render(self, tooltip_text):
        """Render help icon with tooltip (no URL)"""
        if tooltip_text is None:
            return
        self._render_popup(tooltip_text, None)
    
    def render_with_url(self, tooltip_text, url, hyperlink_text="More Info"):
        """Render help icon with tooltip and hyperlink"""
        if tooltip_text is None:
            return
        if url:
            self._render_popup(tooltip_text, [(url, hyperlink_text)])
        else:
            self._render_popup(tooltip_text, None)
    
    def render_with_urls(self, tooltip_text, hyperlinks):
        """Render help icon with tooltip and multiple hyperlinks
        Args:
            tooltip_text: The help text to display
            hyperlinks: List of tuples (url, link_text) for multiple hyperlinks
        """
        if tooltip_text is None:
            return
        self._render_popup(tooltip_text, hyperlinks if hyperlinks else None)
    
    def _render_popup(self, tooltip_text, hyperlinks=None):
        """Internal method to render the help icon and popup
        Args:
            tooltip_text: The help text to display
            hyperlinks: List of tuples (url, link_text) for hyperlinks, or None
        """
        imgui.same_line()
        if self.help_icon_texture is not None:
            icon_size = imgui.get_font_size() 
            imgui.image(self.help_icon_texture.gl_id, icon_size, icon_size)
        else:
            imgui.text_disabled("(?)")
        
        popup_id = f"##HelpPopup_{abs(hash(tooltip_text))}"
        
        icon_hovered = imgui.is_item_hovered()
        icon_rect_min = imgui.get_item_rect_min()
        icon_rect_max = imgui.get_item_rect_max()
        
        is_current_popup = (self._open_popup_id == popup_id)
        
        if icon_hovered and not is_current_popup:
            self._open_popup_id = popup_id
            popup_x = icon_rect_min[0]
            popup_y = icon_rect_max[1] + 5
            self._popup_position = (popup_x, popup_y)  
            imgui.set_next_window_position(popup_x, popup_y)
            imgui.open_popup(popup_id)
        
        if imgui.begin_popup(popup_id):
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            self._render_formatted_text(tooltip_text)
            imgui.pop_text_wrap_pos()
            
            if hyperlinks:
                imgui.separator()
                for url, link_text in hyperlinks:
                    if url and link_text:
                        imgui.push_style_color(imgui.COLOR_TEXT, 0.4, 0.7, 1.0, 1.0)  
                        imgui.selectable(link_text, False, imgui.SELECTABLE_DONT_CLOSE_POPUPS)
                        if imgui.is_item_clicked(0):
                            webbrowser.open(url)
                        imgui.pop_style_color()
            
            mouse_pos = imgui.get_mouse_pos()
            
            buffer_size = 15 
            icon_rect_min_buffered = (icon_rect_min[0] - buffer_size, icon_rect_min[1] - buffer_size)
            icon_rect_max_buffered = (icon_rect_max[0] + buffer_size, icon_rect_max[1] + buffer_size)
            
            mouse_over_icon = (icon_rect_min_buffered[0] <= mouse_pos[0] <= icon_rect_max_buffered[0] and 
                             icon_rect_min_buffered[1] <= mouse_pos[1] <= icon_rect_max_buffered[1])
            
            popup_hovered = imgui.is_window_hovered()
            
            popup_size = imgui.get_window_size()
            
            mouse_near_popup = False
            if self._popup_position:
                popup_x, popup_y = self._popup_position
                popup_rect_min_buffered = (popup_x - buffer_size, popup_y - buffer_size)
                popup_rect_max_buffered = (popup_x + popup_size[0] + buffer_size, 
                                         popup_y + popup_size[1] + buffer_size)
                mouse_near_popup = (popup_rect_min_buffered[0] <= mouse_pos[0] <= popup_rect_max_buffered[0] and 
                                  popup_rect_min_buffered[1] <= mouse_pos[1] <= popup_rect_max_buffered[1])
            
            if not mouse_over_icon and not (popup_hovered or mouse_near_popup):
                self._open_popup_id = None
                self._popup_position = None
                imgui.close_current_popup()
            
            imgui.end_popup()
        elif is_current_popup:
            self._open_popup_id = None
            self._popup_position = None
    
    def _render_formatted_text(self, text):
        """Render text with keys (text before ':') highlighted"""
        if not text:
            return
        
        lines = text.split('\n')
        for line in lines:
            
            if ':' in line and not line.strip().startswith('http'):  
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ""
                
                imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.9, 1.0, 1.0)  
                imgui.text(key + ":")
                imgui.pop_style_color()
                
                if value:
                    imgui.same_line()
                    imgui.text(value)
            else:
                imgui.text(line)
    
    def cleanup(self):
        try:
            if self.help_icon_texture is not None:
                self.help_icon_texture.delete()
                self.help_icon_texture = None
        except Exception as e:
            print(f"Warning: Error during help icon cleanup: {e}")

