import csv
import logging

import imgui
import webbrowser
from utils.resource_paths import get_version, resource_path

logger = logging.getLogger(__name__)

DOCS_BASE_URL = f"https://metacreation-lab.github.io/autolume/{get_version()}"

class HelpIconWidget:
    """Reusable widget for displaying a help icon next to labels"""
    
    def __init__(self):
        self._open_popup_id = None
        self._popup_position = None

    def load_help_texts(self, module_name):
        help_texts = {}
        help_urls = {}
        try:
            csv_path = resource_path("modules", "help_texts.csv")
            if csv_path.exists():
                with open(csv_path, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        if 'module' in row and row['module'] != module_name:
                            continue
                        key = (row.get('key') or '').strip()
                        if not key:
                            continue
                        text = (row.get('text') or '').strip().replace('\\n', '\n')
                        if text:
                            help_texts[key] = text
                        raw_url = (row.get('url') or '').strip()
                        if raw_url:
                            help_urls[key] = self._resolve_docs_url(raw_url)
        except Exception as e:
            logger.warning("Error loading help texts: %s", e)

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

    def render(self, tooltip_text, url=None, hyperlink_text="Read More",
               hyperlinks=None, align_right=False):
        """Render help icon with tooltip and optional hyperlinks.

        Args:
            tooltip_text: The help text to display
            url: Single hyperlink URL (ignored when hyperlinks is given)
            hyperlink_text: Link text for url
            hyperlinks: List of tuples (url, link_text) for multiple hyperlinks
            align_right: Place the icon at the right edge of the content region
                instead of inline after the previous item
        """
        if tooltip_text is None:
            return
        if align_right:
            style = imgui.get_style()
            imgui.same_line()
            target_x = (imgui.get_content_region_max()[0] - imgui.calc_text_size("(?)").x
                        - style.item_spacing[0])
            spacing = target_x - imgui.get_cursor_pos_x() - style.item_spacing[0]
            imgui.dummy(max(spacing, 0), 0)
        if not hyperlinks and url:
            hyperlinks = [(url, hyperlink_text)]
        self._render_popup(tooltip_text, hyperlinks or None)

    def _render_popup(self, tooltip_text, hyperlinks=None):
        """Internal method to render the help icon and popup
        Args:
            tooltip_text: The help text to display
            hyperlinks: List of tuples (url, link_text) for hyperlinks, or None
        """
        imgui.same_line()
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
        """No-op retained for API compatibility (no GL resources to release)."""
        pass

