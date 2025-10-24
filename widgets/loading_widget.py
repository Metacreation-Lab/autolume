import imgui
from utils.gui_utils import imgui_utils

class LoadingOverlayManager:
    """Enhanced loading overlay manager with two different render modes for different types of processes"""
    
    def __init__(self, app):
        self.app = app
        self.is_visible = False
        self.message = ""
        self.popup_id = "LoadingOverlay"
        
        # Progress tracking
        self.show_progress = False
        self.progress_current = 0
        self.progress_total = 0
        self.progress_percentage = 0
        self.current_file = ""
        
        # Render mode: 'simple' or 'detailed'
        self.render_mode = 'simple'
    
    def show_simple(self, message, show_progress=False):
        """Show simple loading overlay for basic processes like video extraction"""
        self.message = message
        self.is_visible = True
        self.show_progress = show_progress
        self.render_mode = 'simple'
        if not show_progress:
            self.reset_progress()
    
    def show_detailed(self, message, show_progress=False):
        """Show detailed loading overlay for complex processes like dataset processing"""
        self.message = message
        self.is_visible = True
        self.show_progress = show_progress
        self.render_mode = 'detailed'
        if not show_progress:
            self.reset_progress()
    
    def hide(self):
        """Hide the loading overlay"""
        self.is_visible = False
        self.message = ""
        self.reset_progress()
    
    def update_progress(self, current, total, current_file=""):
        """Update progress information"""
        self.progress_current = current
        self.progress_total = total
        self.progress_percentage = (current / total * 100) if total > 0 else 0
        self.current_file = current_file
    
    def reset_progress(self):
        """Reset progress information"""
        self.progress_current = 0
        self.progress_total = 0
        self.progress_percentage = 0
        self.current_file = ""
        self.show_progress = False
    
    def render_simple(self):
        """Render simple loading overlay for basic processes"""
        if not self.is_visible or not self.message:
            return
            
        imgui.open_popup(self.popup_id)
        
        # Position the popup in the center of the screen
        popup_width = self.app.content_width // 2.5
        popup_height = self.app.content_height // 2.5
        imgui.set_next_window_position(
            self.app.content_width / 2 - popup_width / 2, 
            self.app.content_height / 2 - popup_height / 2
        )
        imgui.set_next_window_size(popup_width, popup_height, imgui.ONCE)
        
        if imgui.begin_popup_modal(
            self.popup_id, 
            flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE
        )[0]:
            imgui.text(self.message)
            
            if self.show_progress and self.progress_total > 0:
                imgui.separator()
                imgui.spacing()
                
                # Display progress text
                imgui.text(f"Processing: {self.progress_current}/{self.progress_total} videos")
                if self.current_file:
                    imgui.text(f"Current file: {self.current_file}")
                
                # Progress bar
                progress_width = popup_width - 40
                imgui.progress_bar(self.progress_percentage / 100.0, (progress_width, 20))
                
                # Percentage text centered on progress bar
                text_width = imgui.calc_text_size(f"{self.progress_percentage:.1f}%")[0]
                imgui.set_cursor_pos_x((progress_width - text_width) / 2)
                imgui.text(f"{self.progress_percentage:.1f}%")
            
            imgui.end_popup()
    
    def render_detailed(self):
        """Render detailed loading overlay for complex processes with full progress details"""
        if not self.is_visible or not self.message:
            return
            
        imgui.open_popup(self.popup_id)
        
        # Position the popup in the center of the screen
        popup_width = self.app.content_width // 2.5
        popup_height = self.app.content_height // 2.5
        imgui.set_next_window_position(
            self.app.content_width / 2 - popup_width / 2, 
            self.app.content_height / 2 - popup_height / 2
        )
        imgui.set_next_window_size(popup_width, popup_height, imgui.ONCE)
        
        if imgui.begin_popup_modal(
            self.popup_id, 
            flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE
        )[0]:
            imgui.text(self.message)
            
            if self.show_progress and self.progress_total > 0:
                imgui.separator()
                imgui.spacing()
                
                # Display detailed progress information
                imgui.text(f"Processing: {self.progress_current}/{self.progress_total} images")
                if self.current_file:
                    imgui.text(f"Current file: {self.current_file}")
                
                # Progress bar
                progress_width = popup_width - 40
                imgui.progress_bar(self.progress_percentage / 100.0, (progress_width, 20))
                
                # Percentage text centered on progress bar
                text_width = imgui.calc_text_size(f"{self.progress_percentage:.1f}%")[0]
                imgui.set_cursor_pos_x((progress_width - text_width) / 2)
                imgui.text(f"{self.progress_percentage:.1f}%")
            
            imgui.end_popup()
    
    def render(self):
        """Main render method that delegates to the appropriate render function based on mode"""
        if self.render_mode == 'simple':
            self.render_simple()
        elif self.render_mode == 'detailed':
            self.render_detailed()
    
    def cleanup(self):
        """Clean up resources"""
        self.hide()