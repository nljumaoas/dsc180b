import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import os

class TextBubbleTypesetter:
    """
    Class for typesetting translated text into manga speech bubbles.
    
    This class handles the replacement of original text in manga with translated text.
    It uses a simple rectangular masking approach to remove original text and then
    adds translated text with proper formatting and alignment within each bubble.
    The text boxes are slightly expanded horizontally to provide better text fitting.
    """
    
    def __init__(self, font_path):
        """
        Initialize the typesetter with a font path.
        
        Parameters:
        -----------
        font_path : str
            Path to the font file.
        """
        self.font_path = font_path
        self.base_font_size = 22
        self.margin = 5
    
    def _mask_text(self, image_path, input_data):
        """
        Mask text in manga by drawing white rectangles over the bubble areas.
        
        Parameters:
        -----------
        image_path : str
            Path to the manga image
        input_data : list of dict
            List of speech bubble data with keys 'x', 'y', 'w', 'h', 'text_ja', 'text_en'
                
        Returns:
        --------
        PIL.Image
            The masked image with original text removed via white rectangles
        """
        # Open image
        img = Image.open(image_path).convert("RGB")
        
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        
        # Process each bubble
        for bubble in input_data:
            x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]
            # Draw a white rectangle to mask the area
            draw.rectangle([(x, y), (x + w, y + h)], fill='white')
        
        return img
    
    def _format_text_for_bubble(self, text, max_width, max_height):
        """
        Format text to fit within a bubble with line wrapping and font size adjustment.
        
        Parameters:
        -----------
        text : str
            Text to format
        max_width : int
            Maximum width available for text
        max_height : int
            Maximum height available for text
            
        Returns:
        --------
        tuple
            (lines, font_size) where lines is a list of text lines and font_size is the final size
        """
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        font = ImageFont.truetype(self.font_path, self.base_font_size)
        
        # 1. First attempt simple text wrapping
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # If the word is too long, consider breaking the word or adjusting the font size
                    current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # 2. Check if the total height fits
        total_height = 0
        line_spacing = self.base_font_size * 0.3  # 30% line spacing
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            total_height += (bbox[3] - bbox[1]) + line_spacing

        # 3. If height exceeds, consider reducing font size appropriately
        if total_height > max_height and self.base_font_size > 14:
            # Calculate new size based on ratio
            ratio = max_height / total_height
            new_size = max(int(self.base_font_size * ratio * 0.95), 14)  # Minimum size of 10
            
            # Save the original base size
            original_base_size = self.base_font_size
            
            # Temporarily change base size
            self.base_font_size = new_size
            
            # Recursively try with smaller font
            result = self._format_text_for_bubble(text, max_width, max_height)
            
            # Restore the original base size
            self.base_font_size = original_base_size
            
            return result
        
        return lines, self.base_font_size
    
    def _add_translated_text(self, image, input_data):
        """
        Add translated text to the image using bubble coordinates with horizontal expansion.
        
        This method centers text within each bubble and applies a symmetric horizontal
        expansion of 16 pixels (8 pixels on each side) to give text more room to fit.
        Text is formatted to fit the expanded width while maintaining the original height.
        
        Parameters:
        -----------
        image : PIL.Image
            The masked image
        input_data : list of dict
            List of bubble data with coordinates and translations
                
        Returns:
        --------
        PIL.Image
            Image with translated text added and centered in expanded bubbles
        """
        draw = ImageDraw.Draw(image)
        
        # Define a constant width expansion (in pixels)
        constant_width_expansion = 16  # Add 20 pixels total (10 on each side)
        
        # Process each text area
        for bubble in input_data:
            # Get bubble coordinates and translated text
            x, y = bubble["x"], bubble["y"]
            w, h = bubble["w"], bubble["h"]
            
            # Calculate expanded width with constant expansion
            expanded_width = w + constant_width_expansion
            
            # Calculate the new x-coordinate to maintain center point
            original_center_x = x + (w // 2)
            new_x = original_center_x - (expanded_width // 2)
            
            # Ensure expanded bubble stays within image bounds
            img_width, img_height = image.size
            if new_x < 0:
                new_x = 0
            if new_x + expanded_width > img_width:
                expanded_width = img_width - new_x
            
            # Determine which text field to use (text_en or text_translated)
            if "text_translated" in bubble:
                text = bubble["text_translated"]
            elif "text_en" in bubble:
                text = bubble["text_en"]
            else:
                continue  # Skip if no translation is found
            
            # Format text to fit in the expanded bubble
            lines, final_size = self._format_text_for_bubble(
                text,
                expanded_width * 0.9,  # 90% of expanded bubble width
                h * 0.9,               # 90% of original bubble height
            )
            
            # Load the font at the final determined size
            font = ImageFont.truetype(self.font_path, final_size)
            
            # Calculate total text height (for vertical centering)
            total_height = 0
            line_spacing = final_size * 0.3
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                total_height += (bbox[3] - bbox[1]) + line_spacing
            total_height -= line_spacing  # Subtract the line spacing added for the last line
            
            # Calculate the starting y-coordinate to vertically center the text
            start_y = y + (h - total_height) // 2
            
            # Draw each line of text
            current_y = start_y
            for line in lines:
                # Get the width of the current line for horizontal centering
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                
                # Center text in the expanded width
                text_x = new_x + (expanded_width - text_width) // 2
                
                # Draw text (with outline)
                draw.text(
                    (text_x, current_y),
                    line,
                    font=font,
                    fill='black',
                    stroke_width=2,
                    stroke_fill='white'
                )
                
                # Update the y-coordinate for the next line
                current_y += (bbox[3] - bbox[1]) + line_spacing
        
        return image
    
    def typeset_text_bubbles(self, result, output_path, image_mask=None):
        """
        Main method to typeset translated text into manga speech bubbles.
        
        This method first masks the original text with white rectangles based on
        bubble coordinates, then adds translated text with appropriate formatting.
        The image_mask parameter is kept for backward compatibility but is no longer used.
        
        Parameters:
        -----------
        result : dict
            Dictionary containing image path and text data
            Format: {"image": image_path, "text": [bubble_data, ...]}
            where bubble_data contains x, y, w, h, text_ja, and text_translated keys
        output_path : str
            Path to save the typeset image
        image_mask : torch.Tensor or np.ndarray, optional
            Binary mask indicating text pixels (no longer used, kept for compatibility)
            
        Returns:
        --------
        str
            Path to the typeset image
        """
        # Extract image path and bubble data
        image_path = result["image"]
        bubble_data = result["text"]
        
        # if image_mask is not None:
        #     # Step 1: Mask the original text
        #     masked_img = self._mask_text(image_path, image_mask, bubble_data)
        # else:
        #     # If no mask provided, just load the image
        #     masked_img = Image.open(image_path).convert("RGB")
        
        # Step 1: Mask the original text (new method without using mask)
        masked_img = self._mask_text(image_path, bubble_data)
        
        # Step 2: Add translated text using bubble coordinates
        translated_img = self._add_translated_text(masked_img, bubble_data)
        
        # Save the final typeset image
        translated_img.save(output_path)
        print(f"Typeset image saved to: {output_path}")
        
        return output_path
    
    def set_font_size(self, size):
        """
        Set the base font size.
        
        Parameters:
        -----------
        size : int
            Base font size
        """
        self.base_font_size = size
    
    def set_margin(self, margin):
        """
        Set the margin for text masking.
        
        Parameters:
        -----------
        margin : int
            Margin to add around text regions
        """
        self.margin = margin
