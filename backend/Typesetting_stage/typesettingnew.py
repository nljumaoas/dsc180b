import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import os

class TextBubbleTypesetter:
    """
    Class for typesetting translated text into manga speech bubbles
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
    
    def _mask_text(self, image_path, mask_tensor, input_data):
        """
        Mask text in manga using both pixel mask and speech bubble coordinates.
        
        Parameters:
        -----------
        image_path : str
            Path to the manga image
        mask_tensor : torch.Tensor
            Binary mask tensor where 1 indicates text pixels
        input_data : list of dict
            List of speech bubble data with keys 'x', 'y', 'w', 'h', 'text_ja', 'text_en'
            
        Returns:
        --------
        PIL.Image
            The masked image with text removed
        """
        # Open image
        #img = Image.open(image_path).convert("RGB")
        #img_array = np.array(img)
        input_img = Image.open(image_path).convert("RGB")
        mask_height, mask_width = mask_tensor.shape
        img = input_img.resize((mask_width, mask_height))
        img_array = np.array(img)
        
        # Convert mask tensor to numpy if needed
        if isinstance(mask_tensor, torch.Tensor):
            mask = mask_tensor.cpu().numpy()
        else:
            mask = mask_tensor
        
        # Create output image
        result_img = img_array.copy()
        
        # Process each bubble
        for bubble in input_data:
            x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]
            
            # Create a mask for this bubble region
            bubble_region = np.zeros_like(mask)
            bubble_region[y:y+h, x:x+w] = 1
            
            # Get the text mask within this bubble
            text_in_bubble = np.logical_and(mask == 1, bubble_region == 1)
            
            # Skip if no text in this bubble
            if np.sum(text_in_bubble) == 0:
                continue
            
            # Find connected components (text clusters)
            text_img = text_in_bubble.astype(np.uint8) * 255
            
            # Apply dilation to connect nearby text
            kernel = np.ones((3, 3), np.uint8)
            dilated_text = cv2.dilate(text_img, kernel, iterations=1)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_text, connectivity=8)
            
            # Process each text component
            for i in range(1, num_labels):  # Skip background (0)
                # Get component bounding box
                cx = stats[i, cv2.CC_STAT_LEFT]
                cy = stats[i, cv2.CC_STAT_TOP]
                cw = stats[i, cv2.CC_STAT_WIDTH]
                ch = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Expand with margin
                cx_expanded = max(0, cx - self.margin)
                cy_expanded = max(0, cy - self.margin)
                cw_expanded = cw + 2 * self.margin
                ch_expanded = ch + 2 * self.margin
                
                # Create a mask for the expanded text region
                text_region = np.zeros_like(mask)
                text_region[cy_expanded:cy_expanded+ch_expanded, cx_expanded:cx_expanded+cw_expanded] = 1
                
                # Apply white color to the text region
                result_img[text_region == 1] = [255, 255, 255]
        
        # Create final masked image
        final_img = Image.fromarray(result_img)
        
        return final_img
    
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
        if total_height > max_height and self.base_font_size > 10:
            # Calculate new size based on ratio
            ratio = max_height / total_height
            new_size = max(int(self.base_font_size * ratio * 0.95), 10)  # Minimum size of 10
            
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
        Add translated text to the image using the bubble coordinates.
        
        Parameters:
        -----------
        image : PIL.Image
            The masked image
        input_data : list of dict
            List of bubble data with coordinates and translations
            
        Returns:
        --------
        PIL.Image
            Image with translated text added
        """
        draw = ImageDraw.Draw(image)
        
        # Process each text area
        for bubble in input_data:
            # Get bubble coordinates and translated text
            x, y = bubble["x"], bubble["y"]
            w, h = bubble["w"], bubble["h"]
            
            # Determine which text field to use (text_en or text_translated)
            if "text_translated" in bubble:
                text = bubble["text_translated"]
            elif "text_en" in bubble:
                text = bubble["text_en"]
            else:
                continue  # Skip if no translation is found
            
            # Format text to fit in the bubble
            lines, final_size = self._format_text_for_bubble(
                text,
                w * 0.9,  # 90% of bubble width (10% margin)
                h * 0.9,  # 90% of bubble height (10% margin)
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
                text_x = x + (w - text_width) // 2
                
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
        
        Parameters:
        -----------
        result : dict
            Dictionary containing image path and text data
            Format: {"image": image_path, "text": [bubble_data, ...]}
            where bubble_data contains x, y, w, h, text_ja, and text_translated keys
        output_path : str
            Path to save the typeset image
        image_mask : torch.Tensor or np.ndarray, optional
            Binary mask indicating text pixels
            
        Returns:
        --------
        str
            Path to the typeset image
        """
        # Extract image path and bubble data
        image_path = result["image"]
        bubble_data = result["text"]
        
        if image_mask is not None:
            # Step 1: Mask the original text
            masked_img = self._mask_text(image_path, image_mask, bubble_data)
        else:
            # If no mask provided, just load the image
            masked_img = Image.open(image_path).convert("RGB")
        
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
