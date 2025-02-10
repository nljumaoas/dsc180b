from PIL import Image, ImageDraw, ImageFont
import textwrap

class TextBubbleTypesetter:
    def __init__(self, font_path, base_size=22):
        self.font_path = font_path
        self.base_size = base_size

    def format_text_for_bubble(self, text, max_width, max_height, font_size=None):
        """
        Format text by wrapping and adjusting font size if necessary.
        """
        font_size = font_size or self.base_size
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        font = ImageFont.truetype(self.font_path, font_size)

        # Step 1: Wrap text based on width
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
                    current_line = [word]  # For long words

        if current_line:
            lines.append(' '.join(current_line))

        # Step 2: Check total height with line spacing
        total_height = 0
        line_spacing = font_size * 0.3
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            total_height += (bbox[3] - bbox[1]) + line_spacing

        # Step 3: If height exceeds max, reduce font size recursively
        if total_height > max_height:
            min_size = int(font_size * 0.8)
            return self.format_text_for_bubble(text, max_width, max_height, min_size)

        return lines, font_size

    def typeset_text_bubbles(self, input_data, output_path):
        """
        Apply text formatting and typesetting on the provided image.
        input_data: Dictionary containing image path and text bubble data.
        output_path: Path to save the output image.
        """
        # Load image and drawing context
        image = Image.open(input_data["image"])
        draw = ImageDraw.Draw(image)

        # Process each text bubble
        for bubble in input_data["text"]:
            x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]
            draw.rectangle([(x, y), (x + w, y + h)], fill='white')  # Clear area

            # Get the translated text
            text = bubble["text_translated"]

            # Format text within the bubble
            lines, final_size = self.format_text_for_bubble(
                text,  # Translated text
                w * 0.9,  # 10% padding horizontally
                h * 0.9,  # 10% padding vertically
                self.base_size
            )

            font = ImageFont.truetype(self.font_path, final_size)

            # Calculate total height for vertical centering
            total_height = sum(
                [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines]
            ) + (len(lines) - 1) * (final_size * 0.3)

            # Calculate starting y-coordinate
            start_y = y + (h - total_height) // 2

            # Draw each line centered horizontally
            current_y = start_y
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = x + (w - text_width) // 2

                draw.text(
                    (text_x, current_y),
                    line,
                    font=font,
                    fill='black',
                    stroke_width=2,
                    stroke_fill='white'
                )
                current_y += (bbox[3] - bbox[1]) + (final_size * 0.3)

        # Save the modified image
        image.save(output_path)
        return output_path

# Example usage:
if __name__ == "__main__":
    typesetter = TextBubbleTypesetter("/System/Library/Fonts/Supplemental/Arial.ttf")
    input_data = {
        "image": "001.jpg",
        "text": [
            {"x": 172, "y": 194, "w": 126, "h": 229, "text_ja": "だからっ", "text_translated": "I'm telling you!!"},
            {"x": 692, "y": 519, "w": 98, "h": 184, "text_ja": "知らないって言ってるだろっ", "text_translated": "I don't know what you're talking about!"},
            {"x": 363, "y": 754, "w": 91, "h": 178, "text_ja": "そんな借金なんて!", "text_translated": "I don't owe you!"},
            {"x": 233, "y": 483, "w": 78, "h": 153, "text_ja": "そうは言ってもなぁ", "text_translated": "Well, I'm sorry..."},
            {"x": 97, "y": 855, "w": 106, "h": 198, "text_ja": "レーネ...", "text_translated": "Lene..."}
        ]
    }

    typesetter.typeset_text_bubbles(input_data, "output_image.jpg")