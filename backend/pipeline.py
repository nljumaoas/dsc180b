from Translation_stage.translate_page import Translator
from processing_stage.processing import PageProcessor
from TypeSetting_V1.typesetting import TextBubbleTypesetter
import platform

class Pipeline():
    def __init__(self):
        self.page_processor = PageProcessor('../../Manga-Text-Segmentation/model.pkl')
        self.translator = Translator("gpt")
        self.typesetter = self.initialize_typesetter()

    def initialize_typesetter(self):
        if platform.system() == "Linux":
            font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        elif platform.system() == "Darwin":
            font = "/System/Library/Fonts/Supplemental/Arial.ttf"
        elif platform.system() == "Windows":
            font = "/C:/Windows/Fonts/arial.ttf"
        return TextBubbleTypesetter(font)

    def process_translate_typeset(self, image_path, output_path):
        # Process the page
        try:
            image_content, image_mask = self.page_processor.process_page(image_path, return_mask=True)
            print("Pipeline call completed process_page for text")
        except Exception as e:
            print(str(e))
            print("issue with pipeline process page")

        # Translate text
        image_path = image_content["image_paths"]['ja']
        texts = [d['text_ja'] for d in image_content['text']]
        panels_coordinates = image_content['frame']
        try:
            translation_result, chat_history = self.translator.generate_translation(texts, "English", image_path, panels_coordinates)
        except Exception as e:
            print(e)
            print("issue with pipeline generate_translation")

        # Compile results
        result = {"image": image_path, "text": []}
        try:
            for i, content in enumerate(image_content['text']):
                updated = content
                updated['text_translated'] = translation_result[i]
                result['text'].append(updated)
        except Exception as e:
            print(e)
            print("issue with organizing output of translation generated")

        # Typeset translated text
        try:
            self.typesetter.typeset_text_bubbles(result, output_path)
        except Exception as e:
            print(e)
            print("issue with pipeline typesetting")
        return output_path, chat_history

def main(image_path, output_path):
        '''
        # page processing stage
        processor = PageProcessor('../Manga-Text-Segmentation/model.pkl')
        image_content = processor.process_page(image_path)

        # translation stage
        image_path = image_content["image_paths"]['ja']
        texts = []
        for d in image_content['text']:
            texts.append(d['text_ja'])
        panels_coordinates = image_content['frame']
        target_language = "English"
        translator_system = Translator()
        translation_result = translator_system.generate_translation(texts, target_language, image_path, panels_coordinates)

        result = {"image": image_path, "text": []}
        for i in range(len(image_content['text'])):
            updated = image_content['text'][i]
            updated['text_translated'] = translation_result[i]
            result['text'].append(updated)


        # Typesetting Stage
        typesetter = TextBubbleTypesetter("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        typesetter.typeset_text_bubbles(result, "output_image02.jpg")
        '''
        pipeline_obj = Pipeline()
        output = pipeline_obj.process_translate_typeset(image_path, output_path)
        
if __name__ == "__main__":
    import argparse

    # Parse the configuration file from command-line arguments
    parser = argparse.ArgumentParser(description="Translate an image with text by passing in image path.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output file.")
    args = parser.parse_args()

    main(args.image_path, args.output_path)
