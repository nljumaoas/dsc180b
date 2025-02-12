from Translation_stage.translate_page import Translator
from processing_stage.processing import PageProcessor
from TypeSetting_V1.typesetting import TextBubbleTypesetter
import platform

def main(image_path):
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
    if platform.system() == "Linux":
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    elif platform.system() == "Darwin":
        font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    elif platform.system()=="Windows":
        font = "/C:/Windows/Fonts/arial.ttf"
    typesetter = TextBubbleTypesetter(font)
    typesetter.typeset_text_bubbles(result, "output_image.jpg")

if __name__ == "__main__":
    import argparse

    # Parse the configuration file from command-line arguments
    parser = argparse.ArgumentParser(description="Translate an image with text by passing in image path.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()

    main(args.image_path)
