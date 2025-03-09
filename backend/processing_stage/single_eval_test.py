import sys
sys.path.insert(0, '../../../Manga-Text-Segmentation/code/')

from processing import PageProcessor

eval_processor = PageProcessor("../../../Manga-Text-Segmentation/model.pkl")

image_path = "frieren.png"

processed_image, processed_mask = eval_processor.process_page(image_path, return_mask=True)

print(processed_image)
print(processed_mask)