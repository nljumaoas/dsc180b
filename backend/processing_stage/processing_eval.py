import sys
sys.path.insert(0, '../../../Manga-Text-Segmentation/code/')

from processing import PageProcessor
import json

eval_outputs = {}
eval_records = {}

with open('open_mantra/annotation.json') as json_file:
    oma = json.load(json_file)
    print("Annotations loaded")

oma_dict = {}

for book_dict in oma:
    oma_dict[book_dict['book_title']] = book_dict['pages']

eval_processor = PageProcessor("../../../Manga-Text-Segmentation/model.pkl")
print("Processor created")

for book, pages in oma_dict.items():
    for page in pages:
        page_id = f"{book}_{page['page_index']}"
        image_path = f"open_mantra/{page['image_paths']['ja']}"
        oma_panels = page['frame']
        oma_text = page['text']
        
        print(f"Processing {page_id}")
        
        try:
            eval_output = eval_processor.process_page(image_path)
            eval_panels = eval_output['frame']
            eval_text = eval_output['text']
            times = eval_output['times']

            eval_records[page_id] = {
                'oma_panels': oma_panels,
                'oma_text': oma_text,
                'eval_panels': eval_panels,
                'eval_text': eval_text,
                'times': times
            }
        except Exception as e:
            print('Error processing ', page_id)
            eval_records[page_id] = {
                'oma_panels': oma_panels,
                'oma_text': oma_text,
                'eval_panels': 'processing error',
                'eval_text': 'processing error',
                'times': 'processing error'
            }

print('Processed dataset')

with open("eval_records_3.json", "w") as file:
    json.dump(eval_records, file, indent=4)