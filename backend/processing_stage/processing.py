from abc import ABC, abstractmethod
from typing import List
import sys
from PIL import Image as Img
from pyclustering.cluster.optics import optics
from manga_ocr import MangaOcr
import json
from fastai.vision import *
import skimage
from transformers import AutoModel
import numpy as np
import torch
import einops
import shapely
from processing_stage.eval_utilities import Timer

sys.path.insert(0, '../../Manga-Text-Segmentation/code/')

class TextExtractor(ABC):

    @abstractmethod
    def process_page(self, path_to_img: str) -> List[str]:
        pass 

class PageProcessor(TextExtractor):

    MIN_TEXT_SIZE = 15

    def __init__(self, path_to_model: str):
        file = path_to_model.split('/')[-1]
        path = path_to_model[0:-len(file)]

        self.text_segmentation_model = load_learner(path, file)
        self.mocr = MangaOcr()

        # use cpu since my gpu is ancient and doesn't want to cooperate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.magi = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).to(device)

    def process_page(self, path_to_img: str) -> List[str]:
        ## Segmentation and Clustering
        print(f"Processing {path_to_img}...")
        timer_all = Timer()
        timer_all.start()
        cluster_timer = Timer()
        cluster_timer.start()

        img = open_image(path_to_img)
        print("opened image for processing.")

        # Resize the image if sizes not divisible by 2
        size_w, size_h = img.size
        size_w -= size_w % 2
        size_h -= size_h % 2
        img = Image(img.data[:, 0:size_w, 0:size_h])
        print("resized the image")

        # Get pixelwise character position prediction
        pred = self.text_segmentation_model.predict(img)[0]
        print("Get pixelwise character position prediction")
        
        # replacing unpad_tensor functionality temporarily
        # mask = unpad_tensor(pred.px, img.px.shape)[0]
        mask = pred.px[:, 0:img.px.shape[1], 0:img.px.shape[2]][0]

        # Cluster text from the same speech bubbles
        sample = mask.nonzero().tolist()
        print("clustered text from same bubble")

        if len(sample) == 0:
            return []

        radius = 30
        neighbours = 2
        optics_instance = optics(sample, radius, neighbours)
        optics_instance.process() 
        print("optics_instance processed")
        clusters_ind = optics_instance.get_clusters()
        print("optics instance got clusters")
        clusters = [[sample[i] for i in indexes] for indexes in clusters_ind]
        

        ## Element Creation
        print(f"Clusters created! ({cluster_timer.stop()} s)")
        element_timer = Timer()
        element_timer.start()

        # Get a bounding box for each text cluster
        text_boxes = [self.bounding_box(c) for c in clusters]

        # Get a bounding box for each panel
        image_magi = self.read_image_as_np_array(path_to_img)

        with torch.no_grad():
            results = self.magi.predict_detections_and_associations([image_magi])

        panel_boxes = results[0]['panels']

        # Sort the text boxes in reading order
        sorted_text_indices = self.magi.sort_panels_and_text_bboxes_in_reading_order([panel_boxes], [text_boxes])[1][0]
        text_boxes_sorted = [text_boxes[i] for i in sorted_text_indices]

        print(f"Elements created! ({element_timer.stop()} s)")
        text_timer = Timer()
        text_timer.start()

        # Extract the text from each of the bounding boxes
        img_pil = Img.open(path_to_img)
        text_lines = []
        for i in range(len(text_boxes_sorted)):
            (x1, y1, x2, y2) = text_boxes_sorted[i]

            if min(abs(x2-x1), abs(y2-y1)) < self.MIN_TEXT_SIZE:
                text_boxes_sorted.pop(i)
                continue

            cropped_img = img_pil.crop((x1, y1, x2, y2))
            text = self.mocr(cropped_img)

            text_lines.append(text)

        print(f"Text extracted! ({text_timer.stop()} s)")


        ## Output Formatting (matches OpenMantra)

        output = {
            "image_paths": {
                "ja": path_to_img
            },
            "frame": [],
            "text": []
        }

        for panel in panel_boxes:
            output['frame'].append(PageProcessor.xyxy_to_xywh(panel))

        for text_box in text_boxes_sorted:
            output['text'].append(PageProcessor.xyxy_to_xywh(text_box))

        for text_num in np.arange(len(text_boxes_sorted)):
            output["text"][text_num]["text_ja"] = text_lines[text_num]

        print(f"Process complete! ({timer_all.stop()} s)")

        return output

    @classmethod
    def xyxy_to_xywh(cls, xyxy):
        """
        The name is a work in progress okay...
        """
        x1, y1, x2, y2 = xyxy

        xywh = {
            'x': round(x1, 1),
            'y': round(y1, 1),
            'w': round(x2 - x1, 1),
            'h': round(y2 - y1, 1)
        }

        return xywh

    @classmethod
    def bounding_box(cls, points):
        y_coordinates, x_coordinates = zip(*points)

        return (min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates))
    
    @classmethod
    def read_image_as_np_array(cls, image_path):
        with open(image_path, "rb") as file:
            image = Img.open(file).convert("L").convert("RGB")
            image = np.array(image)
        return image
