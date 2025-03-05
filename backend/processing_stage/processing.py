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
from eval_utilities import Timer

sys.path.insert(0, '../../Manga-Text-Segmentation/code/')

class TextExtractor(ABC):

    @abstractmethod
    def process_page(self, path_to_img: str):
        pass 



    MIN_TEXT_SIZE = 15

    def __init__(self, path_to_model: str):
        file = path_to_model.split('/')[-1]
        path = path_to_model[0:-len(file)]

        self.text_segmentation_model = load_learner(path, file)
        self.mocr = MangaOcr()

        # use cpu since my gpu is ancient and doesn't want to cooperate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.magi = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).to(device)

    def process_page(self, path_to_img: str, apply_filter=True, include_mask=False):
        """
        Inputs:
        - path_to_img: path to a single untranslated manga page
        - filtered: if True, processes text using speech bubble filter
            ** set to false if the input page has color in desired text fields, e.g. title page
        - include_mask: if True, returns the mask as an additional output

        Outputs:
        - output_dict: dictionary containing the following information:
            - image_paths: the path to the raw, unprocessed image
            - frame: list of panel box coordinates, in xywh form
            - text: list of text box coordinates, along with the extracted text from each box
            ** both frame and text are in reading order
        - output_mask (optional): PyTorch tensor mask of the page
        """
        ## Segmentation and Clustering
        timer_all = Timer()
        timer_all.start()
        cluster_timer = Timer()
        cluster_timer.start()

        img = open_image(path_to_img)

        # Resize the image if sizes not divisible by 2
        size_w, size_h = img.size
        size_w -= size_w % 2
        size_h -= size_h % 2
        img = Image(img.data[:, 0:size_w, 0:size_h])

        # Get pixelwise character position prediction
        pred = self.text_segmentation_model.predict(img)[0]
        
        # replacing unpad_tensor functionality temporarily
        # mask = unpad_tensor(pred.px, img.px.shape)[0]
        mask = pred.px[:, 0:img.px.shape[1], 0:img.px.shape[2]][0]

        # Cluster text from the same speech bubbles
        sample = mask.nonzero().tolist()

        if len(sample) == 0:
            return []

        radius = 22.5
        neighbours = 2
        optics_instance = optics(sample, radius, neighbours)
        optics_instance.process() 
        clusters_ind = optics_instance.get_clusters()
        clusters = [[sample[i] for i in indexes] for indexes in clusters_ind]
        if apply_filter:
            clusters = self.speech_filter(path_to_img, mask, clusters)

        ## Element Creation
        cluster_time = cluster_timer.stop()
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

        element_time = element_timer.stop()
        text_timer = Timer()
        text_timer.start()

        # Extract the text from each of the bounding boxes
        img_pil = Img.open(path_to_img)
        text_boxes_filtered = []
        text_lines = []

        for text_box in text_boxes_sorted:
            (x1, y1, x2, y2) = text_box
            if min(abs(x2 - x1), abs(y2 - y1)) < self.MIN_TEXT_SIZE:
                continue
            cropped_img = img_pil.crop((x1, y1, x2, y2))
            try:
                text = self.mocr(cropped_img)
            except Exception as e:
                print(f"OCR failed for box ({x1}, {y1}, {x2}, {y2}): {e}")
                continue
            text_boxes_filtered.append(text_box)
            text_lines.append(text)

        text_time = text_timer.stop()


        ## Output Formatting (matches OpenMantra, plus evaluatory metrics)

        output_dict = {
            "image_paths": {
                "ja": path_to_img
            },
            "frame": [],
            "text": []
        }

        for panel in panel_boxes:
            output['frame'].append(self.xyxy_to_xywh(panel))

        for text_box in text_boxes_sorted:
            output['text'].append(self.xyxy_to_xywh(text_box))

        for text_num in np.arange(len(text_boxes_filtered)):
            output["text"][text_num]["text_ja"] = text_lines[text_num]

        process_time = timer_all.stop()
        
        output['times'] = {
            'cluster': cluster_time,
            'element': element_time,
            'text': text_time,
            'process': process_time
        }

        return output

    @classmethod
    def speech_filter(cls, image_path, mask, clusters, threshold=220, white_ratio=0.8, cushion=10):
        """
        Filters clusters based on whether they are inside speech bubbles.
        
        Args:
            image_path (str): Path to the original image.
            mask (np.ndarray): Binary mask where text regions are marked (e.g., 1 for text, 0 for background).
            clusters (list): List of clusters, where each cluster is a list of (y, x) coordinates.
            threshold (int): Pixel value above which a pixel is considered "white".
            white_ratio (float): Proportion of white pixels required to classify the background as white.
            cushion (int): Padding added around the bounding box to capture the background.
        
        Returns:
            list: Clusters that are identified as being inside speech bubbles.
        """
        # Load the image and convert it to an RGB array
        img = Img.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Modify the identified text regions in the image to white
        img_array[mask == 1] = [255, 255, 255]
        
        # Create a copy of the image for analysis
        modified_img_array = img_array.copy()
        
        filtered_clusters = []
        
        for cluster in clusters:
            # Get the bounding box of the cluster
            y_coords, x_coords = zip(*cluster)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add cushion to the bounding box to capture the background
            x_min_cushioned = max(0, x_min - cushion)
            x_max_cushioned = min(img.width, x_max + cushion)
            y_min_cushioned = max(0, y_min - cushion)
            y_max_cushioned = min(img.height, y_max + cushion)
            
            # Extract the region around the cluster
            region = modified_img_array[y_min_cushioned:y_max_cushioned, x_min_cushioned:x_max_cushioned]
            
            # Analyze the background pixels
            white_pixels = np.all(region > threshold, axis=-1)  # True for pixels where R, G, B > threshold
            white_pixel_count = np.sum(white_pixels)
            total_pixels = region.shape[0] * region.shape[1]
            
            # Check if the proportion of white pixels meets the threshold
            if white_pixel_count / total_pixels >= white_ratio:
                filtered_clusters.append(cluster)

        if len(filtered_clusters) == 0:
            return clusters

        
        return filtered_clusters
    
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

class PageProcessor(TextExtractor):

    MIN_TEXT_SIZE = 15

    def __init__(self, path_to_model: str):
        file = path_to_model.split('/')[-1]
        path = path_to_model[0:-len(file)]
        self.text_segmentation_model = load_learner(path, file)
        self.mocr = MangaOcr()
        device = "cuda" if torch.cuda.is_available() else "cpu" # for testing locally without GPU
        self.magi = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).to(device)

    def process_page(self, path_to_img: str, apply_filter=True, return_mask=False):
        """
        Processes raw manga page by identifying panel and text elements as well as extracting text.

        Inputs:
        - path_to_img: path to a single untranslated manga page
        - apply_filter: if True, processes text using speech bubble filter
            ** set to false if the input page has color in desired text fields, e.g. title page
        - return_mask: if True, returns the mask as an additional output

        Outputs:
        - output: dictionary containing the following information:
            - image_paths: the path to the raw, unprocessed image
            - frame: list of panel box coordinates, in xywh form
            - text: list of text box coordinates, along with the extracted text from each box
            ** both frame and text are in reading order
        - mask (optional): PyTorch tensor mask of the page
        """
        ## Segmentation and Clustering
        timer_all = Timer()
        timer_all.start()
        cluster_timer = Timer()
        cluster_timer.start()

        print('process start')

        img = open_image(path_to_img)

        print(f'opened image: {cluster_timer.lap()}')

        # resizes the image if sizes not divisible by 2
        size_w, size_h = img.size
        size_w -= size_w % 2
        size_h -= size_h % 2
        img = Image(img.data[:, 0:size_w, 0:size_h])

        # gets pixelwise character position prediction
        pred = self.text_segmentation_model.predict(img)[0]

        print(f'performed text segmentation: {cluster_timer.lap()}')
        
        # unpads tensor
        mask = pred.px[:, 0:img.px.shape[1], 0:img.px.shape[2]][0]

        # clusters text
        sample = mask.nonzero().tolist()

        if len(sample) == 0:
            return []

        radius = 30
        neighbours = 2
        optics_instance = optics(sample, radius, neighbours)
        optics_instance.process() 
        clusters_ind = optics_instance.get_clusters()
        clusters = [[sample[i] for i in indexes] for indexes in clusters_ind]

        print(f'formed clusters: {cluster_timer.lap()}')

        # applies speech bubble filter
        if apply_filter:
            clusters = self.speech_filter(path_to_img, mask, clusters)
            print(f'filtered clusters: {cluster_timer.lap()}')

            # converts mask to filtered version for typesetting
            if return_mask:
                mask = mask.zero_()
                for cluster in clusters:
                    for y,x in cluster:
                        mask[y, x] = 1 
                

        ## Element Creation
        cluster_time = cluster_timer.stop()
        element_timer = Timer()
        element_timer.start()

        # creates a bounding box for each text cluster
        text_boxes = [self.bounding_box(c) for c in clusters]

        # creates a bounding box for each panel
        image_magi = self.read_image_as_np_array(path_to_img)
        
        # sorts panels and text boxes into reading order
        with torch.no_grad():
            results = self.magi.predict_detections_and_associations([image_magi])
        panel_boxes = results[0]['panels']

        sorted_text_indices = self.magi.sort_panels_and_text_bboxes_in_reading_order([panel_boxes], [text_boxes])[1][0]
        text_boxes_sorted = [text_boxes[i] for i in sorted_text_indices]

        element_time = element_timer.stop()
        text_timer = Timer()
        text_timer.start()

        # extracts the text from each text box
        img_pil = Img.open(path_to_img)
        text_lines = []
        for text_box in text_boxes_sorted:
            (x1, y1, x2, y2) = text_box

            # excludes text boxes below minimum size to reduce element misidentification
            if min(abs(x2-x1), abs(y2-y1)) < self.MIN_TEXT_SIZE:
                continue

            cropped_img = img_pil.crop((x1, y1, x2, y2))
            text = self.mocr(cropped_img)

            text_lines.append(text)

        text_time = text_timer.stop()


        ## Output Formatting (matches OpenMantra, plus evaluatory metrics)

        output = {
            "image_paths": {
                "ja": path_to_img
            },
            "frame": [],
            "text": []
        }

        for panel in panel_boxes:
            output['frame'].append(self.xyxy_to_xywh(panel))

        for text_box in text_boxes_sorted:
            output['text'].append(self.xyxy_to_xywh(text_box))

        for text_num in np.arange(len(text_boxes_sorted)):
            output["text"][text_num]["text_ja"] = text_lines[text_num]

        process_time = timer_all.stop()
        
        output['times'] = {
            'cluster': cluster_time,
            'element': element_time,
            'text': text_time,
            'process': process_time
        }

        if return_mask:
            return output, mask
        return output

    @classmethod
    def speech_filter(cls, image_path, mask, clusters, threshold=250, white_ratio=0.85, cushion=10):
        """
        Filters clusters based on whether they are inside speech bubbles.
        
        Args:
            image_path (str): Path to the original image.
            mask (np.ndarray): Binary mask where text regions are marked (e.g., 1 for text, 0 for background).
            clusters (list): List of clusters, where each cluster is a list of (y, x) coordinates.
            threshold (int): Pixel value above which a pixel is considered "white".
            white_ratio (float): Proportion of white pixels required to classify the background as white.
            cushion (int): Padding added around the bounding box to capture the background.
        
        Returns:
            list: Clusters that are identified as being inside speech bubbles.
        """
        # Load the image and convert it to an RGB array
        input = Img.open(image_path)
        img_array = np.array(input)
        
        # Modify the identified text regions in the image to white
        img_array[mask == 1] = [255, 255, 255]
        
        # Create a copy of the image for analysis
        modified_img_array = img_array.copy()
        
        filtered_clusters = []
        
        for cluster in clusters:
            # Get the bounding box of the cluster
            y_coords, x_coords = zip(*cluster)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add cushion to the bounding box to capture the background
            x_min_cushioned = max(0, x_min - cushion)
            x_max_cushioned = min(input.width, x_max + cushion)
            y_min_cushioned = max(0, y_min - cushion)
            y_max_cushioned = min(input.height, y_max + cushion)
            
            # Extract the region around the cluster
            region = modified_img_array[y_min_cushioned:y_max_cushioned, x_min_cushioned:x_max_cushioned]
            
            # Analyze the background pixels
            white_pixels = np.all(region > threshold, axis=-1)  # True for pixels where R, G, B > threshold
            white_pixel_count = np.sum(white_pixels)
            total_pixels = region.shape[0] * region.shape[1]
            
            # Check if the proportion of white pixels meets the threshold
            if white_pixel_count / total_pixels >= white_ratio:
                filtered_clusters.append(cluster)
        
        return filtered_clusters
    
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