import time
from PIL import Image as Img
import numpy as np
import matplotlib as plt
from difflib import SequenceMatcher 

class Timer:
    def __init__(self):
        self.start_time = None
        self.lap_time = None

    def start(self):
        self.start_time = time.time()
        self.lap_start = time.time()

    def lap(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start() before check().")
        lap_time = time.time() - self.lap_start
        self.lap_start = time.time()
        return lap_time

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start() before stop().")
        elapsed_time = time.time() - self.start_time
        self.start_time = None  # Reset timer
        return elapsed_time

def oma_text_isolator(book, page):
    """
    book: tojime no siora would expect om_a[0]
    page: page number (int)
    """
    page_field = book['pages'][page]
    page_text = []

    for text_field in page_field['text']:
        page_text.append(text_field['text_ja'])

    return page_text

def oma_element_isolator(book, page, lang=False):
    """
    book: tojime no siora would expect om_a[0]
    page: page number (int)
    """
    page_field = book['pages'][page]
    panel_boxes = []
    text_boxes = []

    for panel_box in page_field['frame']:
        panel_boxes.append(panel_box)
    
    for text_box in page_field['text']:
        if lang == False:
            text_box = {
                'x': text_box['x'],
                'y': text_box['y'],
                'w': text_box['w'],
                'h': text_box['h']
            }

        text_boxes.append(text_box)

    return panel_boxes, text_boxes

def page_text_similarity(p1, p2):
    """
    Inputs should be equal-length lists of strings of text of the same language
    """
    if len(p1) != len(p2):
        return "Length mismatch!"
    
    text_n = len(p1)
    similarities = []
    for i in np.arange(text_n):
        similarities.append(SequenceMatcher(None, p1[i], p2[i]).ratio())

    average = sum(similarities) / text_n
    
    return similarities, average

def overlay_boxes(image_path, boxes, color='red', order=False):
    image = Img.open(image_path)
    width, height = image.size

    box_color = color

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image)

    centers = []

    # Draw bounding boxes
    for box in boxes:
        rect = patches.Rectangle(
            (box['x'], box['y']),
            box['w'], box['h'],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        center_x = box['x'] + box['w'] / 2
        center_y = box['y'] + box['h'] / 2
        centers.append((center_x, center_y))

    if order and len(centers) > 1:
        for i in range(len(centers) - 1):
            x_values = [centers[i][0], centers[i + 1][0]]
            y_values = [centers[i][1], centers[i + 1][1]]
            ax.plot(x_values, y_values, color=color, alpha=0.5, linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.show()
        