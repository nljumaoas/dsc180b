import time

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start() before stop().")
        elapsed_time = time.time() - self.start_time
        self.start_time = None  # Reset timer
        return elapsed_time

import time

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start() before stop().")
        elapsed_time = time.time() - self.start_time
        self.start_time = None  # Reset timer
        return np.round(elapsed_time, 2)

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
        