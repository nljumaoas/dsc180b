# Developing a LLM-Driven Multi-Agent Framework for Multimodal Translation
---

## Overview

This project features a manga translation pipeline that allows Japanese text sourced from context-heavy environments like manga to be seamlessly translated into English; the process is split into three main stages:

1. **Page Processing:** Uses techniques such as text segmentation and OCR to identify and order text and panel elements as well as extract text to be passed as additional context for the translation stage.
2. **Translation:** Leverages a multi-agent framework designed to produce high-quality, context-aware translations, maintaining internal consistency even across large translation projects. Improves upon conventional machine translation by utilizing visual context as well as multi-agent methods to preserve subleties such as variations in text sizing and localization of native idiosyncrasies.
3. **Typesetting:** Removes identified Japanese text and replaces it with the framework output, resulting in a translated English page that emphasizes readability while preserving the original style and format.

While this pipeline is capable of running locally, it is designed to have user interaction conducted through a local Flask frontend that uses API calls to connect to a remote server, allowing computation to be offloaded in order to minimize latency while maintaining an intuitive user experience. 

*(Code Checkpoint: the pipeline can currently be run locally, Flask/remote server integration will be implemented for a live demo at the capstone showcase)*

## Setup

1. After cloning this repository, clone the following repository in the same directory as well:
- https://github.com/sqbly/Manga-Text-Segmentation


2. Download the text segmentation model and move it into the `Manga-Text-Segmentation` directory.
```bash
wget -O model.pkl https://github.com/juvian/Manga-Text-Segmentation/releases/download/v1.0/fold.0.-.final.refined.model.2.pkl
```

3. Navigate to the `dsc180b` directory and install the required packages in a conda environment.
```conda
conda env create -f environment.yml
```

4.  Your directory should now look like this:
```
├── Manga-Text-Segmentation
│   └── model.pkl
├── dsc180b (This Repository)
│   └── <whatever important files we need>
└── <any other imports>
```
