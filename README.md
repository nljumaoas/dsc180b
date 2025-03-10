# Developing a LLM-Driven Multi-Agent Framework for Multimodal Translation
---

## Overview

This project features a manga translation pipeline that allows Japanese text sourced from context-heavy environments like manga to be seamlessly translated into English; the process is split into three main stages:

1. **Page Processing:** Uses techniques such as text segmentation and OCR to identify and order text and panel elements as well as extract text to be passed as additional context for the translation stage.
2. **Translation:** Leverages a multi-agent framework designed to produce high-quality, context-aware translations, maintaining internal consistency even across large translation projects. Improves upon conventional machine translation by utilizing visual context as well as multi-agent methods to preserve subleties such as variations in text sizing and localization of native idiosyncrasies.
3. **Typesetting:** Removes identified Japanese text and replaces it with the framework output, resulting in a translated English page that emphasizes readability while preserving the original style and format.

While this pipeline is capable of running locally, it is designed to have user interaction conducted through a local Flask frontend that uses API calls to connect to a remote server, allowing computation to be offloaded in order to minimize latency while maintaining an intuitive user experience. A link to our static website can be found [here](https://nljumaoas.github.io/multiagent-translation/).

## Setup

1. After cloning this repository, clone the following repository in the same directory as well:
- https://github.com/sqbly/Manga-Text-Segmentation


2. Download the text segmentation model and move it into the `Manga-Text-Segmentation` directory.
```bash
wget -O model.pkl https://github.com/juvian/Manga-Text-Segmentation/releases/download/v1.0/fold.0.-.final.refined.model.2.pkl
```

3. Navigate to the `dsc180b/backend/processing_stage` directory and install the required packages in a conda environment.
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

5. Download the LLM to be run as agents locally through Ollama if running the project without API key. This include downloading Ollama first, then the individual agent models:
```bash
wget https://ollama.com/install.sh
bash install.sh
ollama serve
ollama pull llava:7b
ollama pull llama3.1:8b
```
If you see warnings on unable to detect gpu after installed ollama, run:
```bash
apt-get update
apt-get install pciutils
```

6. Before running the pipeline, interact with Ollama from your local machine by running it as a server. Then run the pipeline from a new terminal:
```bash
ollama serve
ollama pull llava:7b
ollama pull llama3.1:8b
```

# Launching Demo
Make sure the environment is setup first, and store your openAI api key to the environment:
```bash
export API_KEY=<your key>
```
Then navigate to backend folder, launch the demo:
```bash
python app.py
```
