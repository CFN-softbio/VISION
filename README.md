# VISION: A Modular AI Assistant for Natural Human-Instrument Interaction at Scientific User Facilities

VISION (Virtual Scientific Companion) is an AI-driven assistant designed to streamline operations at synchrotron beamlines and scientific facilities through natural language interaction. Built on a modular architecture, VISION is an assembly of **Cognitive Blocks (Cogs)**—specialized AI components tailored for tasks like transcription, classification, code generation, data analysis, and scientific querying. These cogs operate in predefined **workflows**, enabling seamless communication between users and complex instrumentation.

Key workflows include natural language-controlled (audio, text, or both) **beamline operations**, where commands are classified, converted to executable code, and deployed for data acquisition or analysis; **custom function addition**, where custom functions defined by the user in natural language are dynamically integrated into the system; and a **domain-specific chatbot**, capable of answering scientific queries with precision. 

![Overview of the VISION architecture](images/vision_architecture.png "VISION Modular Architecture")

*Figure 1: Overview of the VISION architecture*

![NSLS-II GUI Deployment](images/vision_demo.png "NSLS-II GUI Deployment")

*Figure 2: VISION deployment at NSLS-II 11-BM CMS: GUI was launched at the beamline workstation, with backend processing performed on HAL. LLM-based cog results are displayed to prompt user confirmation, followed by execution on Bluesky or other software.*


## Paper
Please see our paper here - https://arxiv.org/abs/2412.18161

## Directory Structure
Highlight of the most important files:
```
📁 VISION/
├── 📁 backend/
│   ├── 📁 src/
│   │   └── 📁 hal_beam_com/
│   │       └── cog_manager.py      (main entry-point for backend)
│   └── 📁 tests/                   (contains testing framework to generate evaluate LLMs on our datasets)
└── 📁 frontend/
    └── 📁 UI/
        └── 📁 program/
            └── executable.py        (main entry-point for front-end)
```

## How to run
For each of the folders, create a separate virtual environment through Python 3.12.7 and run `pip install -r ./requirements.txt`.

Then execute:
* `python ./frontend/UI/program/executable.py` for the frontend and 
* `python ./backend/src/hal_beam_com/cog_manager.py` for the backend.

If you want to use Claude's Anthropic as the main model for the cogs (LLMs with a specialized prompt), please set the `ANTHROPIC_API_KEY` in your environment variables.

Alternatively, you can switch the model setup by changing the `ACTIVE_CONFIG` in `./backend/src/hal_beam_com/utils.py`. Running most models will require having Ollama installed and being able to run your selected models. New models can be added through `base_models_path` and then used when adding it to a configuration in `model_configurations`.

More detailed explanations of the install are in the README's of the respective directories.

For this public facing repo, local communication (sending information between the front and back-end using local files) is standard (`CustomS3.py`). Minio functionality can be restored by replacing `CustomS3.py` with the code in `MinioCustomS3.py`. 

## Citation
```bibtex
@misc{mathur2024visionmodularaiassistant,
      title={VISION: A Modular AI Assistant for Natural Human-Instrument Interaction at Scientific User Facilities}, 
      author={Shray Mathur and Noah van der Vleuten and Kevin Yager and Esther Tsai},
      year={2024},
      eprint={2412.18161},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.18161}, 
}
```
