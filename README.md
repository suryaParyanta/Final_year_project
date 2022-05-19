# Final_year_project

This project is to explore the existing problem of image occlusion in general as well as face recognition via facial masks. 
The repository includes the implementaion of the following:
* General image occlusion problem using MNIST dataset.
* Occluded face recognition with self-generated masks.

# Environment Setup

Download or clone this repository
```bash
git clone https://github.com/suryaParyanta/Final_year_project.git
```

Create a conda virtual environment:
```bash
conda create -n ENV_NAME python=3.8.5
conda activate ENV_NAME
```

Install pytorch 1.7.1 with cuda toolkit 11.0 from this [link](https://pytorch.org/get-started/previous-versions/) (it is recommended to use pip instead of conda):
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install required packages:
```bash
cd Final_year_project
pip install -r requirements.txt
```

# Project Directories

* `Code/` folder contains the implementation of models, dataset, dataloader, and prototype generation (the output is stored on `prototype_weight/` folder).
* `config/` folder contains model configuration files used for training and evaluation.
* `feature_dictionary/` folder contains feature dictionary weight obtained from vMF clustering.
* `notebooks/` folder contains visualization of models (the output is stored on `visualization/` folder).

# Getting Started

To train the model with one or more gpu, run the following command (Note: if you want to run on CPU, set `--device cuda`):
```bash
python train_net.py --config_file PATH_TO_CONFIG_FILE --device_ids GPU_IDS
```

To run face verification evaluation, run the following command:
```bash
python eval_net.py --config_file PATH_TO_CONFIG_FILE --weight PATH_TO_MODEL_WEIGHT
```
