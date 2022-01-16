import os
import yaml
from pathlib import Path
import sys
sys.path.append(os.getcwd())

import logging
logging.basicConfig(format = "%(asctime)s-%(levelname)s:   %(message)s")
logger = logging.getLogger(__name__)
print = logger.info

import torch
from torchvision import transforms

from Code.dataset_helpers import get_dataset, get_data_loader
from Code.network.vgg_backbone import initialize_vgg, initialize_vgg_attn, initialize_vgg_attn_prototype
from Code.network.resnet_backbone import initialize_LResNet50_IR, initialize_LResNet50_attn
from Code.network.classifier import MarginCosineProduct


def get_default_cfg():
    RESNETS = {
        "WEIGHTS": "",
        "LAYERS": [],
        "FILTER": []
    }

    VGG = {
        "WEIGHTS": "",
        "LAYERS": [],
        "FILTER": []
    }

    ATTN = {
        "NUM_CLUSTERS": 0,
        "WEIGHTS":  "",
        "PERCENT_U": 0.2,
        "PERCENT_L": 0.8
    }

    PROTOTYPE = {
        "NUM_PROTOTYPES": 0,
        "WEIGHTS": "",
        "METHOD": "kmeans"
    }

    DATASETS = {
        "TRAIN": [],
        "VAL": [],
        "TEST": []
    }

    SOLVER = {
        "NUM_EPOCHS": 30,
        "BATCH_SIZE": 64,
        "BASE_LR": 1e-3,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 5e-4,
        "STEP_SIZE": 5,
        "STEP_FACTOR": 0.5,
        "VC_LAMBDA": 1.0,
        "P_LAMBDA": 1.0
    }

    default_cfg = {
        "MODEL": {
            "NAME": "",
            "WEIGHTS": "",
            "OUT_FEATURES": -1,
            "NUM_CLASSES": 10,
            "NORM_ATTN": False,
            "USE_THRESH": False,
            "NORM_BEFORE_ATTN": False,
            "RECURRENT_STEP": 0,
            "RESNETS": RESNETS,
            "VGG": VGG,
            "ATTN": ATTN,
            "PROTOTYPE": PROTOTYPE
        },
        "DATASETS": DATASETS,
        "SOLVER": SOLVER,
        "SAVE": "results"
    }

    return default_cfg


def merge_cfg(cfg1, cfg2):
    for key in cfg1.keys():
        if cfg2.get(key, None) is None:
            continue

        if isinstance(cfg1[key], dict):
            merge_cfg(cfg1[key], cfg2[key])    
        else:
            cfg1[key] = cfg2[key]


def merge_cfg_from_file(cfg, cfg_file):
    with open(cfg_file, 'r') as f:
        _C = yaml.load(f, yaml.Loader)

    base_cfg = _C.get("BASE", "")
    dir = os.path.dirname(cfg_file)
    base_file = os.path.join(dir, base_cfg)
    
    # merge with base
    if os.path.exists(base_file) and not os.path.isdir(base_file):
        with open(base_file, 'r') as f:
            _C_BASE = yaml.load(f, yaml.Loader)
        merge_cfg(cfg, _C_BASE)

    # merge with child
    merge_cfg(cfg, _C)
    
    return cfg


def get_model_from_cfg(cfg):
    assert cfg["MODEL"]["NAME"] in ("vgg", "vgg_attn", "vgg_attn_proto", "resnet", "resnet_attn"), "Unsupported model name"

    freeze_epoch = -1
    if cfg["MODEL"]["NAME"] == "vgg":
        model = initialize_vgg(cfg["MODEL"]["NUM_CLASSES"])
        classifier = None

    elif cfg["MODEL"]["NAME"] == "vgg_attn":
        model = initialize_vgg_attn(
            cfg["MODEL"]["ATTN"]["NUM_CLUSTERS"],
            cfg["MODEL"]["ATTN"]["WEIGHTS"],
            cfg["MODEL"]["RECURRENT_STEP"],
            cfg["MODEL"]["ATTN"]["PERCENT_U"],
            cfg["MODEL"]["ATTN"]["PERCENT_L"]
        )
        # freeze feature dictionary for first 10 epochs
        freeze_epoch = 10
        model.feature_dict.requires_grad = False

        classifier = MarginCosineProduct(
            cfg["MODEL"]["OUT_FEATURES"],
            cfg["MODEL"]["NUM_CLASSES"]
        )
    
    elif cfg["MODEL"]["NAME"] == "vgg_attn_proto":
        model = initialize_vgg_attn_prototype(
            cfg["MODEL"]["NUM_CLASSES"],
            cfg["MODEL"]["PROTOTYPE"]["NUM_PROTOTYPES"],
            cfg["MODEL"]["ATTN"]["NUM_CLUSTERS"],
            cfg["MODEL"]["PROTOTYPE"]["WEIGHTS"],
            cfg["MODEL"]["ATTN"]["WEIGHTS"],
            cfg["MODEL"]["PROTOTYPE"]["METHOD"],
            cfg["MODEL"]["RECURRENT_STEP"],
            cfg["MODEL"]["ATTN"]["PERCENT_U"],
            cfg["MODEL"]["ATTN"]["PERCENT_L"]
        )
        # freeze feature dictionary for several epochs
        freeze_epoch = 10
        model.feature_dict.requires_grad = False
        classifier = None
    
    elif cfg["MODEL"]["NAME"] == "resnet":
        model = initialize_LResNet50_IR()
        classifier = MarginCosineProduct(
            cfg["MODEL"]["OUT_FEATURES"],
            cfg["MODEL"]["NUM_CLASSES"]
        )
    
    elif cfg["MODEL"]["NAME"] == "resnet_attn":
        model = initialize_LResNet50_attn(
            cfg["MODEL"]["ATTN"]["NUM_CLUSTERS"],
            cfg["MODEL"]["ATTN"]["WEIGHTS"],
            cfg["MODEL"]["RECURRENT_STEP"],
            cfg["MODEL"]["ATTN"]["PERCENT_U"],
            cfg["MODEL"]["ATTN"]["PERCENT_L"]
        )
        # freeze feature dictionary for several epochs
        freeze_epoch = 10
        model.feature_dict.requires_grad = False

        classifier = MarginCosineProduct(
            cfg["MODEL"]["OUT_FEATURES"],
            cfg["MODEL"]["NUM_CLASSES"]
        )
    
    # load pretrained backbone if any
    for k in ["VGG", "RESNETS"]:
        if os.path.exists(cfg["MODEL"][k]["WEIGHTS"]):
            print(f"Load pretrained backbone from: {cfg['MODEL'][k]['WEIGHTS']}")
            model.load_state_dict(torch.load(cfg["MODEL"][k]["WEIGHTS"]), strict = False)

            freeze_epoch = 10
            # freeze the model (finetune the classifier/prototype first)
            for name, param in model.named_parameters():
                if "resnet" in cfg["MODEL"]["NAME"]:
                    if "layer4" not in name and "fc" not in name:
                        param.requires_grad = False
                else:
                    if "layer5" not in name and "prototype_weight" not in name and "gamma" not in name:
                        param.requires_grad = False

    # load pretrained weight if any
    if os.path.exists(cfg["MODEL"]["WEIGHTS"]):
        print(f"Load pretrained weight from: {cfg['MODEL']['WEIGHTS']}")
        model.load_state_dict(torch.load(cfg["MODEL"]["WEIGHTS"]))
    
    return model, classifier, freeze_epoch


def get_dataloader_from_cfg(cfg):
    supp_train_data = ["", "mnist_train", "casia"]
    supp_test_data = ["", "mnist_train", "mnist_test", "mnist_test_occ_black", "mnist_test_occ_gauss", "mnist_test_occ_flower", "lfw", "lfw_masked"]
    assert cfg["DATASETS"]["TRAIN"] in supp_train_data, "Unsupported train dataset name."
    assert cfg["DATASETS"]["VAL"] in supp_test_data, "Unsupported val/test dataset name."
    assert cfg["DATASETS"]["TEST"] in supp_test_data, "Unsupported val/test dataset name."
    
    loaders = []
    num_dataset_images = []
    keys = ["TRAIN", "VAL", "TEST"]

    for k in keys:
        transform = None
        if "mnist" in cfg["DATASETS"][k]:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        elif cfg["DATASETS"][k] == "casia":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])

        # get pytorch dataset
        if cfg["DATASETS"][k] == "mnist_train":
            root = "dataset/MNIST_224X224_3"
            train_annot = "pairs_train.txt"
            train_data = "train"

            dataset, _ = get_dataset(
                root,
                train_data,
                train_annot,
                transform
            )

        elif cfg["DATASETS"][k] == "casia":
            root = "dataset"
            train_annot = "CASIA_aligned_list.txt"
            train_data = "CASIA_aligned"

            dataset, _ = get_dataset(
                root,
                train_data,
                train_annot,
                transform,
                mode = 'annotation'
            )

        elif cfg["DATASETS"][k] == "mnist_test":
            root = "dataset/MNIST_224X224_3"
            train_annot = "pairs_test.txt"
            train_data = "test"

            dataset, _ = get_dataset(
                root,
                train_data,
                train_annot,
                transform
            )

        elif cfg["DATASETS"][k] == "mnist_test_occ_black":
            root = "dataset/MNIST_224X224_3"
            train_annot = "pairs_test.txt"
            train_data = "test_occ_black"

            dataset, _ = get_dataset(
                root,
                train_data,
                train_annot,
                transform
            )

        elif cfg["DATASETS"][k] == "mnist_test_occ_gauss":
            root = "dataset/MNIST_224X224_3"
            train_annot = "pairs_test.txt"
            train_data = "test_occ_gauss"

            dataset, _ = get_dataset(
                root,
                train_data,
                train_annot,
                transform
            )

        elif cfg["DATASETS"][k] == "mnist_test_occ_flower":
            root = "dataset/MNIST_224X224_3"
            train_annot = "pairs_test.txt"
            train_data = "test_occ_flower"

            dataset, _ = get_dataset(
                root,
                train_data,
                train_annot,
                transform
            )
        
        else:
            dataset = None
        
        # get pytorch dataloader
        num_images = 1
        loader = None
        if dataset is not None:
            print("")
            print(f"{k} DATASET")
            print(f"Root folder: {root}")
            print(f"Annotation file: {train_annot}")
            print(f"Images directory: {train_data}")

            num_images = len(dataset)
            loader = get_data_loader(
                dataset,
                cfg["SOLVER"]["BATCH_SIZE"]
            )
        num_dataset_images.append(num_images)
        loaders.append(loader)

    train_loader, val_loader, test_loader = loaders
    return num_dataset_images, train_loader, val_loader, test_loader


def get_parameters_from_cfg(cfg):
    params = {
        "lr": cfg["SOLVER"]["BASE_LR"],
        "num_epoch": cfg["SOLVER"]["NUM_EPOCHS"],
        "momentum": cfg["SOLVER"]["MOMENTUM"],
        "weight_decay": cfg["SOLVER"]["WEIGHT_DECAY"],
        "lambda1": cfg["SOLVER"]["P_LAMBDA"],
        "lambda2": cfg["SOLVER"]["VC_LAMBDA"],
        "step_size": cfg["SOLVER"]["STEP_SIZE"],
        "step_factor": cfg["SOLVER"]["STEP_FACTOR"],
        "val_root": "",
        "val_images": "",
        "val_filelist": "",
        "test_root": "",
        "test_images": "",
        "test_filelist": ""
    }

    for k in ["VAL", "TEST"]:
        if cfg["DATASETS"][k] == "lfw_masked":
            print("")
            print(f"{k} DATASET")
            print("Root folder: dataset/LFW_pairs_aligned")
            print("Annotation file: pairs_masked.txt")
            print("Images directory: Combined")

            params[f"{k.lower()}_root"] = "dataset/LFW_pairs_aligned"
            params[f"{k.lower()}_images"] = "Combined"
            params[f"{k.lower()}_filelist"] = "pairs_masked.txt"
        elif cfg["DATASETS"][k] == "lfw":
            print("")
            print(f"{k} DATASET")
            print("Root folder: dataset/LFW_pairs_aligned")
            print("Annotation file: pairs.txt")
            print("Images directory: Images")

            params[f"{k.lower()}_root"] = "dataset/LFW_pairs_aligned"
            params[f"{k.lower()}_images"] = "Images"
            params[f"{k.lower()}_filelist"] = "pairs.txt"

    return params


def get_savedir(cfg, cfg_file):
    save_dir = Path(cfg_file).stem
    save_dir = os.path.join(cfg["SAVE"], save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        idx = 2
        while os.path.exists(save_dir + f'_{idx}'):
            idx += 1  
        os.makedirs(save_dir + f'_{idx}')
        save_dir = save_dir + f'_{idx}'

    # save config file into the directory
    save_filename = "config.yaml"
    with open(os.path.join(save_dir, save_filename), 'w') as f:
        yaml.dump(cfg, f)

    return save_dir


if __name__ == "__main__":
    # Run from main project directories
    cfg_file = "config/LResNet50IR_Attn/Resnet_Attn_64_Second_LFW.yaml"
    cfg = get_default_cfg()
    merge_cfg_from_file(cfg, cfg_file)