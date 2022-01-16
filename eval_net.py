import os
import logging
import argparse

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
from torchvision import transforms

from config.config_parser import *

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file',
        help = 'Path to the config file',
        type = str
    )

    parser.add_argument(
        '--device',
        help = 'Device to use for training',
        default = 'cuda',
        type = str
    )

    return parser.parse_args()


def get_accuracy(model, test_loader, device:str = 'cpu'):
    """
    Compute raw top-1 accuracy of the model.

    :param model:       Neural network model
    :param test_loader: Test data loader
    :param device:      Device used for computation

    :returns: Top-1 accuracy of the test dataset
    :rtype:   int
    """
    model.to(device = device)
    model.eval()

    acc = 0
    for batch_data, label in test_loader:
        batch_data = batch_data.to(device = device)

        with torch.no_grad():
            score, *_ = model(batch_data)
            score_prob = F.softmax(score['score'], dim=1)
            predictions = torch.max(score_prob, dim=1)[1].to('cpu')

            curr_acc = (predictions == label).float().sum()
            acc += curr_acc.detach().item()
        
    return acc


def k_fold(n:int = 6000, n_folds:int = 10):
    """
    Get the training and test data indexes for K-fold cross validation.

    :param n:       Number of examples
    :param n_folds: Number of cross validation folds

    :returns: 2D List that contains training indexes and test indexes
    :rtype:   list 
    """
    folds = []
    base = list(range(n))

    for i in range(n_folds):
        test_idx = base[i * n//n_folds:(i+1) * n//n_folds]
        train_idx = list(set(base) - set(test_idx))
        folds.append([train_idx, test_idx])

    return folds


def extract_feature(img, model, device:str = 'cpu'):
    """
    Extract features from the image (with horizontal flip).

    :param img: PIL Image
    :param model: Neural network model
    :param device: Device used for computation

    :returns: Features with 1024 dimensions
    :rtype: torch.FloatTensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    img, img_flip = transform(img), transform(F_t.hflip(img))
    img, img_flip = img.unsqueeze(0).to(device) , img_flip.unsqueeze(0).to(device)

    feature1 = model(img)[0]['features']
    feature2 = model(img_flip)[0]['features']
    features = torch.cat([feature1, feature2], 1)[0].to('cpu')

    return features


def get_best_threshold(thresholds, distance_pred):
    """
    Determine the best threshold based on the top-1 accuracy.

    :param thresholds:    List of thresholds
    :param distance_pred: List of predictions, each prediction in the form of:
                             (filename1, filename2, distance, ground_truth)
    
    :returns: The best threshold that gives maximum accuracy
    :rtype:   float
    """
    best_th = best_acc = 0

    for t in thresholds:
        curr_acc = get_lfw_accuracy(t, distance_pred)

        if curr_acc > best_acc:
            best_th = t
            best_acc = curr_acc
    
    return best_th


def get_lfw_accuracy(threshold:float, distances:list):
    """
    Calculate LFW accuracy based on distance predictions.

    :param threshold: Threshold to determine identity matching
    :param distances: List of predictions + ground truth

    :returns: Accuracy on LFW dataset
    :rtype:   float
    """
    y_true = []
    y_pred = []

    for d in distances:
        same_flag = 1 if float(d[2]) > threshold else 0
        
        y_pred.append(same_flag)
        y_true.append(int(d[3]))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = 100.0 * np.count_nonzero(y_true == y_pred) / len(y_true)

    return accuracy


def lfw_eval(model, root, image_dir, pairs_filelist, device = 'cpu', file_ext = ''):
    """
    Perform 10-fold cross validation of LFW evaluation. The process consists of:
       1. Receive input of image pairs and pass each of them into the model
       2. Compute the distance of both image features
       3. If distance > threshold, predict 1 (same identity) else predict 0
       4. Determine the best threshold for each fold
       5. Calculate the accuracy mean, accuracy std, and threshold mean accross all folds.
    
    :param model:          Neural network model for evaluation
    :param root:           Root directory of LFW dataset
    :param image_dir:      Image directory of LFW dataset
    :param pairs_filelist: List of pairs file for evaluation
    :param device:         Device used for computation
    :param file_ext:       Extension of image file (Ex. file_ext = '_ext' -> img1_ext.jpg)
    """
    model.to(device = device)
    model.eval()

    predictions = []

    with open(os.path.join(root, pairs_filelist), 'r') as f:
        pairs_lines = f.readlines()[:]

    with torch.no_grad():
        for i in range(len(pairs_lines)):
            pairs = pairs_lines[i].replace('\n', '').split()

            if len(pairs) == 3:
                # if annotation in form of: img1  img2  label
                if os.path.splitext(pairs[0])[1] in ['.jpg', '.png'] and os.path.splitext(pairs[1])[1] in ['.jpg', '.png']:
                    same_flag = int(pairs[2])
                    filename1 = pairs[0]
                    filename2 = pairs[1]
                else:
                    # same identity
                    same_flag = 1
                    fileid1 = '0' * (4 - len(str(pairs[1])) ) + str(pairs[1])
                    fileid2 = '0' * (4 - len(str(pairs[2])) ) + str(pairs[2])
                    filename1 = f'{pairs[0]}_' + fileid1 + file_ext + '.jpg'
                    filename2 = f'{pairs[0]}_' + fileid2 + file_ext + '.jpg'
            elif len(pairs) == 4:
                # different identity
                same_flag = 0
                fileid1 = '0' * (4 - len(str(pairs[1])) ) + str(pairs[1])
                fileid2 = '0' * (4 - len(str(pairs[3])) ) + str(pairs[3])
                filename1 = f'{pairs[0]}_' + fileid1 + file_ext + '.jpg'
                filename2 = f'{pairs[2]}_' + fileid2 + file_ext + '.jpg'

            with open(os.path.join(root, image_dir, filename1), 'rb') as f:
                img1 = Image.open(f).convert('RGB')
            
            with open(os.path.join(root, image_dir, filename2), 'rb') as f:
                img2 = Image.open(f).convert('RGB')

            feature1 = extract_feature(img1, model, device = device)
            feature2 = extract_feature(img2, model, device = device)

            distance = feature1.dot(feature2) / (feature1.norm() * feature2.norm() + 1e-5)
            predictions.append([filename1, filename2, distance.item(), same_flag])
    
    accuracy = []
    best_thresholds = []
    folds = k_fold(n = len(predictions), n_folds = 10)
    thresholds = np.arange(-1, 1, 0.005)
    distance_pred = np.array(predictions)

    for train_idx, test_idx in folds:
        best_th = get_best_threshold(thresholds, distance_pred[train_idx])
        acc = get_lfw_accuracy(best_th, distance_pred[test_idx])

        accuracy.append(acc)
        best_thresholds.append(best_th)

    return np.mean(accuracy), np.mean(best_thresholds), np.std(accuracy)


def setup_eval(args, cfg):
    assert len(cfg['DATASETS']['VAL']) or len(cfg['DATASETS']['TEST']), "Dataset names are not specified"
    cfg['DATASETS']['TRAIN'] = ""

    model, *_ = get_model_from_cfg(cfg)

    num_images, _, val_loader, test_loader = get_dataloader_from_cfg(cfg)
    
    params = get_parameters_from_cfg(cfg)
    params["num_val_images"] = num_images[1]
    params["num_test_images"] = num_images[2]

    if val_loader is not None:
        print("")
        print(f"Evaluation on {cfg['DATASETS']['VAL']} using {cfg['MODEL']['NAME']}")
        val_acc = get_accuracy(model, val_loader, device = args.device)
        print(f"Accuracy on {cfg['DATASETS']['VAL']}: {round(val_acc / params['num_val_images'], 3)}")
    elif params["val_root"] != "":
        print("")
        print(f"Evaluation on {cfg['DATASETS']['VAL']} using {cfg['MODEL']['NAME']}")
        
        acc, threshold, std = lfw_eval(
            model,
            params["val_root"],
            params["val_images"],
            params["val_filelist"],
            device = args.device
        )
        print(f"Accuracy on {cfg['DATASETS']['VAL']}: {round(acc, 3)} %,  std = {round(std, 3)},  threshold = {round(threshold, 3)}")
    
    if test_loader is not None:
        print("")
        print(f"Evaluation on {cfg['DATASETS']['TEST']} using {cfg['MODEL']['NAME']}")
        test_acc = get_accuracy(model, test_loader, device = args.device)
        print(f"Accuracy on {cfg['DATASETS']['TEST']}: {round(test_acc / params['num_test_images'], 3)}")
    elif params["test_root"] != "":
        print("")
        print(f"Evaluation on {cfg['DATASETS']['TEST']} using {cfg['MODEL']['NAME']}")
        
        acc, threshold, std = lfw_eval(
            model,
            params["test_root"],
            params["test_images"],
            params["test_filelist"],
            device = args.device
        )
        print(f"Accuracy on {cfg['DATASETS']['TEST']}: {round(acc, 3)} %,  std = {round(std, 3)},  threshold = {round(threshold, 3)}")


if __name__ == '__main__':
    args = parse_argument()

    # setup logging
    logging.basicConfig(format = "%(asctime)s-%(levelname)s:   %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    cfg = get_default_cfg()
    cfg = merge_cfg_from_file(cfg, args.config_file)

    setup_eval(args, cfg)