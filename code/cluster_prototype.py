import os
import sys
sys.path.append(os.getcwd())

import pickle
import argparse
import numpy as np
from sklearn.cluster import KMeans

import torch
from torchvision import transforms

from code.dataset_helpers import get_dataset, get_data_loader, DatasetSampler
from code.network.vgg_backbone import initialize_vgg, Identity


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root',
        help = 'Dataset root path',
        default = 'dataset/MNIST_224X224_3',
        type = str
    )

    parser.add_argument(
        '--num_classes',
        help = 'Number of classes in the dataset',
        default = 10,
        type = str
    )

    parser.add_argument(
        '--num_prototype',
        help = 'Number of prototype per class',
        default = 4,
        type = int
    )


    parser.add_argument(
        '--save_path',
        help = "Save path (.pkl file)",
        default = '',
        type = str
    )

    parser.add_argument(
        '--device',
        help = 'Device to use for training',
        default = 'cuda',
        type = str
    )

    return parser.parse_args()

    
def get_initial_prototype(model, train_dataset, num_classes:int, num_prototype:int = 4, save_path:str = 'prototype_weight/prototype_vgg16.pkl', device:str = 'cpu'):
    """
    Generate initial prototype weight. The process involves:
       1. Forward propagation to obtain feature map
       2. For each class, perform K-means clustering based on the feature map from step-1
    
    :param model:         Neural network model
    :param train_dataset: Training examples
    :param num_classes:   Number of classes in the dataset
    :param num_prototype: Number of prototype to be generated
    :param save_path:     Save directory
    :param device:        Device used for computation

    :returns: Prototype weight stored on the pickle file
    """
    batch_size = 128
    proto_weight = np.ndarray((num_classes, num_prototype, 512, 7, 7)).astype(np.float32) #(num_cls, num_proto, C, H, W)

    model.to(device)
    model.eval()
    for i in range(num_classes):
        mask = [1 if label == i else 0 for _, label in train_dataset]
        num_images = sum(mask)
        pool5_features = np.ndarray((num_images, 512*7*7)).astype(np.float32)

        # prepare the DataLoader
        mask = torch.tensor(mask)   
        sampler = DatasetSampler(mask, train_dataset)
        train_loader = get_data_loader(train_dataset, batch_size = batch_size, shuffle = False, sampler = sampler)

        for batch_idx, (data, label) in enumerate(train_loader):
            idx_start = batch_idx * batch_size
            idx_end = min([(batch_idx+1) * batch_size, num_images])
            
            # forward propagation
            with torch.no_grad():
                data = data.to(device)
                feature_map, *_ = model(data)
                feature_map = feature_map["score"].to('cpu').detach().numpy()
            
            pool5_features[idx_start:idx_end, :] = feature_map

        print(pool5_features.shape)

        kmeans = KMeans(n_clusters = num_prototype)
        kmeans.fit(pool5_features)
        centers = kmeans.cluster_centers_

        centers = np.reshape(centers, (num_prototype, 512, 7, 7))
        proto_weight[i, :, :, :, :] = centers

    if os.path.exists(save_path):
        idx = 2
        while os.path.exists(save_path.split(".")[0] + f"_{idx}" + ".pkl"):
            idx += 1
        save_path = save_path.split(".")[0] + f"_{idx}" + ".pkl"
    
    print(f"Saving prototype weight into: {save_path}")
    with open(save_path, 'wb') as prototype_file:
        pickle.dump(proto_weight, prototype_file)


if __name__ == '__main__':
    args = parse_argument()

    if args.save_path == "":
        args.save_path = os.path.join("prototype_weight", f"prototype_{args.num_classes}_{args.num_prototype}_MNIST.pkl")

    train_annot = 'pairs_train.txt'
    train_data = 'train'

    transform = transforms.Compose([transforms.ToTensor()])
    MNIST_train, _ = get_dataset(
        args.root,
        train_data,
        train_annot,
        transform
    )

    # load model
    model = initialize_vgg()
    model.load_state_dict(torch.load("pretrained_weight/VGG_MNIST/best.pt"))
    for param in model.parameters():
        param.requires_grad = False

    # do not need fully-connected layer
    model.fc = Identity()

    get_initial_prototype(
        model, 
        MNIST_train, 
        num_classes = args.num_classes, 
        num_prototype = args.num_prototype,
        save_path = args.save_path,
        device = args.device
    )