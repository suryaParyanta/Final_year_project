import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms


from Code.dataset_helpers import get_dataset, get_data_loader
from Code.network.vgg_backbone import initialize_vgg, initialize_vgg_attn, initialize_vgg_attn_prototype
from Code.network.resnet_backbone import initialize_LResNet50_attn
from Code.network.classifier import MarginCosineProduct
from eval_net import get_accuracy, lfw_eval


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        help = 'Dataset used',
        default = 'casia',
        type = str
    )

    parser.add_argument(
        '--model_type',
        help = 'Model used for evaluation',
        choices = ['vgg16', 'vgg16_prototype', 'vgg16_attn', 'resnet', 'resnet_attn'],
        default = 'resnet_attn',
        type = str
    )

    parser.add_argument(
        '--prototype_weight',
        help = 'Path to prototype weight',
        default = 'prototype_weight/prototype_vgg16.pkl',
        type = str
    )

    parser.add_argument(
        '--feature_dict',
        help = 'Path to feature dictionary weight file',
        default = 'feature_dictionary/CASIA/100000/dictionary_second_64.pickle',
        type = str
    )

    parser.add_argument(
        '--pretrained_weight',
        help = 'Path to pretrained VGG weight',
        default = 'results/resnet50_casia/epoch_25.pth',
        type = str
    )

    parser.add_argument(
        '--lr',
        help = 'Initial learning rate during training',
        default = 1e-4,
        type = float
    )

    parser.add_argument(
        '--momentum',
        help = 'Momentum of optimizer',
        default = 0.9,
        type = float
    )

    parser.add_argument(
        '--weight_decay',
        help = 'L2 regularization during training',
        default = 5e-4,
        type = float
    )

    parser.add_argument(
        '--step_size',
        help = 'Update learning rate for every specified epoch',
        default = 5,
        type = int
    )

    parser.add_argument(
        '--gamma',
        help = 'Learning rate update factor',
        default = 0.5,
        type = float
    )

    parser.add_argument(
        '--lambda1',
        help = 'Weight of prototype loss (if any)',
        default = 1.0,
        type = float
    )

    parser.add_argument(
        '--lambda2',
        help = 'Weight of feature dictionary loss',
        default = 1.0,
        type = float
    )

    parser.add_argument(
        '--batch_size',
        help = 'Batch size per iteration',
        default = 16,
        type = int
    )

    parser.add_argument(
        '--num_epoch',
        help = 'Number of epochs for training',
        default = 30,
        type = int
    )

    parser.add_argument(
        '--workers',
        help = 'How many workers to load data',
        default = 4,
        type = int
    )

    parser.add_argument(
        '--save_dir',
        help = 'Save directory',
        default = 'results/resnet_64_second_attn',
        type = str
    )

    parser.add_argument(
        '--device',
        help = 'Device to use for training',
        default = 'cuda',
        type = str
    )

    return parser.parse_args()


def train(model, args, train_loader, val_loader = None, device = 'cpu', save_dir = 'results/vgg16_prototype'):
    criterion = nn.CrossEntropyLoss()

    num_epoch = args.num_epoch
    reg_lambda = {
        'lambda1': args.lambda1,
        'lambda2': args.lambda2
    }

    optimizer = optim.SGD(
        model.parameters(),
        lr = args.lr,
        weight_decay = args.weight_decay,
        momentum = args.momentum
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)

    print('\nTraining the Model ...')
    model = model.to(device = device, dtype = torch.float32)

    for epoch in range(1, num_epoch+1):
        epoch_train_loss = 0
        model.train()

        for idx, (data, label) in enumerate(train_loader, 1):
            optimizer.zero_grad()

            data, label = data.to(device = device), label.to(device = device)
            score, additional_loss, additional_output = model(data)
            loss = calculate_loss(score, label, criterion, other_loss=additional_loss, reg = reg_lambda)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach().item()
        
        epoch_train_loss /= len(train_loader)

        # evaluation
        train_accuracy = get_accuracy(model, train_loader, device=device)
        
        if val_loader is not None:
            val_accuracy = get_accuracy(model, val_loader, device=device)
            print(f"Epoch {epoch}:  train_loss = {epoch_train_loss}, train_accuracy = {100*train_accuracy/args.num_train_images} %, test_accuracy = {100 * val_accuracy/args.num_test_images} %")
        else:
            print(f"Epoch {epoch}:  train_loss = {epoch_train_loss}, train_accuracy = {100*train_accuracy/args.num_train_images} %")
        
        # save model
        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pt'))

        lr_scheduler.step()


def train_face_recognition(model, classifier, model_eval, args, train_loader, device = 'cpu', save_dir = 'results/vgg16_prototype', log_interval = 1000):
    criterion = nn.CrossEntropyLoss()

    num_epoch = args.num_epoch
    reg_lambda = {
        'lambda2': args.lambda2
    }

    optimizer = optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr = args.lr,
        weight_decay = args.weight_decay,
        momentum = args.momentum
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)

    print('\nTraining the Model ...')
    model = model.to(device = device, dtype = torch.float32)
    classifier = classifier.to(device = device, dtype = torch.float32)

    loss_display = 0.0
    for epoch in range(1, num_epoch+1):
        model.train()
        classifier.train()

        for idx, (data, label) in enumerate(train_loader, 1):
            curr_iter = (epoch - 1) * len(train_loader) + idx
            optimizer.zero_grad()

            data, label = data.to(device = device), label.to(device = device)
            feature, additional_loss, additional_output = model(data)
            score = classifier(feature['features'], label)
            score_dict = {'score': score}
            loss = calculate_loss(score_dict, label, criterion, other_loss = additional_loss, reg = reg_lambda)

            loss.backward()
            optimizer.step()

            loss_display += loss.detach().item()

            if curr_iter % log_interval == 0:
                print(f"Iteration {curr_iter}: train_loss = {loss_display / log_interval}")
                loss_display = 0.0 

        # save the model
        try:
            model_state_dict = model.module.state_dict()
        except:
            model_state_dict = model.state_dict()

        torch.save(model_state_dict, os.path.join(save_dir, f'epoch_{epoch}.pt'))
        torch.save(classifier.state_dict(), os.path.join(save_dir, f'mcp_epoch_{epoch}.pt'))

        # LFW evaluation
        model_eval.load_state_dict(torch.load(os.path.join(save_dir, f'epoch_{epoch}.pt')))
    
        accuracy, th, acc_std = lfw_eval(model_eval, args.test_root, args.test_images, args.test_filelist, device = device)
        print(f"Epoch {epoch}: lfw_accuracy = {accuracy} %,  std = {acc_std},  threshold = {th}")

        lr_scheduler.step()


def calculate_loss(score, label, criterion, other_loss = {}, reg = {}):
    logits = score['score']
    loss = criterion(logits, label)

    if len(other_loss) > 0:
        for k in other_loss.keys():
            if k == 'prototype_loss':
                loss += other_loss[k] * reg['lambda1']
            elif k == 'vc_loss':
                print(other_loss[k])
                loss += other_loss[k] * reg['lambda2']
            else:
                loss += other_loss[k]
    
    return loss


if __name__ == '__main__':
    args = parse_argument()

    if args.dataset == 'mnist':
        root = '../dataset/MNIST_224X224_3'

        # Train dataset
        train_annot = 'pairs_train.txt'
        train_data = 'train'

        # Test dataset
        test_annot = 'pairs_test.txt'
        test_data = 'test'

        num_classes = 10
        save_dir = args.save_dir + '_mnist'

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # get dataloader for training and evaluation
        MNIST_train, MNIST_test = get_dataset(
            root,
            train_data,
            train_annot,
            transform,
            test_data_dir = test_data,
            test_annot_path = test_annot,
            transform_test = transform
        )

        args.num_train_images = len(MNIST_train)
        if MNIST_test is not None:
            args.num_test_images = len(MNIST_test)

        train_loader = get_data_loader(MNIST_train, batch_size = args.batch_size, num_workers = args.workers)
        test_loader = get_data_loader(MNIST_test, batch_size = 64)

        model = initialize_vgg_attn_prototype(args.prototype_weight, args.feature_dict, num_classes = num_classes)
        
        if os.path.exists(args.pretrained_weight):
            print('Using pretrain weight from', args.pretrained_weight)
            model.load_state_dict(torch.load(args.pretrained_weight), strict = False)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train(
            model, 
            args, 
            train_loader,
            val_loader = test_loader,
            device = args.device,
            save_dir = save_dir 
        )

    elif args.dataset == 'casia':
        num_clusters = 64
        root = 'dataset'

        # Train dataset
        train_data = 'CASIA_aligned'
        train_annot = 'CASIA_aligned_list.txt'

        # Test dataset
        args.test_root = 'dataset/LFW_pairs_aligned'
        args.test_images = 'Images'
        args.test_filelist = 'pairs.txt'

        num_classes = 10575
        save_dir = args.save_dir + '_casia'

        # transformation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])

        # training dataset
        CASIA_train, _ = get_dataset(
            root,
            train_data,
            train_annot,
            train_transform,
            mode = 'annotation'
        )
        CASIA_loader = get_data_loader(CASIA_train, batch_size = args.batch_size, num_workers = args.workers)

        if args.model_type == "vgg_16_attn":
            model = initialize_vgg_attn(num_clusters, args.feature_dict)
            model_eval = initialize_vgg_attn(num_clusters, args.feature_dict)
        elif args.model_type == "resnet_attn":
            model = initialize_LResNet50_attn(num_clusters, args.feature_dict)
            model_eval = initialize_LResNet50_attn(num_clusters, args.feature_dict)

        if os.path.exists(args.pretrained_weight):
            print("Load pre-trained weight from:", args.pretrained_weight)
            model.load_state_dict(torch.load(args.pretrained_weight), strict = False)
        
        for param in model.parameters():
            param.requires_grad = True
        
        model = nn.DataParallel(model)

        classifier = MarginCosineProduct(in_features = 512, out_features = num_classes)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            idx = 2
            while os.path.exists(save_dir + f'_{idx}'):
                idx += 1  
            os.makedirs(save_dir + f'_{idx}')
            save_dir = save_dir + f'_{idx}'

        train_face_recognition(
            model,
            classifier,
            model_eval,
            args,
            CASIA_loader,
            device = args.device,
            save_dir = save_dir,
            log_interval = 1000
        )