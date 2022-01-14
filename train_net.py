import os
import argparse
import logging
logging.basicConfig(format = "%(asctime)-15s %(levelname)-8s %(message)s")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from config.config_parser import *
from eval_net import get_accuracy, lfw_eval


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file',
        help = 'Path to the config file',
        type = str
    )

    parser.add_argument(
        '--device_ids',
        default = [0],
        nargs = '+',
        help = 'Device ids for training',
        type = int

    )
    parser.add_argument(
        '--device',
        default = 'cuda',
        help = 'Device to use for training',
        type = str
    )

    return parser.parse_args()


def train(model, model_eval, params, train_loader, val_loader, test_loader, device = 'cpu', save_dir = 'results/vgg16_prototype'):
    criterion = nn.CrossEntropyLoss()

    num_epoch = params["num_epoch"]
    reg_lambda = {
        'lambda1': params["lambda1"],
        'lambda2': params["lambda2"]
    }

    optimizer = optim.SGD(
        model.parameters(),
        lr = params["lr"],
        weight_decay = params["weight_decay"],
        momentum = params["momentum"]
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size = params["step_size"], 
        gamma = params["step_factor"]
    )

    print('Training the Model ...')
    model = model.to(device = device, dtype = torch.float32)

    for epoch in range(1, num_epoch+1):
        epoch_train_loss = 0
        model.train()

        for idx, (data, label) in enumerate(train_loader, 1):
            optimizer.zero_grad()

            data, label = data.to(device = device), label.to(device = device)
            score, additional_loss, additional_output = model(data)
            loss = calculate_loss(score, label, criterion, other_loss = additional_loss, reg = reg_lambda)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach().item()
        
        epoch_train_loss /= len(train_loader)

        # save model
        try:
            model_state_dict = model.module.state_dict()
        except:
            model_state_dict = model.state_dict()

        torch.save(model_state_dict, os.path.join(save_dir, f'epoch_{epoch}.pt'))

        # evaluation
        model_eval.load_state_dict(torch.load(os.path.join(save_dir, f'epoch_{epoch}.pt')))

        train_accuracy = get_accuracy(model_eval, train_loader, device=device)
        log_msg = f"Epoch {epoch}:  train_loss = {epoch_train_loss}, train_acc = {round(100 * train_accuracy / params['num_train_images'], 3)} %"
        
        if val_loader is not None:
            val_accuracy = get_accuracy(model_eval, val_loader, device=device)
            log_msg += f", val_acc = {round(100 * val_accuracy / params['num_val_images'], 3)} %"

        if epoch % 5 == 0 and test_loader is not None:
            test_accuracy = get_accuracy(model_eval, test_loader, device=device)
            log_msg += f", test_acc = {round(100 * test_accuracy / params['num_test_images'], 3)} %"

        print(log_msg)

        lr_scheduler.step()


def train_face_recognition(model, classifier, model_eval, params, train_loader, device = 'cpu', save_dir = 'results/vgg16_prototype', log_interval = 1000):
    criterion = nn.CrossEntropyLoss()

    num_epoch = params["num_epoch"]
    reg_lambda = {
        'lambda2': params["lambda2"]
    }

    optimizer = optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr = params["lr"],
        weight_decay = params["weight_decay"],
        momentum = params["momentum"]
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size = params["step_size"], 
        gamma = params["step_factor"]
    )

    print('Training the Model ...')
    model = model.to(device = device, dtype = torch.float32)
    classifier = classifier.to(device = device, dtype = torch.float32)

    loss_display = 0.0
    for epoch in range(1, num_epoch+1):
        model.train()
        classifier.train()

        # unfreeze model parameters for epoch 6 onwards
        if epoch == 6:
            for p in model.parameters():
                p.requires_grad = True

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

        if os.path.exists(params["val_root"]):
            accuracy, th, acc_std = lfw_eval(model_eval, params["val_root"], params["val_images"], params["val_filelist"], device = device)
            print(f"Epoch {epoch}: val_acc = {accuracy} %,  std = {acc_std},  threshold = {th}")

        if epoch % 5 == 0 and os.path.exists(params["test_root"]):
            accuracy, th, acc_std = lfw_eval(model_eval, params["test_root"], params["test_images"], params["test_filelist"], device = device)
            print(f"Epoch {epoch}: test_acc = {accuracy} %,  std = {acc_std},  threshold = {th}")

        lr_scheduler.step()


def calculate_loss(score, label, criterion, other_loss = {}, reg = {}):
    logits = score['score']
    loss = criterion(logits, label)

    if len(other_loss) > 0:
        for k in other_loss.keys():
            if k == 'prototype_loss':
                loss += other_loss[k].mean() * reg['lambda1']
            elif k == 'vc_loss':
                loss += other_loss[k].mean() * reg['lambda2']
            else:
                loss += other_loss[k].mean()
    
    return loss


def setup(args, cfg, save_dir):

    model, classifier = get_model_from_cfg(cfg)
    model_eval, _ = get_model_from_cfg(cfg)

    for p in model_eval.parameters():
        p.requires_grad = False

    num_images, train_loader, val_loader, test_loader = get_dataloader_from_cfg(cfg)

    params = get_parameters_from_cfg(cfg)
    params["num_train_images"] = num_images[0]
    params["num_val_images"] = num_images[1]
    params["num_test_images"] = num_images[2]

    # if the gpu exists
    if len(args.device_ids) > 0:
        model = nn.DataParallel(model, device_ids = args.device_ids)

    if classifier is None:
        # classification task
        train(
            model, model_eval,
            params,
            train_loader, val_loader, test_loader,
            args.device,
            save_dir
        )
    else:
        train_face_recognition(
            model,
            classifier,
            model_eval,
            params,
            train_loader,
            args.device,
            save_dir,
            log_interval = 1
        )


if __name__ == '__main__':
    args = parse_argument()
    cfg = get_default_cfg()

    if args.device == "cpu":
        args.device_ids = []

    cfg = merge_cfg_from_file(cfg, args.config_file)
    save_dir = get_savedir(cfg, args.config_file)

    # setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(save_dir, "training.log"), 'a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s--%(levelname)s:    %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    print = logger.info

    setup(args, cfg, save_dir)

    # if args.dataset == 'mnist':
    #     root = '../dataset/MNIST_224X224_3'

    #     # Train dataset
    #     train_annot = 'pairs_train.txt'
    #     train_data = 'train'

    #     # Test dataset
    #     test_annot = 'pairs_test.txt'
    #     test_data = 'test'

    #     num_classes = 10
    #     save_dir = args.save_dir + '_mnist'

    #     transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])

    #     # get dataloader for training and evaluation
    #     MNIST_train, MNIST_test = get_dataset(
    #         root,
    #         train_data,
    #         train_annot,
    #         transform,
    #         test_data_dir = test_data,
    #         test_annot_path = test_annot,
    #         transform_test = transform
    #     )

    #     args.num_train_images = len(MNIST_train)
    #     if MNIST_test is not None:
    #         args.num_test_images = len(MNIST_test)

    #     train_loader = get_data_loader(MNIST_train, batch_size = args.batch_size, num_workers = args.workers)
    #     test_loader = get_data_loader(MNIST_test, batch_size = 64)

    #     model = initialize_vgg_attn_prototype(args.prototype_weight, args.feature_dict, num_classes = num_classes)
        
    #     if os.path.exists(args.pretrained_weight):
    #         print('Using pretrain weight from', args.pretrained_weight)
    #         model.load_state_dict(torch.load(args.pretrained_weight), strict = False)
        
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     train(
    #         model, 
    #         args, 
    #         train_loader,
    #         val_loader = test_loader,
    #         device = args.device,
    #         save_dir = save_dir 
    #     )

    # elif args.dataset == 'casia':
    #     num_clusters = 64
    #     root = 'dataset'

    #     # Train dataset
    #     train_data = 'CASIA_aligned'
    #     train_annot = 'CASIA_aligned_list.txt'

    #     # Test dataset
    #     args.test_root = 'dataset/LFW_pairs_aligned'
    #     args.test_images = 'Images'
    #     args.test_filelist = 'pairs.txt'

    #     num_classes = 10575
    #     save_dir = args.save_dir + '_casia'

    #     # transformation
    #     train_transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    #     ])

    #     # training dataset
    #     CASIA_train, _ = get_dataset(
    #         root,
    #         train_data,
    #         train_annot,
    #         train_transform,
    #         mode = 'annotation'
    #     )
    #     CASIA_loader = get_data_loader(CASIA_train, batch_size = args.batch_size, num_workers = args.workers)

    #     if args.model_type == "vgg_16_attn":
    #         model = initialize_vgg_attn(num_clusters, args.feature_dict)
    #         model_eval = initialize_vgg_attn(num_clusters, args.feature_dict)
    #     elif args.model_type == "resnet_attn":
    #         model = initialize_LResNet50_attn(num_clusters, args.feature_dict)
    #         model_eval = initialize_LResNet50_attn(num_clusters, args.feature_dict)

    #     if os.path.exists(args.pretrained_weight):
    #         print("Load pre-trained weight from:", args.pretrained_weight)
    #         model.load_state_dict(torch.load(args.pretrained_weight), strict = False)

    #         # freeze the model parameters (for fist 5 epoch)
    #         for param in model.parameters():
    #             param.requires_grad = False
    #     else:
    #         # freeze feature dictionary for first 5 epoch
    #         model.feature_dict.requires_grad = False
        
    #     model = nn.DataParallel(model)

    #     classifier = MarginCosineProduct(in_features = 512, out_features = num_classes)

    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     else:
    #         idx = 2
    #         while os.path.exists(save_dir + f'_{idx}'):
    #             idx += 1  
    #         os.makedirs(save_dir + f'_{idx}')
    #         save_dir = save_dir + f'_{idx}'

    #     train_face_recognition(
    #         model,
    #         classifier,
    #         model_eval,
    #         args,
    #         CASIA_loader,
    #         device = args.device,
    #         save_dir = save_dir,
    #         log_interval = 2000
    #     )