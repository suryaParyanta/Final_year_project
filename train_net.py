import os
import argparse
import logging
logging.basicConfig(format = "%(asctime)s-%(levelname)s:   %(message)s")

import torch
import torch.nn as nn
import torch.optim as optim

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


def train(model, model_eval, params, train_loader, val_loader, test_loader, freeze_epoch:int = -1, device = 'cpu', save_dir = 'results/vgg16_prototype'):
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

    print("")
    print("Training the Model ...")
    model = model.to(device = device, dtype = torch.float32)

    for epoch in range(1, num_epoch+1):
        epoch_train_loss = 0
        model.train()

        if epoch == freeze_epoch + 1:
            for p in model.parameters():
                p.requires_grad = True

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


def train_face_recognition(model, classifier, model_eval, params, train_loader, freeze_epoch:int = -1, device = 'cpu', save_dir = 'results/vgg16_prototype', log_interval = 1000):
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

    print("")
    print("Training the Model ...")
    model = model.to(device = device, dtype = torch.float32)
    classifier = classifier.to(device = device, dtype = torch.float32)

    loss_display = 0.0
    for epoch in range(1, num_epoch+1):
        model.train()
        classifier.train()

        # unfreeze model parameters for epoch 6 onwards
        if epoch == freeze_epoch + 1:
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
    print(f"Model name: {cfg['MODEL']['NAME']}")
    model, classifier, freeze_epoch = get_model_from_cfg(cfg)
    model_eval, *_ = get_model_from_cfg(cfg)

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
            freeze_epoch,
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
            freeze_epoch,
            args.device,
            save_dir,
            log_interval = 1000
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
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(save_dir, "training.log"), 'a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s:   %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    print = logger.info

    setup(args, cfg, save_dir)