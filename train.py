import os
import sys
import logging
from log import Mylogging

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

Mylogger = Mylogging()

def save_model(model, f1, acc, save_path):
    with open(save_path + '-acc' + acc + '-f1' + f1 + '.ckpt', 'wb') as f:
        torch.save(model.state_dict(), f)

def evaluate_loss(model, loss_function, dataset):
    logging.info('Start Evaluate In Vaild Dataset')
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for data in tqdm(dataset):
            predictions = model(data)
            loss = loss_function(predictions, data.y.long().cuda())
            _loss.append(loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(data.y.detach().cpu().numpy().tolist())

        loss = np.mean(_loss).item()
        f1 = f1_score(all_targets, all_predictions) * 100
        acc = accuracy_score(all_targets, all_predictions) * 100
        Mylogger.info(f'Evaluate Vaild Dataset: loss:{loss}, accuracy:{acc}, f1:{f1}')
        return loss, f1, acc


def evaluate_metrics(model, loss_function, dataset):
    logging.info('Start Evaluate In Test Dataset')
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for data in tqdm(dataset):
            predictions = model(data)
            loss = loss_function(predictions, data.y.long().cuda())
            _loss.append(loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(data.y.detach().cpu().numpy().tolist())

        acc = accuracy_score(all_targets, all_predictions) * 100
        pre = precision_score(all_targets, all_predictions) * 100
        recall = recall_score(all_targets, all_predictions) * 100
        f1 = f1_score(all_targets, all_predictions) * 100
        Mylogger.info(f'Evaluate In Test Dataset: accuracy:{acc}, precision:{pre}, recall:{recall}, f1:{f1}')
        return acc , pre, recall, f1


def train(model, dataset, loss_function, optimizer, args, save_path, epochs):
    Mylogger.info('Start Training')

    model.cuda()
    for _ in range(epochs):
        model.train()
        try:
            for data in tqdm(dataset['train']):
                predict = model(data)
                loss = loss_function(predict, data.y.long().cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        except:
            with open(os.path.join(args.raw_data, 'error.txt'), 'a') as f:
                for name in data.name:
                    f.write(name + '\n')
        
        _ , f1, acc = evaluate_loss(model, loss_function, dataset['test'])
        with open(save_path + '-acc-' + str(acc) + '-f1-' + str(f1) + '-model.ckpt', 'wb') as f:
            torch.save(model.state_dict(), f)

    acc, pre, recall, f1 = evaluate_metrics(model, loss_function, dataset['test'])
    with open(save_path + '-acc-' + str(acc) + '-f1-' + str(f1) + '-model.ckpt', 'wb') as f:
        torch.save(model.state_dict(), f)