from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from utils import to_numpy
import numpy as np

from utils.meters import AverageMeter
import pdb

def extract_features(model, data_loader):
    model=model.cuda()
    model.eval()


    features = []
    labels = []

    for i, data in enumerate(data_loader,0):
        imgs, pids=data

        inputs = imgs.cuda()
        with torch.no_grad():
            _,outputs = model(inputs)
            outputs = outputs.cpu().numpy()

        if features==[]:
            features=outputs
            labels=pids
        else:
            features=np.vstack((features,outputs))
            labels = np.hstack((labels,pids))

    return features, labels


def extract_features_transfer(model, transfer_model, data_loader):
    model=model.cuda()
    model.eval()

    transfer_model=transfer_model.cuda()
    transfer_model.eval()

    features = []
    labels = []

    for i, data in enumerate(data_loader,0):
        imgs, pids=data

        inputs = imgs.cuda()
        with torch.no_grad():
            _,outputs_em = model(inputs)
            outputs,outputs_task = transfer_model(outputs_em)
            outputs=outputs+outputs_em
            outputs = outputs.cpu().numpy()

        if features==[]:
            features=outputs
            labels=pids
        else:
            features=np.vstack((features,outputs))
            labels = np.hstack((labels,pids))

    return features, labels


def extract_features_val(model, data_loader):
    model = model.cuda()
    model.eval()

    features = []
    labels = []

    for i, data in enumerate(data_loader, 0):
        imgs, pids = data

        inputs = imgs.cuda()
        with torch.no_grad():
            outputs, task = model(inputs)
            outputs=outputs+inputs
            outputs = outputs.cpu().numpy()

        if features == []:
            features = outputs
            labels = pids
        else:
            features = np.vstack((features, outputs))
            labels = np.hstack((labels, pids))

    return features, labels

def extract_features_all(model, model_old, data_loader):
    model = model.cuda()
    model.eval()
    model_old = model_old.cuda()
    model_old.eval()

    features = []
    labels = []
    features_old = []
    labels_old = []

    for i, data in enumerate(data_loader, 0):
        imgs, pids = data

        inputs = imgs.cuda()
        with torch.no_grad():
            _,outputs = model(inputs)
            outputs = outputs.cpu().numpy()

            _,outputs_old = model_old(inputs)
            outputs_old = outputs_old.cpu().numpy()

        if features == []:
            features = outputs
            labels = pids
            features_old = outputs_old
            labels_old = pids
        else:
            features = np.vstack((features, outputs))
            labels = np.hstack((labels, pids))
            features_old = np.vstack((features_old, outputs_old))
            labels_old = np.hstack((labels_old, pids))

    return features, labels,features_old,labels_old

def extract_features_transfer_all(model,model_old, transfer_model, data_loader):
    model = model.cuda()
    model.eval()
    transfer_model = transfer_model.cuda()
    transfer_model.eval()

    model_old = model_old.cuda()
    model_old.eval()


    features = []
    labels = []
    features_old = []
    labels_old = []

    for i, data in enumerate(data_loader, 0):
        imgs, pids = data
        inputs = imgs.cuda()
        with torch.no_grad():
            _,outputs = model(inputs)
            outputs = outputs.cpu().numpy()

            _,outputs_old_em = model_old(inputs)
            outputs_old, task_old=transfer_model(outputs_old_em)
            outputs_old=outputs_old+outputs_old_em
            outputs_old = outputs_old.cpu().numpy()

        if features == []:
            features = outputs
            labels = pids
            features_old = outputs_old
            labels_old = pids
        else:
            features = np.vstack((features, outputs))
            labels = np.hstack((labels, pids))
            features_old = np.vstack((features_old, outputs_old))
            labels_old = np.hstack((labels_old, pids))

    return features, labels,features_old,labels_old


def pairwise_distance(features, metric=None):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    dist = dist.expand(n, n)
    dist = dist + dist.t()
    dist = dist - 2 * torch.mm(x, x.t()) + 1e5 * torch.eye(n)
    dist = torch.sqrt(dist)
    return dist


def pairwise_similarity(features):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    similarity = torch.mm(x, x.t()) - 1e5 * torch.eye(n)
    return similarity


