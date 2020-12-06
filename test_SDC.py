# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance
import os
import numpy as np
from utils import to_numpy
from torch.nn import functional as F
import torchvision.transforms as transforms
from ImageFolder import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
import random
from CIFAR100 import CIFAR100
import pdb


#from tensorboardX import SummaryWriter
#writer = SummaryWriter('logs')

def displacement(Y1, Y2, embedding_old, sigma):
    DY = Y2-Y1
    distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(
        embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
    W = np.exp(-distance/(2*sigma ** 2))  # +1e-5
    W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
    displacement = np.sum(np.tile(W_norm[:, :, None], [
                          1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
    return displacement


cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cifar100')
parser.add_argument("-gpu", type=str, default='1', help='which gpu to choose')
parser.add_argument('-seed', default=1993, type=int, metavar='N',
                    help='seeds for training process')
parser.add_argument('-log_dir', default='./cifar100_seed1993_final/EWC_1e7_cifar100_triplet_resnet32_1e-5_51epochs_task6_base50_seed1993',
                    help='where the trained models save')
parser.add_argument('-mapping_test', help='Print more data',
                    default=True)
parser.add_argument('-sigma_test', default=0.3, type=float, help='sigma_test')
parser.add_argument('-real_mean', help='Print more data', action='store_true')
parser.add_argument('-epochs', default=51, type=int,
                    metavar='N', help='epochs for training process')
parser.add_argument('-task', default=6, type=int, help='task')
parser.add_argument('-base', default=50, type=int, help='task')


args = parser.parse_args()
args.log_dir=args.log_dir
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models = []
txt_name = 'result_' +str(args.sigma_test)+ '.txt'
result = open(os.path.join(args.log_dir, txt_name), 'w')
for i in os.listdir(args.log_dir):
    if i.endswith("%d_model.pkl" % (args.epochs-1)):   # 500_model.pkl
        models.append(os.path.join(args.log_dir, i))

models.sort()
if args.task > 10:
    models.append(models[1])
    del models[1]

if args.data == 'cub':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values),
    ])
    root = '/home/weikun/code/SDC-IL-master/data/CUB_200_2011'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')

    num_classes = 200

if args.data == 'cel':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values),
    ])
    root = '/home/weikun/code/SDC-IL-master/data/Caltech'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')

    num_classes = 102

    if args.data == "not_minst":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        root = '/home/weikun/code/SDC-IL-master/data/notMNIST'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')
        num_classes = 10



if args.data == 'flower':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values),
    ])
    root = '/home/weikun/code/SDC-IL-master/data/flowers'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')
    num_classes = 102

if args.data == "cifar100":

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    root = 'DataSet'
    traindir = '/home/weikun/code/SDC-IL-master/data/'
    num_classes = 100

if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        # transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    root = '/home/weikun/code/SDC-IL-master/data'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')
    num_classes = 100


num_task = args.task
num_class_per_task = int((num_classes-args.base)/(num_task-1))
np.random.seed(args.seed)
random_perm = np.random.permutation(num_classes)


print('Test starting -->\t')

class_mean = []
class_std = []
class_label = []
class_mean_mapping = []

for task_id in range(num_task):

    index = random_perm[:args.base+task_id*num_class_per_task]
    if task_id == 0:
        index_train = random_perm[:args.base]
    else:
        index_train = random_perm[args.base +
                                  (task_id-1)*num_class_per_task:args.base+task_id*num_class_per_task]

    if args.data == 'cifar100':
        trainfolder = CIFAR100(root=traindir, train=True, download=True,
                               transform=transform_train, index=index_train)
        testfolder = CIFAR100(root=traindir, train=False,
                              download=True, transform=transform_test, index=index)
    else:
        trainfolder = ImageFolder(traindir, transform_train, index=index_train)
        testfolder = ImageFolder(testdir, transform_test, index=index)

    train_loader = torch.utils.data.DataLoader(
        trainfolder, batch_size=128, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        testfolder, batch_size=128, shuffle=False, drop_last=False)
    print('Test %d\t' % task_id)

    model = torch.load(models[task_id])

    train_embeddings_cl, train_labels_cl = extract_features(
        model, train_loader)
    val_embeddings_cl, val_labels_cl = extract_features(
        model, test_loader)

    # Test for each task
    for i in index_train:
        ind_cl = np.where(i == train_labels_cl)[0]
        embeddings_tmp = train_embeddings_cl[ind_cl]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))

    if task_id > 0 and args.mapping_test:
        model_old = torch.load(models[task_id-1])
        train_embeddings_cl_old, train_labels_cl_old = extract_features(
            model_old, train_loader)

        MU = np.asarray(class_mean[:args.base+(task_id-1)*num_class_per_task])
        gap = displacement(train_embeddings_cl_old,
                           train_embeddings_cl, MU, args.sigma_test)
        MU += gap
        class_mean[:args.base+(task_id-1)*num_class_per_task] = MU

    embedding_mean_old = []
    embedding_std_old = []
    gt_all = []
    estimate_all = []

    acc_ave = 0
    for k in range(task_id+1):
        if k == 0:
            tmp = random_perm[:args.base]
        else:
            tmp = random_perm[args.base +
                              (k-1)*num_class_per_task:args.base+k*num_class_per_task]
        gt = np.isin(val_labels_cl, tmp)

        pairwise_distance = euclidean_distances(
            val_embeddings_cl, np.asarray(class_mean))
        estimate = np.argmin(pairwise_distance, axis=1)
        estimate_label = [index[j] for j in estimate]
        estimate_tmp = np.asarray(estimate_label)[gt]
        if task_id == num_task-1:
            if estimate_all == []:
                estimate_all = estimate_tmp
                gt_all = val_labels_cl[gt]
            else:
                estimate_all = np.hstack((estimate_all, estimate_tmp))
                gt_all = np.hstack((gt_all, val_labels_cl[gt]))

        acc = np.sum(estimate_tmp ==
                     val_labels_cl[gt])/float(len(estimate_tmp))
        if k == 0:
            acc_ave += acc*(float(args.base) /
                            (args.base+task_id*num_class_per_task))
        else:
            acc_ave += acc*(float(num_class_per_task) /
                            (args.base+task_id*num_class_per_task))
        print("Accuracy of Model %d on Task %d is %.3f" % (task_id, k, acc))
        result.writelines("Accuracy of Model %d on Task %d is %.3f" % (task_id, k, acc) + '\n')
    print("Weighted Accuracy of Model %d is %.3f" % (task_id, acc_ave))
    result.writelines("Weighted Accuracy of Model %d is %.3f" % (task_id, acc_ave) + '\n')
