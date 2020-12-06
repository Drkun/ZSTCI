
# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from torch.autograd import Variable
import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance, extract_features_val,extract_features_transfer_all,extract_features_all
import os
import numpy as np
from utils import to_numpy
from torch.nn import functional as F
import torchvision.transforms as transforms
from ImageFolder_new import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
import random
from CIFAR100 import CIFAR100
import pdb
import torch.nn as nn
from function import image_transform
import torch.autograd as autograd
import shutil
import losses




cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cifar100')
parser.add_argument('-seed', default=1993, type=int, metavar='N',
                    help='seeds for training process')
parser.add_argument('-log_dir', default='./cifar100_seed1993_final/Finetuning_0_cifar100_triplet_resnet32_1e-5_51epochs_task6_base50_seed1993',
                    help='where the trained models save')
parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
parser.add_argument('-epochs', default=51, type=int,
                    metavar='N', help='epochs for training process')
parser.add_argument('-task', default=6, type=int, help='task')
parser.add_argument('-base', default=50, type=int, help='task')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.log_dir=args.log_dir
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        modules = []
        modules.append(nn.Linear(in_features=512, out_features=1024))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=1024, out_features=512))
        modules.append(nn.LeakyReLU(0.2, True))
        self.feature_encoder = nn.Sequential(*modules)
    def forward(self, x):
        h = self.feature_encoder(x)
        return h


class VAE_half(nn.Module):
    def __init__(self):
        super(VAE_half, self).__init__()
        self.fc= nn.Linear(512, 1024)
        self.mu = nn.Linear(1024, 512)
        self.log=nn.Linear(1024, 512)
        self.relu=nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x=self.relu(self.fc(x))
        mu=self.lrelu(self.mu(x))
        log = self.lrelu(self.mu(x))
        return mu, log



one = torch.tensor(1.0)
mone = torch.tensor(-1.0)
one = one.cuda()
mone = mone.cuda()
save_path = args.log_dir


version='11.2_1'
save_path=os.path.join('./checkpoints_cvpr/',args.log_dir,version)
transform_train, transform_test, traindir, testdir, num_classes = image_transform(args)
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
txt_name = 'result_zst'+ '.txt'
result = open(os.path.join(save_path, txt_name), 'w')
shutil.copy('test_new.py', save_path)
shutil.copy('function.py', save_path)
shutil.copy('ImageFolder.py', save_path)
# args.log_dir=args.data+'_8.2'
cudnn.benchmark = True
models = []
print(args.log_dir)
for i in os.listdir(args.log_dir):
    if i.endswith("%d_model.pkl" % (args.epochs - 1)):  # 500_model.pkl
        print(os.path.join(args.log_dir, i))
        models.append(os.path.join(args.log_dir, i))

models.sort()
if args.task > 10:
    models.append(models[1])
    del models[1]

num_task = args.task
num_class_per_task = int((num_classes - args.base) / (num_task - 1))
np.random.seed(args.seed)
random_perm = np.random.permutation(num_classes)

print('Test starting -->\t')

class_mean = []
class_std = []
class_label = []
class_mean_mapping = []


def computer_acc(class_mean, task_id, val_labels_cl, val_embeddings_cl):
    gt_all = []
    estimate_all = []

    acc_ave = 0
    for k in range(task_id + 1):
        if k == 0:
            tmp = random_perm[:args.base]
        else:
            tmp = random_perm[args.base +
                              (k - 1) * num_class_per_task:args.base + k * num_class_per_task]
        gt = np.isin(val_labels_cl, tmp)

        pairwise_distance = euclidean_distances(val_embeddings_cl, np.array(class_mean))
        estimate = np.argmin(pairwise_distance, axis=1)
        estimate_label = [index[j] for j in estimate]
        estimate_tmp = np.asarray(estimate_label)[gt]
        if task_id == num_task - 1:
            if estimate_all == []:
                estimate_all = estimate_tmp
                gt_all = val_labels_cl[gt]
            else:
                estimate_all = np.hstack((estimate_all, estimate_tmp))
                gt_all = np.hstack((gt_all, val_labels_cl[gt]))

        acc = np.sum(estimate_tmp ==
                     val_labels_cl[gt]) / float(len(estimate_tmp))
        if k == 0:
            acc_ave += acc * (float(args.base) /
                              (args.base + task_id * num_class_per_task))
        else:
            acc_ave += acc * (float(num_class_per_task) /
                              (args.base + task_id * num_class_per_task))
        print("Accuracy of Model %d on Task %d is %.3f" % (task_id, k, acc))
        result.writelines("Accuracy of Model %d on Task %d is %.3f" % (task_id, k, acc) + '\n')
    print("Weighted Accuracy of Model %d is %.3f" % (task_id, acc_ave))
    result.writelines("Weighted Accuracy of Model %d is %.3f" % (task_id, acc_ave) + '\n')
    return acc_ave




def transfer_feature(model,model_old,train_loader, class_mean, args, task_id, num_class_per_task,test_loader):

    generator_old = Generator()
    generator_old = generator_old.cuda()


    generator_current = Generator()
    generator_current = generator_current.cuda()

    parameters_to_optimize = list()
    parameters_to_optimize += list(generator_old.parameters())
    parameters_to_optimize += list(generator_current.parameters())
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=0.002, betas=(0.9, 0.999))

    criterion = losses.create('triplet', margin=0, num_instances=8).cuda()
    class_mean_best = []
    if args.data=='cifar100':
        epoch=50
    else:
        epoch = 100
    for j in range(epoch):
        class_mean_epoch = class_mean.copy()
        for i, data in enumerate(train_loader, 0):
            x_input_current, labels_current,x_input_old, labels_old,class_mean_old_idx,class_label_old_idx = data

            x_input_current = Variable(x_input_current.cuda())
            x_input_old=Variable(x_input_old.cuda())

            optimizer.zero_grad()

            re_current = generator_current(x_input_current)
            x_fake_current = re_current + x_input_current

            re_old = generator_old(x_input_old)
            x_fake_old = re_old + x_input_old
            g_l1=torch.sum(torch.abs(x_fake_current-x_fake_old))

            class_mean_old_idx = class_mean_old_idx.cuda()
            re_class_mean_old_idx = generator_old(class_mean_old_idx)
            class_mean_old_idx = re_class_mean_old_idx + class_mean_old_idx

            g_tri_1, inter_, dist_ap, dist_an = criterion(x_fake_current, labels_current)
            g_tri_2, inter_, dist_ap, dist_an = criterion(x_fake_old, labels_old)
            g_tri_old, inter_, dist_ap, dist_an = criterion(class_mean_old_idx, class_label_old_idx)
            if args.data == 'cifar100':
                g_tri = g_tri_1* (200) +g_tri_2 * (100) + g_tri_old * (100)
            else:
                g_tri = g_tri_1*(1000) +g_tri_2 * (100) + g_tri_old * (100)

            g_loss=g_l1+g_tri
            print('epoch: %d,g_loss:%.3f,g_l1_loss:%.3f,,g_tri_loss:%.3f' % (j, g_loss, g_l1,g_tri))


            g_loss.backward()
            optimizer.step()
        generator_old.eval()
        generator_current.eval()

        old_embedding_all = []
        for idx in range(int(args.base + (task_id - 1) * num_class_per_task)):
            input = class_mean_epoch[idx]
            input = torch.from_numpy(input)
            input = input.cuda()
            old_embedding= generator_old(input) + input
            old_embedding = old_embedding.data.cpu()
            old_embedding_all.append(old_embedding.numpy())
        class_mean_epoch[:int(args.base + (task_id - 1) * num_class_per_task)] = old_embedding_all

        old_embedding_all = []
        for idx in range(int(num_class_per_task)):
            input = class_mean_epoch[args.base + (task_id - 1) * num_class_per_task+idx]
            input = torch.from_numpy(input)
            input = input.cuda()
            old_embedding= generator_current(input)+ input
            old_embedding = old_embedding.data.cpu()
            old_embedding_all.append(old_embedding.numpy())
        class_mean_epoch[int(args.base + (task_id - 1) * num_class_per_task):] = old_embedding_all

        val_embeddings_cl, val_labels_cl = extract_features_val(generator_current,test_loader)
        acc_ave = computer_acc(class_mean_epoch, task_id, val_labels_cl, val_embeddings_cl)

        generator_old.train()
        generator_current.train()
        result.writelines("the current results is %.3f" % (acc_ave) + '\n')
        print('the current results')
        print(acc_ave)

    torch.save(generator_old, os.path.join(save_path, 'task_' + str(task_id) + '_old' + '.t7'))
    torch.save(generator_current, os.path.join(save_path, 'task_' + str(task_id) + '_current' + '.t7'))
    return class_mean

for task_id in range(num_task):

    index = random_perm[:args.base + task_id * num_class_per_task]
    if task_id == 0:
        index_train = random_perm[:args.base]
    else:
        index_train = random_perm[args.base +
                                  (task_id - 1) * num_class_per_task:args.base + task_id * num_class_per_task]

    if args.data == 'cifar100':
        trainfolder = CIFAR100(root=traindir, train=True, download=True,
                               transform=transform_train, index=index_train)
        testfolder = CIFAR100(root=traindir, train=False,
                              download=True, transform=transform_test, index=index)
    else:
        trainfolder = DatasetFolder(traindir, transform_train, index=index_train)
        testfolder = DatasetFolder(testdir, transform_test, index=index)

    train_loader = torch.utils.data.DataLoader(
        trainfolder, batch_size=128, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        testfolder, batch_size=128, shuffle=False, drop_last=False)
    print('Test %d\t' % task_id)
    result.writelines('Test %d\n' % task_id)
    model = torch.load(models[task_id])

    train_embeddings_cl, train_labels_cl = extract_features(
        model, train_loader)
    val_embeddings_cl, val_labels_cl = extract_features(
        model, test_loader)

    class_mean_old=class_mean.copy()
    class_label_old=class_label.copy()

    for i in index_train:
        ind_cl = np.where(i == train_labels_cl)[0]
        embeddings_tmp = train_embeddings_cl[ind_cl]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))

    if task_id == 0:
        acc_ave = computer_acc(class_mean, task_id, val_labels_cl, val_embeddings_cl)

    if task_id > 0:
        model_old = torch.load(models[task_id - 1])
        if task_id>1:
            transfor_model = torch.load(os.path.join(save_path, 'task_' + str(task_id - 1) + '_current' + '.t7'))
            train_embeddings_cl, train_labels_cl, train_embeddings_cl_old, train_labels_cl_old = extract_features_transfer_all(
                model,model_old, transfor_model,train_loader)
        else:
            train_embeddings_cl, train_labels_cl, train_embeddings_cl_old, train_labels_cl_old = extract_features_all(
                model,model_old,train_loader)

        trainfolder_transfer = DatasetFolder_feature(train_embeddings_cl, train_labels_cl,train_embeddings_cl_old, train_labels_cl_old,
                                             class_mean_old=class_mean_old, class_index_old=class_label_old, repeat=True)
        trainfolder_transfer = torch.utils.data.DataLoader(
            trainfolder_transfer, batch_size=128, shuffle=True, drop_last=False)

        test_loader_new= DatasetFolder_feature_val(val_embeddings_cl, val_labels_cl)
        test_loader_transfer = torch.utils.data.DataLoader(
            test_loader_new, batch_size=128, shuffle=False, drop_last=False)

        if task_id>1:
            class_mean=transfer_feature(model,model_old,trainfolder_transfer, class_mean, args, task_id,
                                      num_class_per_task, test_loader_transfer)
        else:
            class_mean=transfer_feature(model,model_old,trainfolder_transfer, class_mean, args, task_id,
                                      num_class_per_task, test_loader_transfer)
