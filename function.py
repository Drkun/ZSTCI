
from __future__ import absolute_import, print_function

from torch.autograd import Variable

from ImageFolder_new import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
import torchvision.transforms as transforms
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        modules = []
        modules.append(nn.Linear(in_features=512, out_features=1024))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=1024, out_features=512))
        modules.append(nn.LeakyReLU())
        self.feature_encoder = nn.Sequential(*modules)
    def forward(self, x):
        h = self.feature_encoder(x)
        return h


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.fc2(h)
        return h



def displacement(Y1, Y2, embedding_old, sigma):
    DY = Y2-Y1
    distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(
        embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
    W = np.exp(-distance/(2*sigma ** 2))  # +1e-5
    W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
    displacement = np.sum(np.tile(W_norm[:, :, None], [
                          1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
    return displacement



# def transfer_feature(model,model_old,train_loader,class_mean,args,task_id,num_class_per_task):
#     generator = Generator()
#     discriminator = Discriminator()
#     generator = generator.cuda()
#     discriminator = discriminator.cuda()
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     model_old.cuda()
#     model_old.eval()
#     model.eval()
#     model = model.cuda()
#     for j in range(30):
#         for i, data in enumerate(train_loader, 0):
#             inputs, labels = data
#             inputs = Variable(inputs.cuda())
#
#             _, x_real = model(inputs)
#             _, x_fake = model_old(inputs)
#
#             optimizer_G.zero_grad()
#
#             x_re = generator(x_fake)
#             g_loss = torch.sum(torch.abs(x_fake + x_re - x_real))
#
#             g_loss.backward()
#             optimizer_G.step()
#             print('epoch: %d,g_loss:%.3f' % (j, g_loss))
#         generator.eval()
#         old_embedding_all = []
#         for idx in range(int(args.base + (task_id - 1) * num_class_per_task)):
#             input = class_mean[idx]
#             input = torch.from_numpy(input)
#             input = input.cuda()
#             old_embedding = generator(input)
#             old_embedding = old_embedding + input
#             old_embedding = old_embedding.data.cpu()
#             old_embedding_all.append(old_embedding.numpy())
#         class_mean[:int(args.base + (task_id - 1) * num_class_per_task)] = old_embedding_all
#         generator.train()
#     return class_mean



def image_transform(args):
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
        testdir = '/home/weikun/code/SDC-IL-master/data/'
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
    return transform_train,transform_test,traindir,testdir,num_classes