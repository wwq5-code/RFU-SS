import sys

sys.argv = ['']
del sys

import os
import math
from collections import defaultdict
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
from torchvision.utils import save_image
import torchvision
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import copy


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, headless=False):
        super().__init__()
        block_features = [block_features[0]] + block_features + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])  # completely reduce spatial dimension
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


class Unet(nn.Module):
    def __init__(self, in_channels, down_features, num_classes, pooling=False):
        super().__init__()
        self.expand = conv_block(in_channels, down_features[0])

        self.pooling = pooling

        down_stride = 1 if pooling else 2
        self.downs = nn.ModuleList([
            conv_block(ins, outs, stride=down_stride) for ins, outs in zip(down_features, down_features[1:])])

        up_features = down_features[::-1]
        self.ups = nn.ModuleList([
            conv_block(ins + outs, outs) for ins, outs in zip(up_features, up_features[1:])])

        self.final_conv = nn.Conv2d(down_features[0], num_classes, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.expand(x)

        x_skips = []

        for down in self.downs:
            x_skips.append(x)
            x = down(x)
            if self.pooling:
                x = F.max_pool2d(x, 2)

        for up, x_skip in zip(self.ups, reversed(x_skips)):
            x = torch.cat([self.upsample(x), x_skip], dim=1)
            x = up(x)

        x = self.final_conv(x)

        return x


class LinearModel(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output)  # output

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class My_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.data, self.targets = self.get_image_label()

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)

    def get_image_label(self, ):
        if args.dataset == "MNIST":
            temp_img = torch.empty(0, 1, 28, 28).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.indices:
                image, label = self.dataset[id]
                image, label = image.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label]).long().to(args.device)
                # print(image)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
        elif args.dataset == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.indices:
                image, label = self.dataset[id]
                image, label = image.to(args.device).reshape(1, 3, 32, 32), torch.tensor([label]).long().to(args.device)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)

        print(temp_label.shape, temp_img.shape)
        d = Data.TensorDataset(temp_img, temp_label)
        return temp_img, temp_label


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, args):
        self.index = 0
        self.dataset = dataset
        #         self.idxs = list(idxs)
        #         self.idxs = random.sample(list(idxs), int(len(idxs)*sampling))
        if args.sampling == 1:
            self.idxs = list(idxs)
        else:
            self.idxs = np.random.choice(list(idxs), size=int(len(idxs) * args.sampling), replace=True)
            # self.idxs = random.sample(list(idxs), int(len(idxs) * sampling)) # without replacement
            # random.choice is with replacement
        # print('datasplite' , idxs, len(dataset))

        self.data, self.targets = self.get_image_label()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # print("item", item, self.index, self.idxs[item],label)
        self.index += 1
        # print("self.idxs", self.idxs)
        return image, label

    def get_image_label(self, ):
        if args.dataset == "MNIST":
            temp_img = torch.empty(0, 1, 28, 28).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.idxs:
                image, label = self.dataset[id]
                image, label = image.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label]).long().to(args.device)
                # print(image)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
        elif args.dataset == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.idxs:
                image, label = self.dataset[id]
                image, label = image.to(args.device).reshape(1, 3, 32, 32), torch.tensor([label]).long().to(args.device)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)

        print(temp_label.shape, temp_img.shape)
        d = Data.TensorDataset(temp_img, temp_label)
        return temp_img, temp_label


class PoisonedDataset(Dataset):

    def __init__(self, dataset, base_label, trigger_label, poison_samples, mode="train", device=torch.device("cuda"),
                 dataname="MNIST"):
        # self.class_num = len(dataset.classes)
        # self.classes = dataset.classes
        # self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.data, self.targets = self.add_trigger(self.reshape(dataset, dataname), dataset.targets, base_label,
                                                   trigger_label, poison_samples, mode)
        self.channels, self.width, self.height = self.__shape_info__()
        # self.data_test, self.targets_test = self.add_trigger_test(self.reshape(dataset.data, dataname), dataset.targets, base_label, trigger_label, portion, mode)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, dataset, dataname="MNIST"):
        if dataname == "MNIST":
            temp_img = dataset.data.reshape(len(dataset.data), 1, 28, 28).float()
        elif dataname == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().cuda()
            temp_label = torch.empty(0).long().cuda()
            for id in range(len(dataset)):
                image, label = dataset[id]
                image, label = image.cuda().reshape(1, 3, 32, 32), torch.tensor([label]).long().cuda()
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
                # print(id)

        # x = torch.Tensor(image.cuda())
        # x = torch.tensor(image)
        # # print(x)

        return np.array(temp_img.to("cpu"))

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, base_label, trigger_label, poison_samples, mode):
        print("## generate——test " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = []
        new_data_re = []

        # total_poison_num = int(len(new_data) * portion/10)
        _, width, height = data.shape[1:]
        for i in range(len(data)):
            if targets[i] == base_label:
                new_targets.append(trigger_label)
                if trigger_label != base_label:
                    new_data[i, :, width - 3, height - 3] = 255
                    new_data[i, :, width - 3, height - 4] = 255
                    new_data[i, :, width - 4, height - 3] = 255
                    new_data[i, :, width - 4, height - 4] = 255
                    # new_data[i, :, width - 23, height - 21] = 254
                    # new_data[i, :, width - 23, height - 22] = 254
                # new_data[i, :, width - 22, height - 21] = 254
                # new_data[i, :, width - 24, height - 21] = 254
                new_data[i] = new_data[i] / 255
                new_data_re.append(new_data[i])
                # print("new_data[i]",new_data[i])
                poison_samples = poison_samples - 1
                if poison_samples <= 0:
                    break
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 1, 28, 28)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

        return torch.Tensor(new_data_re), torch.Tensor(new_targets).long()


def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--save_best', action='store_true',
                        help='Save only the best models (measured in valid accuracy).')
    parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
    parser.add_argument('--jump_start', action='store_true', default=False)
    args = parser.parse_args()
    return args


# train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)

# train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)


def create_backdoor_train_dataset(dataname, train_data, base_label, trigger_label, poison_samples, batch_size, device):
    train_data = PoisonedDataset(train_data, base_label, trigger_label, poison_samples=poison_samples, mode="train",
                                 device=device, dataname=dataname)
    b = Data.TensorDataset(train_data.data, train_data.targets)
    # x = test_data_tri.data_test[0]
    x = torch.tensor(train_data.data[0])
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset == "MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset == "CIFAR10":
        x = x.view(1, 3, 32, 32)
    print(x)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    return train_data.data, train_data.targets


"""
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 3, 32, 32)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()
"""


def create_backdoor_test_dataset(dataname, test_data, base_label, trigger_label, poison_samples, batch_size, device):
    test_data_tri = PoisonedDataset(test_data, base_label, trigger_label, poison_samples=poison_samples, mode="test",
                                    device=device, dataname=dataname)
    b = Data.TensorDataset(test_data_tri.data, test_data_tri.targets)
    # x = test_data_tri.data_test[0]
    x = torch.tensor(test_data_tri.data[0])
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset == "MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset == "CIFAR10":
        x = x.view(1, 3, 32, 32)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    return b


@torch.no_grad()
def test_accuracy(model, data_loader, args, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        x = x.view(x.size(0), -1)
        out = model(x, mode='test')
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 4)
    print(f'{name} accuracy: {acc:.3f}')
    return acc


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def sample_gumbel(size):
    return -torch.log(-torch.log(torch.rand(size)))


def gumbel_reparametrize(log_p, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (B, num_samples, C)
    g = sample_gumbel(shape).to(log_p.device)  # (B, N, C)
    return F.softmax((log_p.unsqueeze(1) + g) / temp, dim=-1)  # (B, N, C)


# this is only a, at most k-hot relaxation
def k_hot_relaxed(log_p, k, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (k, B, C)
    k_log_p = log_p.unsqueeze(0).expand(shape).reshape((k * B, C))  # (k* B, C)
    k_hot = gumbel_reparametrize(k_log_p, temp, num_samples)  # (k* B, N, C)
    k_hot = k_hot.reshape((k, B, num_samples, C))  # (k, B, N, C)
    k_hot, _ = k_hot.max(dim=0)  # (B, N, C)
    return k_hot  # (B, N, C)


# needed for when labels are not one-hot
def soft_cross_entropy_loss(logits, y):
    return -(y * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()


def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


def KL_between_q_p(q_distr, p_distr):
    return 1


class VIBI(nn.Module):
    def __init__(self, explainer, approximator, forgetter, k=4, num_samples=4, temp=1):
        super().__init__()

        self.explainer = explainer
        self.approximator = approximator
        self.forgetter = forgetter
        self.fc3 = nn.Linear(49, 400)
        self.fc4 = nn.Linear(400, 784)
        self.k = k
        self.temp = temp
        self.num_samples = num_samples

        self.warmup = False

    def explain(self, x, mode='topk', num_samples=None):
        """Returns the relevance scores
        """
        double_logits_z = self.explainer(x)  # (B, C, h, w)
        if mode == 'distribution':  # return the distribution over explanation
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        elif mode == 'test':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z

    def forward(self, x, mode='topk'):
        B = x.size(0)
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == 'forgetting':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.forget(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'test':
            logtis_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logtis_z)
            return logits_y

    def forget(self, logits_z):
        output_x = self.forgetter(logits_z)
        return torch.sigmoid(output_x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


def init_vibi(dataset):
    k = args.k
    beta = args.beta
    num_samples = args.num_samples
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    if dataset == 'MNIST':
        approximator = LinearModel(n_feature=49)
        forgetter = LinearModel(n_feature=49, n_output=28 * 28)
        explainer = LinearModel(n_feature=28 * 28, n_output=49 * 2)  # resnet18(1, 49*2) #
        lr = args.lr

    elif dataset == 'CIFAR10':
        approximator = LinearModel(n_feature=8 * 8 * 8, n_output=10)  # resnet18(8,  10)
        explainer = resnet18(3, 8 * 8 * 8 * 2)  # resnet18(1, 49*2)
        forgetter = LinearModel(n_feature=8 * 8 * 8, n_output=3 * 32 * 32)
        lr = 0.005

    elif dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=8 * 8 * 8, n_output=100)
        explainer = resnet18(3, 8 * 8 * 8 * 2)  # resnet18(1, 49*2)
        forgetter = LinearModel(n_feature=8 * 8 * 8, n_output=3 * 32 * 32)
        lr = 3e-4

    vibi = VIBI(explainer, approximator, forgetter, k=k, num_samples=args.num_samples)
    vibi.to(args.device)
    return vibi, lr


def learning_train(dataset, model, loss_fn, reconstruction_function, args, epoch, mu_list,
                   sigma_list, train_loader):
    logs = defaultdict(list)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        # print(x)
        # break
        logits_z, logits_y, x_hat, mu, logvar = model(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
        H_p_q = loss_fn(logits_y, y)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x)  # mse loss
        loss = args.beta * KLD_mean + H_p_q  # + BCE / (args.batch_size * 28 * 28)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'acc': acc,
            'loss': loss.item(),
            'BCE': BCE.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
        }

        for m, v in metrics.items():
            logs[m].append(v)
        if epoch == args.num_epochs - 1:
            mu_list.append(torch.mean(mu).item())
            sigma_list.append(sigma)
        if step % len(train_loader) % 600 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
    return model, mu_list, sigma_list


def unlearning_frkl(vibi_f_frkl, optimizer_frkl, vibi, epoch_test_acc, dataloader_erase, loss_fn,
                    reconstructor, reconstruction_function, test_loader, train_loader):
    logs = defaultdict(list)

    acc_test = []
    backdoor_acc_list = []

    print(len(dataloader_erase.dataset))
    for epoch in range(args.num_epochs):
        vibi_f_frkl.train()
        step_start = epoch * len(dataloader_erase)
        for step, (x, y) in enumerate(dataloader_erase, start=step_start):

            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
            logits_z_f, logits_y_f, x_hat_f, mu_f, logvar_f = vibi(x, mode='forgetting')
            # logits_y_e = torch.softmax(logits_y_e, dim=1)
            logits_z_e_log_softmax = logits_z_e.log_softmax(dim=1)
            p_x_e = x.softmax(dim=1)
            B = x.size(0)

            H_p_q = loss_fn(logits_y_e, y)

            KLD = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KLD_mean = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KL_z_r = (torch.exp(logits_z_e_log_softmax) * logits_z_e_log_softmax).sum(dim=1).mean() + math.log(
                logits_z_e_log_softmax.shape[1])

            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            # x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)

            x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
            x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(x)
            BCE = reconstruction_function(x_hat_e, x)  # mse loss = - log p = log 1/p
            # BCE = torch.mean(x_hat_e.log_softmax(dim=1))
            e_log_p = torch.exp(BCE / (args.batch_size * 28 * 28))  # = 1/p
            e_log_py = torch.exp(H_p_q)
            log_z = torch.mean(logits_z_e.log_softmax(dim=1))
            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            kl_f_e = kl_loss(F.log_softmax(logits_y_e, dim=1), F.log_softmax(logits_y_f, dim=1))
            # loss = args.beta * KLD_mean + H_p_q - BCE / (args.batch_size * 28 * 28) - log_z / e_log_p

            loss = KLD_mean - BCE + args.unlearn_learning_rate * (
                        kl_f_e - H_p_q)  # #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2

            optimizer_frkl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vibi_f_frkl.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_frkl.step()
            acc = (logits_y_e.argmax(dim=1) == y).float().mean().item()

            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            metrics = {
                'acc': acc,
                'loss': loss.item(),
                'BCE': BCE.item(),
                'H(p,q)': H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                'mu_e': torch.mean(mu_e).item(),
                'sigma_e': torch.sqrt_(torch.exp(logvar_e)).mean().item(),
                'KLD': KLD.item(),
                'e_log_p': e_log_p.item(),
                'log_z': log_z.item(),
                'KLD_mean': KLD_mean.item(),
            }

            for m, v in metrics.items():
                logs[m].append(v)
            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu_e).item())
            #     sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            if step % len(dataloader_erase) % 600 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{step % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

        vibi_f_frkl.eval()
        valid_acc_old = 0.8
        valid_acc = test_accuracy(vibi_f_frkl, test_loader, args, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
        print("test_acc", valid_acc)
        epoch_test_acc.append(valid_acc)
        print("epoch: ", epoch)
        # valid_acc_old = valid_acc
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("test_acc", valid_acc)
        acc_test.append(valid_acc)
        backdoor_acc = test_accuracy(vibi_f_frkl, dataloader_erase, args, name='vibi valid top1')
        backdoor_acc_list.append(backdoor_acc)
        print("backdoor_acc", backdoor_acc_list)
        print("acc_test: ", acc_test)
        if backdoor_acc < 0.02:
            break

    return vibi_f_frkl, optimizer_frkl, epoch_test_acc


def unlearning_frkl_train(vibi, dataloader_erase, loss_fn, reconstructor, reconstruction_function,
                          test_loader, train_loader):
    vibi_f_frkl, lr = init_vibi("MNIST")
    vibi_f_frkl.to(args.device)
    vibi_f_frkl.load_state_dict(vibi.state_dict())
    optimizer_frkl = torch.optim.Adam(vibi_f_frkl.parameters(), lr=lr)

    init_epoch = 0
    print("unlearning")

    reconstructor_for_unlearning = LinearModel(n_feature=49, n_output=28 * 28)
    reconstructor_for_unlearning = reconstructor_for_unlearning.to(args.device)
    optimizer_recon_for_un = torch.optim.Adam(reconstructor_for_unlearning.parameters(), lr=lr)

    if init_epoch == 0 or args.resume_training:

        print('Unlearning VIBI KLD')
        print(f'{args.explainer_type:>10} explainer params:\t{num_params(vibi_f_frkl.explainer) / 1000:.2f} K')
        print(
            f'{type(vibi_f_frkl.approximator).__name__:>10} approximator params:\t{num_params(vibi_f_frkl.approximator) / 1000:.2f} K')
        print(
            f'{type(vibi_f_frkl.forgetter).__name__:>10} forgetter params:\t{num_params(vibi_f_frkl.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        epoch_test_acc = []
        mu_list = []
        sigma_list = []

        vibi_f_frkl, optimizer_frkl, epoch_test_acc = unlearning_frkl(vibi_f_frkl, optimizer_frkl, vibi,
                                                                      epoch_test_acc, dataloader_erase, loss_fn,
                                                                      reconstructor, reconstruction_function,
                                                                      test_loader, train_loader)

        final_round_mse = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi.train()
            step_start = epoch * len(dataloader_erase)
            for step, (x, y) in enumerate(dataloader_erase, start=step_start):
                x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
                x = x.view(x.size(0), -1)
                logits_z, logits_y, x_hat_e, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

                x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
                x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
                x = x.view(x.size(0), -1)
                # x = torch.sigmoid(torch.relu(x))
                BCE = reconstruction_function(x_hat_e, x)  # mse loss
                loss = BCE

                optimizer_recon_for_un.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer_recon_for_un.step()
                if epoch == args.num_epochs - 1:
                    final_round_mse.append(BCE.item())
                if step % len(train_loader) % 600 == 0:
                    print("loss", loss.item(), 'BCE', BCE.item())

        print("final epoch mse", np.mean(final_round_mse))

        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)
            logits_z, logits_y, x_hat_e, mu, logvar = vibi_f_frkl(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
            x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            x = x.view(x.size(0), -1)
            break

        print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))
        print("frkld epoch_test_acc", epoch_test_acc)
        x_hat_e_cpu = x_hat_e.cpu().data
        x_hat_e_cpu = x_hat_e_cpu.clamp(0, 1)
        x_hat_e_cpu = x_hat_e_cpu.view(x_hat_e_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_hat_e_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()

        x_cpu = x.cpu().data
        x_cpu = x_cpu.clamp(0, 1)
        x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()
    return vibi_f_frkl, optimizer_frkl


def retraining_train(vibi, vibi_retrain, vibi_f_frkl, dataloader_remain, dataloader_erase, reconstructor,
                     reconstruction_function,
                     loss_fn, optimizer_retrain, test_loader, train_loader):
    init_epoch = 0
    print("retraining")
    logs = defaultdict(list)
    valid_acc = 0.8
    acc_test = []
    poison_acc = []
    backdoor_acc_list = []

    if init_epoch == 0 or args.resume_training:

        print('Retraining VIBI')
        print(f'{args.explainer_type:>10} explainer params:\t{num_params(vibi_retrain.explainer) / 1000:.2f} K')
        print(
            f'{type(vibi_retrain.approximator).__name__:>10} approximator params:\t{num_params(vibi_retrain.approximator) / 1000:.2f} K')
        print(
            f'{type(vibi_retrain.forgetter).__name__:>10} forgetter params:\t{num_params(vibi_retrain.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        epoch_test_acc = []
        KL_fr = []
        KL_er = []
        KL_nipsr = []
        KL_kl = []
        mu_list_f = []
        sigma_list_f = []
        mu_list_e = []
        sigma_list_e = []
        mu_list_r = []
        sigma_list_r = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi_retrain.train()
            step_start = epoch * len(dataloader_remain)
            for step, (x, y) in enumerate(dataloader_remain, start=step_start):

                x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
                x = x.view(x.size(0), -1)
                logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = vibi_retrain(x,
                                                                               mode='forgetting')  # (B, C* h* w), (B, N, 10)

                logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
                # logits_z_e_kl, logits_y_e_kl, x_hat_e_kl, mu_e_kl, logvar_e_kl = vibi_f_kl(x, mode='forgetting')
                # logits_z_e_nips, logits_y_e_nips, x_hat_e_nips, mu_e_nips, logvar_e_nips = vibi_f_nips(x,
                #                                                                                        mode='forgetting')
                # print(x_hat_e)
                logits_z_f, logits_y_f, mu_f, logvar_f = vibi(x, mode='distribution')
                # logits_y_r = torch.softmax(logits_y_r, dim=1)
                logits_z_r_softmax = logits_z_r.log_softmax(dim=1)
                p_x_r = x.softmax(dim=1)

                KLD_element = mu_r.pow(2).add_(logvar_r.exp()).mul_(-1).add_(1).add_(logvar_r).cuda()
                KLD = torch.mean(KLD_element).mul_(-0.5).cuda()
                KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
                # x_hat_r = torch.sigmoid(reconstructor(logits_z_r))
                x_hat_r = x_hat_r.view(x_hat_r.size(0), -1)
                x = x.view(x.size(0), -1)
                x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
                x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)

                x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
                x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
                BCE = reconstruction_function(x_hat_r, x)  # mse loss
                H_p_q = loss_fn(logits_y_r, y)
                loss_r = args.beta * KLD_mean + H_p_q  # + BCE / (args.batch_size * 28 * 28)

                optimizer_retrain.zero_grad()
                loss_r.backward()
                torch.nn.utils.clip_grad_norm_(vibi_retrain.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer_retrain.step()

                KLD_fr = 0.5 * torch.mean(
                    logvar_r - logvar_f + (torch.exp(logvar_f) + (mu_f - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

                KLD_er = 0.5 * torch.mean(
                    logvar_r - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

                # KLD_klr = 0.5 * torch.mean(
                #     logvar_r - logvar_e_kl + (torch.exp(logvar_e_kl) + (mu_e_kl - mu_r).pow(2)) / torch.exp(
                #         logvar_r) - 1)
                # KLD_nips = 0.5 * torch.mean(
                #     logvar_r - logvar_e_nips + (torch.exp(logvar_e_nips) + (mu_e_nips - mu_r).pow(2)) / torch.exp(
                #         logvar_r) - 1)
                acc = (logits_y_r.argmax(dim=1) == y).float().mean().item()
                # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
                metrics = {
                    'acc': acc,
                    'loss': loss_r.item(),
                    'BCE': BCE.item(),
                    'H(p,q)': H_p_q.item(),
                    'mu_r': torch.mean(mu_r).item(),
                    'sigma_r': torch.sqrt_(torch.exp(logvar_r)).mean().item(),
                    'KLD_fr': KLD_fr.item(),
                    'KLD_er': KLD_er.item(),
                    'KLD_mean': KLD_mean.item(),
                }

                for m, v in metrics.items():
                    logs[m].append(v)
                if epoch == args.num_epochs - 1:
                    KL_fr.append(KLD_fr.item())
                    KL_er.append(KLD_er.item())
                    # KL_kl.append(KLD_klr.item())
                    # KL_nipsr.append(KLD_nips.item())
                if epoch == args.num_epochs - 1:
                    mu_list_r.append(torch.mean(mu_r).item())
                    sigma_list_r.append(torch.sqrt_(torch.exp(logvar_r)).mean().item())
                    mu_list_f.append(torch.mean(mu_f).item())
                    sigma_list_f.append(torch.sqrt_(torch.exp(logvar_f)).mean().item())
                    mu_list_e.append(torch.mean(mu_e).item())
                    sigma_list_e.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())

                if step % len(dataloader_remain) % 20000 == 0:
                    print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(dataloader_remain):3d}] '
                          + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

            vibi.eval()
            valid_acc_old = valid_acc
            valid_acc = test_accuracy(vibi_retrain, test_loader, args, name='vibi valid top1')
            interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            logs['val_acc'].extend(interpolate_valid_acc)
            print("test_acc", valid_acc)
            epoch_test_acc.append(valid_acc)

            print("epoch: ", epoch)
            # valid_acc_old = valid_acc
            valid_acc = test_accuracy(vibi_retrain, test_loader, args, name='vibi valid top1')
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            print("test_acc", valid_acc)
            acc_test.append(valid_acc)
            backdoor_acc = test_accuracy(vibi_retrain, dataloader_erase, args, name='vibi valid top1')
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            backdoor_acc_list.append(backdoor_acc)
            print("backdoor_acc", backdoor_acc_list)

        print('mu_r', np.mean(mu_list_r), 'sigma_r', np.mean(sigma_list_r))
        print('mu_e', np.mean(mu_list_e), 'sigma_e', np.mean(sigma_list_e))
        print('mu_f', np.mean(mu_list_f), 'sigma_f', np.mean(sigma_list_f))
        print("epoch_test_acc", epoch_test_acc)
        print("KL_fr", np.mean(KL_fr), "KL_er", np.mean(KL_er), "KL_nips", np.mean(KL_nipsr), "KL_kl", np.mean(KL_kl))
        # print(KL_er)
        x_hat_r_cpu = x_hat_r.cpu().data
        x_hat_r_cpu = x_hat_r_cpu.clamp(0, 1)
        x_hat_r_cpu = x_hat_r_cpu.view(x_hat_r_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_hat_r_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()

        x_hat_e_cpu = x_hat_e.cpu().data
        x_hat_e_cpu = x_hat_e_cpu.clamp(0, 1)
        x_hat_e_cpu = x_hat_e_cpu.view(x_hat_e_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_hat_e_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()

        x_cpu = x.cpu().data
        x_cpu = x_cpu.clamp(0, 1)
        x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()  # x_hat_r_cpu = x_hat_r.cpu().data

    return vibi_retrain


def unlearning_main_body(args):
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    device = args.device
    print("device", device)

    dataset = args.dataset

    if dataset == 'MNIST':
        transform = T.Compose([
            T.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trans_mnist = transforms.Compose([transforms.ToTensor(), ])
        train_set = MNIST('../../data/mnist', train=True, transform=trans_mnist, download=True)
        test_set = MNIST('../../data/mnist', train=False, transform=trans_mnist, download=True)
        train_set_no_aug = train_set
    elif dataset == 'CIFAR10':
        train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                     T.ToTensor(),
                                     ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
        test_transform = T.Compose([T.ToTensor(),
                                    ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
        train_set = CIFAR10('../../data/cifar', train=True, transform=train_transform, download=False)
        test_set = CIFAR10('../../data/cifar', train=False, transform=test_transform, download=False)
        train_set_no_aug = CIFAR10('../../data/cifar', train=True, transform=test_transform, download=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    shadow_ratio = 0.0
    full_ratio = 1 - shadow_ratio
    unlearning_ratio = args.erased_local_r

    length = len(train_set)
    shadow_size, full_size = int(shadow_ratio * length), int(full_ratio * length)
    remaining_size, erasing_size = int((1 - unlearning_ratio) * full_size), int(unlearning_ratio * full_size)
    print('remaining_size', remaining_size)
    remaining_size = full_size - erasing_size
    print('remaining_size', remaining_size, shadow_size, full_size, erasing_size)

    remaining_set, erasing_set = torch.utils.data.random_split(train_set, [remaining_size, erasing_size])

    print(len(remaining_set))
    print(len(remaining_set.dataset.data))

    remaining_set = My_subset(remaining_set.dataset, remaining_set.indices)
    erasing_set = My_subset(erasing_set.dataset, erasing_set.indices)

    # dataloader_shadow = DataLoader(shadow_set, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    poison_samples = int(length) * args.erased_local_r
    poison_data, poison_targets = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                                base_label=1,
                                                                trigger_label=2, poison_samples=poison_samples,
                                                                batch_size=args.local_bs, device=args.device)

    if args.dataset == 'MNIST':
        data_reshape = remaining_set.data.reshape(len(remaining_set.data), 1, 28, 28)
    elif args.dataset == 'CIFAR10':
        data_reshape = remaining_set.data.reshape(len(remaining_set.data), 3, 32, 32)

    print('train_set.data.shape', train_set.data.shape)
    print('poison_data.shape', poison_data.shape)

    data = torch.cat([poison_data.to(args.device), data_reshape.to(args.device)], dim=0)
    targets = torch.cat([poison_targets.to(args.device), remaining_set.targets.to(args.device)], dim=0)

    poison_trainset = Data.TensorDataset(data, targets)  # Data.TensorDataset(data, targets)
    pure_backdoored_set = Data.TensorDataset(poison_data, poison_targets)

    """in a backdoored medol, we need to unlearn the trigger, 
    so the remaining dataset is all the clean samples, and the erased dataset is the poisoned samples
    here we set the pure_backdoored as the erased dataset"""
    erasing_set = pure_backdoored_set

    dataloader_full = DataLoader(poison_trainset, batch_size=args.batch_size, shuffle=True)

    dataloader_remain = DataLoader(remaining_set, batch_size=args.batch_size, shuffle=True)
    dataloader_erase = DataLoader(erasing_set, batch_size=args.batch_size, shuffle=True)

    # for step, (x, y) in enumerate(dataloader_full):
    #     print(x)
    #     break
    #
    # for step, (x, y) in enumerate(dataloader_remain):
    #     print(x)
    #     break

    print('full size', len(poison_trainset), 'remain size', len(remaining_set.data), 'erased size', len(erasing_set))
    beta = args.beta

    explainer_type = args.explainer_type

    init_epoch = 0
    best_acc = 0
    logs = defaultdict(list)

    vibi, lr = init_vibi("MNIST")
    vibi.to(args.device)

    valid_acc = 0.8
    loss_fn = nn.CrossEntropyLoss()

    reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
    reconstructor = reconstructor.to(device)
    optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=lr)

    torch.cuda.manual_seed(42)

    reconstruction_function = nn.MSELoss(size_average=True)

    acc_test = []
    print("learning")
    if init_epoch == 0 or args.resume_training:

        print('Training VIBI')
        print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
        print(
            f'{type(vibi.approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')
        print(f'{type(vibi.forgetter).__name__:>10} forgetter params:\t{num_params(vibi.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        mu_list = []
        sigma_list = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi.train()
            step_start = epoch * len(dataloader_full)
            vibi, mu_list, sigma_list = learning_train(dataloader_full, vibi, loss_fn, reconstruction_function, args,
                                                       epoch, mu_list, sigma_list, train_loader)
            vibi.eval()
            valid_acc_old = valid_acc
            valid_acc = test_accuracy(vibi, test_loader, args, name='vibi valid top1')
            interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            logs['val_acc'].extend(interpolate_valid_acc)
            print("test_acc", valid_acc)
            backdoor_acc = test_accuracy(vibi, dataloader_erase, args, name='vibi valid top1')
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            print("backdoor_acc", backdoor_acc)
            acc_test.append(valid_acc)

        print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))
        #
        final_round_mse = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi.train()
            step_start = epoch * len(dataloader_erase)
            for step, (x, y) in enumerate(dataloader_erase, start=step_start):
                x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
                x = x.view(x.size(0), -1)
                logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

                x_hat = torch.sigmoid(reconstructor(logits_z))
                x_hat = x_hat.view(x_hat.size(0), -1)
                x = x.view(x.size(0), -1)
                # x = torch.sigmoid(torch.relu(x))
                BCE = reconstruction_function(x_hat, x)  # mse loss
                loss = BCE

                optimizer_recon.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer_recon.step()
                if epoch == args.num_epochs - 1:
                    final_round_mse.append(BCE.item())
                if step % len(train_loader) % 600 == 0:
                    print("loss", loss.item(), 'BCE', BCE.item())
        print("final_round mse", np.mean(final_round_mse))

        for step, (x, y) in enumerate(test_loader, start=step_start):
            x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)
            logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
            x_hat = torch.sigmoid(reconstructor(logits_z))
            x_hat = x_hat.view(x_hat.size(0), -1)
            x = x.view(x.size(0), -1)
            break

        x_hat_cpu = x_hat.cpu().data
        x_hat_cpu = x_hat_cpu.clamp(0, 1)
        x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()
        x_cpu = x.cpu().data
        x_cpu = x_cpu.clamp(0, 1)
        x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()



    vibi_f_frkl, optimizer_frkl = unlearning_frkl_train(copy.deepcopy(vibi).to(args.device), dataloader_erase, loss_fn,
                                                        reconstructor,
                                                        reconstruction_function, test_loader, train_loader)

    vibi_retrain, lr = init_vibi("MNIST")
    vibi_retrain.to(args.device)
    optimizer_retrain = torch.optim.Adam(vibi_retrain.parameters(), lr=lr)



    vibi_retrain = retraining_train(vibi, vibi_retrain, vibi_f_frkl, dataloader_remain, dataloader_erase, reconstructor,
                                    reconstruction_function,
                                    loss_fn, optimizer_retrain, test_loader, train_loader)

    vibi_w = vibi.state_dict()
    diff_grad = vibi.state_dict()
    for k in diff_grad.keys():
        diff_grad[k] = diff_grad[k] - diff_grad[k]

    retrain_net_w = vibi_retrain.state_dict()
    distance = 0
    for k in vibi_w.keys():
        diff_grad[k] = retrain_net_w[k] - vibi_w[k]
        distance += torch.norm(diff_grad[k].float(), p=2)
    print("original retrain-learning_distance", distance)

    vibi_f_frkl_w = vibi_f_frkl.state_dict()
    diff_grad = vibi_f_frkl.state_dict()
    for k in diff_grad.keys():
        diff_grad[k] = diff_grad[k] - diff_grad[k]
    retrain_net_w = vibi_retrain.state_dict()
    distance = 0
    for k in vibi_f_frkl_w.keys():
        diff_grad[k] = retrain_net_w[k] - vibi_f_frkl_w[k]
        distance += torch.norm(diff_grad[k].float(), p=2)
    print("retrain-rkfl_unlearning_distance", distance)

    print('Beta', beta)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.gpu = 0
    args.num_users = 10
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = True
    args.model = 'z_linear'
    args.local_bs = 100
    args.local_ep = 10
    args.batch_size = 100
    args.num_epochs = 10
    args.dataset = 'MNIST'
    args.xpl_channels = 1
    args.epochs = int(10)
    args.add_noise = False
    args.beta = 0.001
    args.lr = 0.001
    args.erased_size = 1500  # 120
    args.poison_portion = 0.0
    args.erased_portion = 0.4
    args.erased_local_r = 0.02
    ## in unlearning, we should make the unlearned model first be backdoored and then forget the trigger effect
    args.unlearn_learning_rate = 0.1
    args.self_sharing_rate = 0.1
    args.unl_conver_r = 2
    print('args.beta', args.beta, 'args.lr', args.lr)
    print('args.erased_portion', args.erased_portion, 'args.erased_local_r', args.erased_local_r)
    print('args.unlearn_learning_rate', args.unlearn_learning_rate, 'args.self_sharing_rate', args.self_sharing_rate)
    unlearning_main_body(args)
