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
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np


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


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'], default='ResNet_4x')
parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
parser.add_argument('--beta', type=float, default=0.01, help='beta in objective J = I(y,t) - beta * I(x,t).')
parser.add_argument('--unlearning_ratio', type=float, default=0.4)
parser.add_argument('--num_samples', type=int, default=4,
                    help='Number of samples used for estimating expectation over p(t|x).')
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--save_best', action='store_true', help='Save only the best models (measured in valid accuracy).')
parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
parser.add_argument('--jump_start', action='store_true', default=False)
args = parser.parse_args()

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

device = 'cuda'

print("device", device)
dataset = args.dataset

if dataset == 'MNIST':
    # transform = T.Compose([T.ToTensor(),
    #                        T.Normalize(0.1307, 0.3080)])
    transform = T.Compose([
        T.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = MNIST('../../data/mnist', train=True, transform=transform, download=True)
    test_set = MNIST('../../data/mnist', train=False, transform=transform, download=True)
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

batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

shadow_ratio = 0.0
full_ratio = 1 - shadow_ratio
unlearning_ratio = args.unlearning_ratio

length = len(train_set)
shadow_size, full_size = int(shadow_ratio * length), int(full_ratio * length)
remaining_size, erasing_size = int( (1 - unlearning_ratio) * full_size), int( unlearning_ratio * full_size)
print('remaining_size', remaining_size)
remaining_size = full_size - erasing_size
print('remaining_size', remaining_size,shadow_size, full_size , erasing_size )

shadow_set, full_set = torch.utils.data.random_split(train_set, [shadow_size, full_size])
remaining_set, erasing_set = torch.utils.data.random_split(full_set, [remaining_size, erasing_size])

dataloader_full = DataLoader(full_set, batch_size=batch_size, shuffle=True)
dataloader_remain = DataLoader(remaining_set, batch_size=batch_size, shuffle=True)
dataloader_erase = DataLoader(erasing_set, batch_size=batch_size, shuffle=True)
# dataloader_shadow = DataLoader(shadow_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)


# train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)

# train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)

@torch.no_grad()
def test_accuracy(model, data_loader, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        out = model(x, mode='test')
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
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
        lr = 0.001

    elif dataset == 'CIFAR10':
        approximator = resnet18(3, 10)

        if explainer_type == 'ResNet_8x':
            block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        if explainer_type == 'ResNet_4x':
            block_features = [64] * 3 + [128] * 4 + [256] * 5
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        elif explainer_type == 'ResNet_2x':
            block_features = [64] * 4 + [128] * 5
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        elif explainer_type == 'Unet':
            explainer = Unet(3, [64, 128, 256, 512], xpl_channels)
        else:
            raise ValueError

        lr = 0.005
        temp_warmup = 4000

    model_ckpt = f'models/{dataset}_{explainer_type}_{xpl_channels}_k={k}_b={beta}.pt'
    results_loc = f'results/{dataset}_{explainer_type}_{xpl_channels}_k={k}_b={beta}'

    if args.jump_start and not os.path.exists(model_ckpt):
        load_ckpt = f'models/{dataset}_{explainer_type}_{xpl_channels}_k={k}_b=0.0.pt'

    os.makedirs(results_loc, exist_ok=True)

    vibi = VIBI(explainer, approximator, forgetter, k=k, num_samples=args.num_samples)
    vibi.to(device)
    return vibi, lr


def learning_train(dataset, model, step_start, loss_fn, reconstruction_function, optimizer):
    for step, (x, y) in enumerate(dataset, start=step_start):
        x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        logits_z, logits_y, x_hat, mu, logvar = model(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
        H_p_q = loss_fn(logits_y, y)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x)  # mse loss
        loss = beta * KLD_mean + H_p_q  + BCE / (batch_size * 28 * 28)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
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
            print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
    return  model, optimizer


beta = args.beta
num_samples = args.num_samples
xpl_channels = args.xpl_channels
explainer_type = args.explainer_type

init_epoch = 0
best_acc = 0
logs = defaultdict(list)

vibi, lr = init_vibi("MNIST")
vibi.to(device)

vibi_retrain, lr = init_vibi("MNIST")
vibi_retrain.to(device)

optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)
optimizer_retrain = torch.optim.Adam(vibi_retrain.parameters(), lr=lr)

valid_acc = 0.8
loss_fn = nn.CrossEntropyLoss()

reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
reconstructor = reconstructor.to(device)
optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=lr)

torch.cuda.manual_seed(42)

reconstruction_function = nn.MSELoss(size_average=False)

print("learning")
if init_epoch == 0 or args.resume_training:

    print('Training VIBI')
    print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
    print(f'{type(vibi.approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')
    print(f'{type(vibi.forgetter).__name__:>10} forgetter params:\t{num_params(vibi.forgetter) / 1000:.2f} K')
    # inspect_explanations()
    mu_list = []
    sigma_list = []
    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        vibi.train()
        step_start = epoch * len(dataloader_full)
        vibi, optimizer = learning_train(dataloader_full, vibi, step_start, loss_fn, reconstruction_function, optimizer)
        vibi.eval()
        valid_acc_old = valid_acc
        valid_acc = test_accuracy(vibi, test_loader, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
        print("test_acc", valid_acc)

    print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))

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
            loss = BCE / (batch_size * 28 * 28) * 10

            optimizer_recon.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_recon.step()
            if epoch == args.num_epochs-1:
                final_round_mse.append( BCE.item())
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

vibi_f_temp, lr = init_vibi("MNIST")
vibi_f_temp.to(device)
vibi_f_temp.load_state_dict(vibi.state_dict())


init_epoch = 0
print("unlearning")

reconstructor_for_unlearning = LinearModel(n_feature=49, n_output=28 * 28)
reconstructor_for_unlearning = reconstructor_for_unlearning.to(device)
optimizer_recon_for_un = torch.optim.Adam(reconstructor_for_unlearning.parameters(), lr=lr)

if init_epoch == 0 or args.resume_training:

    print('Unlearning VIBI KLD')
    print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
    print(f'{type(vibi.approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')
    print(f'{type(vibi.forgetter).__name__:>10} forgetter params:\t{num_params(vibi.forgetter) / 1000:.2f} K')
    # inspect_explanations()
    epoch_test_acc = []
    mu_list = []
    sigma_list = []
    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        vibi.train()
        step_start = epoch * len(dataloader_erase)
        for step, (x, y) in enumerate(dataloader_erase, start=step_start):

            x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi(x, mode='forgetting')
            logits_z_f, logits_y_f, x_hat_f, mu_f, logvar_f = vibi_f_temp(x, mode='forgetting')
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
            e_log_p = torch.exp(BCE / (batch_size * 28 * 28))  # = 1/p
            e_log_py = torch.exp(H_p_q)
            log_z = torch.mean(logits_z_e.log_softmax(dim=1))
            loss = beta * KLD_mean + H_p_q - log_z / e_log_py

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
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
            if epoch == args.num_epochs - 1:
                mu_list.append(torch.mean(mu_e).item())
                sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            if step % len(dataloader_erase) % 600 == 0:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

        vibi.eval()
        valid_acc_old = valid_acc
        valid_acc = test_accuracy(vibi, test_loader, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
        print("test_acc", valid_acc)
        epoch_test_acc.append(valid_acc)

    print("epoch_test_acc", epoch_test_acc)
    x_hat_e_cpu = x_hat_e.cpu().data
    x_hat_e_cpu = x_hat_e_cpu.clamp(0, 1)
    x_hat_e_cpu = x_hat_e_cpu.view(x_hat_e_cpu.size(0), 1, 28, 28)
    grid = torchvision.utils.make_grid(x_hat_e_cpu, nrow=4, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    final_round_mse = []
    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        vibi.train()
        step_start = epoch * len(dataloader_erase)
        for step, (x, y) in enumerate(dataloader_erase, start=step_start):
            x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)
            logits_z, logits_y, x_hat_e, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

            x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            BCE = reconstruction_function(x_hat_e, x)  # mse loss
            loss = BCE / (batch_size * 28 * 28) * 10

            optimizer_recon_for_un.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_recon_for_un.step()
            if epoch == args.num_epochs -1 :
                final_round_mse.append(BCE.item())
            if step % len(train_loader) % 600 == 0:
                print("loss", loss.item(), 'BCE', BCE.item())
    print("final epoch mse", np.mean(final_round_mse))

    for step, (x, y) in enumerate(test_loader, start=step_start):
        x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        logits_z, logits_y, x_hat_e, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
        x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
        x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
        x = x.view(x.size(0), -1)
        break

    print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))
    print("epoch_test_acc", epoch_test_acc)
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


init_epoch = 0
print("retraining")
if init_epoch == 0 or args.resume_training:

    print('Retraining VIBI')
    print(f'{explainer_type:>10} explainer params:\t{num_params(vibi_retrain.explainer) / 1000:.2f} K')
    print(
        f'{type(vibi_retrain.approximator).__name__:>10} approximator params:\t{num_params(vibi_retrain.approximator) / 1000:.2f} K')
    print(
        f'{type(vibi_retrain.forgetter).__name__:>10} forgetter params:\t{num_params(vibi_retrain.forgetter) / 1000:.2f} K')
    # inspect_explanations()
    epoch_test_acc = []
    KL_fr = []
    KL_er = []
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

            x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)
            logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = vibi_retrain(x,
                                                                           mode='forgetting')  # (B, C* h* w), (B, N, 10)

            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi(x, mode='forgetting')
            # print(x_hat_e)
            logits_z_f, logits_y_f, mu_f, logvar_f = vibi_f_temp(x, mode='distribution')
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
            loss_r = beta * KLD_mean + H_p_q + BCE / (batch_size * 28 * 28)

            optimizer_retrain.zero_grad()
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(vibi_retrain.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_retrain.step()

            KLD_fr = 0.5 * torch.mean(
                logvar_r - logvar_f + (torch.exp(logvar_f) + (mu_f - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

            KLD_er = 0.5 * torch.mean(
                logvar_r - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

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
        valid_acc = test_accuracy(vibi_retrain, test_loader, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
        print("test_acc", valid_acc)
        epoch_test_acc.append(valid_acc)

    print('mu_r', np.mean(mu_list_r), 'sigma_r', np.mean(sigma_list_r))
    print('mu_e', np.mean(mu_list_e), 'sigma_e', np.mean(sigma_list_e))
    print('mu_f', np.mean(mu_list_f), 'sigma_f', np.mean(sigma_list_f))
    print("epoch_test_acc", epoch_test_acc)
    print("KL_fr", np.mean(KL_fr), "KL_er", np.mean(KL_er))
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

print('Beta', beta)
