import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import numpy as np
import os
import argparse
import math
torch.set_num_threads(1)
from read_data import ImageList


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='rc', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='t', metavar='S',
                    help='target dataset')
parser.add_argument('--tradeoff_pam', type=float, default=0,
                        help='tradeoff of pam')
parser.add_argument('--lr', type=float, default=0.1,
                        help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001,
                        help='learning rate decay')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
parser.add_argument('--sigma_k', type=int, default=36,
                        help='top k align')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_transforms = {
    'train': tran.rr_train(),
    'val': tran.rr_train(),
    'test': tran.rr_eval(),
}
# set dataset
batch_size = {"train": 36, "val": 36, "test": 4}
rc="rr-realistic.txt"
rl="rr-real.txt"



if args.src =='rl':
    source_path = rl
elif args.src =='rc':
    source_path = rc

if args.tgt =='rl':
    target_path = rl
elif args.tgt =='rc':
    target_path = rc


dsets = {"train": ImageList(open(source_path).readlines(), transform=data_transforms["train"]),
         "val": ImageList(open(target_path).readlines(),transform=data_transforms["val"]),
         "test": ImageList(open(target_path).readlines(),transform=data_transforms["test"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=0)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=64)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val','test']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Regression_test(loader, model):
    MSE=0
    MAE=0
    number = 0
    with torch.no_grad():
        for (imgs, labels) in loader['test']:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            labels = torch.cat((labels1, labels2), dim=1)
            labels = labels.float() / 39
            pred = model(imgs)
            MSE += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE += torch.nn.L1Loss(reduction='sum')(pred,labels)
            number += imgs.size(0)

    print("\tMSE : {0}\n".format(MSE/number))
    print("\tMAE : {0}\n".format(MAE/number))



def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

class net_R(nn.Module):
    def __init__(self):
        super(net_R,self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
    def forward(self,x):
        feature = self.model_fc(x)
        outC= self.classifier_layer(feature)
        return(outC,feature)


def RSD(Feature_s, Feature_t,k=36):
    u_s, s_s, v_s = torch.svd(Feature_s)
    u_t, s_t, v_t = torch.svd(Feature_t)
    _, cospa, _ = torch.svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return -torch.norm(sinpa[:k],1)



net = net_R()
net = net.to(device)

net.train(True)
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, net.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, net.classifier_layer.parameters()), "lr": 1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
train_cross_loss = train_transfer_loss = train_total_loss = 0.0
len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])
for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
test_interval = 500
num_iter = 20002
for iter_num in range(1, num_iter + 1):
    net.train(True)
    optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                 weight_decay=0.0005)
    optimizer.zero_grad()
    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["train"])
    if iter_num % len_target == 0:
        iter_target = iter(dset_loaders["val"])
    data_source = iter_source.next()
    data_target = iter_target.next()
    inputs_source, labels_source = data_source
    labels1 = labels_source[:,0]
    labels2 = labels_source[:,1]
    labels1 = labels1.unsqueeze(1)
    labels2 = labels2.unsqueeze(1)
    labels_source = torch.cat((labels1,labels2),dim=1)
    labels_source = labels_source.float()/39
    inputs_target, labels_target = data_target
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    inputs = inputs.to(device)
    labels = labels_source.to(device)
    dc_target = dc_target.to(device)
    inputs_s = inputs.narrow(0, 0, batch_size["train"])
    inputs_t = inputs.narrow(0, batch_size["train"], batch_size["train"])
    outC_s, feature_s = net(inputs_s)
    outC_t, feature_t = net(inputs_t)

    pamloss = PAM(feature_s, feature_t, k=args.sigma_k)
    disloss = args.tradeoff_pam * pamloss

    classifier_loss = nn.MSELoss()(outC_s, labels)
    total_loss = classifier_loss - disloss
    total_loss.backward()
    optimizer.step()
    train_cross_loss += classifier_loss.item()
    train_total_loss += total_loss.item()
    if iter_num % test_interval == 0:
        print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average Transfer Loss: {:.4f};  Average Training Loss: {:.4f}".format(
            iter_num, train_cross_loss / float(test_interval), train_transfer_loss / float(test_interval),
            train_total_loss / float(test_interval)))
        train_cross_loss = train_transfer_loss = train_total_loss =train_sigma = 0.0
    if iter_num % test_interval == 0:
        net.eval()
        Regression_test(dset_loaders, net.predict_layer)
