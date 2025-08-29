from __future__ import print_function, division
import os
import cv2
import time
import math
import copy
import torch
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# ignore warning
warnings.filterwarnings("ignore")

device = torch.device("cuda")

from lib import loader, modules

class RLELoss(nn.Module):
    '''
    RLE Regression Loss
    '''

    def __init__(self, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-4)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels.reshape(pred_jts.shape)
        gt_uv_weight = labels.reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:

            return loss.sum() / len(loss)

        else:
            return loss.sum()

if __name__ == "__main__":
    try:
        os.mkdir('results')
    except OSError as error:
        print("结果文件已存在")

    simName = "DPM"  # Options: DPM, ZSDPMtoIRT2, DPMtoIRT2, DPMcars, IRT2carsCDPM, IRT2carsCDPMtoIRT,

    with open('results/' + simName + 'Log.txt', 'w') as f:
        print('Training Accuracy', file=f)
        print('-' * 20, file=f)

    batch_size = 16
    Loc_train = loader.locDL(phase="train")
    Loc_val = loader.locDL(phase="val")

    dataloaders = {
        'train': DataLoader(Loc_train, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True),
        'val': DataLoader(Loc_val, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    }

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = modules.Loc_Flow()

    model.cuda()

    def my_loss(output, target):
        loss = torch.sum((output - target) ** 2, 1)
        loss = torch.sqrt(loss)
        loss = torch.mean(loss)
        return loss

    def calc_loss_dense(pred, target, metrics):
        loss = my_loss(pred, target)  # *256*256
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        return loss

    def los(pred, target, metrics):
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        return loss

    def loses(pred, target, metrics):
        criterion = nn.L1Loss()
        loss = criterion(pred, target)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        return loss

    def los1(losses, target, metrics):
        loss = losses
        metrics['loss'] += losses.data.cpu().numpy() * target.size(0)
        return loss


    def print_metrics(metrics, epoch_samples, phase):
        outputs1 = []
        for k in metrics.keys():
            outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        print("{}: {}".format(phase, ", ".join(outputs1)))

    def train_model(model, optimizer, scheduler, num_epochs=30):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()

            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        with open('results/Log.txt', 'a') as f:
                            print("learning rate", param_group['lr'], file=f)

                    model.train()
                else:
                    model.eval()

                metrics = defaultdict(float)
                epoch_samples = 0

                for inputs, target_loc in tqdm(dataloaders[phase], desc='train'):

                    inputs = inputs.to(device)
                    targets = target_loc.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):

                        # outputs1 = model(inputs)
                        outputs1 = model(inputs, targets)

                        LF = RLELoss()
                        loss = LF(outputs1, targets)
                        loss = los1(loss, targets, metrics)
                        # print(loss)

                        # loss = los(outputs1, targets, metrics)
                        # print(loss)

                        if phase == 'train':

                            loss.backward()
                            optimizer.step()

                    epoch_samples += targets.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        model.load_state_dict(best_model_wts)
        return model

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=28, gamma=0.1)
    model = train_model(model, optimizer_ft, exp_lr_scheduler)

    stringer = 'results/' + simName + 'BestModel.pt'
    torch.save(model.state_dict(), stringer)