import torch
import numpy as np
import torch.nn as nn
from thop import profile
from .real_nvp import RealNVP
from easydict import EasyDict
from einops import rearrange
import torch.distributions as distributions

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.LeakyReLU(0.2, True),
        nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False)
    )

# loc-flow
def nets():
    return nn.Sequential(nn.Linear(2, 256),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256),
                         nn.LeakyReLU(),
                         nn.Linear(256, 2),
                         nn.Tanh())

def nett():
    return nn.Sequential(nn.Linear(2, 256),
                         nn.LeakyReLU(),
                         nn.Linear(256, 256),
                         nn.LeakyReLU(),
                         nn.Linear(256, 2))

class TokenToCoordinate(nn.Module):
    def __init__(self, input_channels, L):
        super(TokenToCoordinate, self).__init__()
        self.conv = nn.Conv1d(input_channels, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(L, 2)

    def forward(self, x):
        b = x.shape[0]
        x = x.permute(0, 2, 1)  # B, C, L
        x = self.conv(x)  # B, 2, L
        # x = x.view(b, -1)  # B, L, 2
        x = self.linear(x)  # B, 2
        return x

class Loc_Flow(nn.Module):
    def __init__(self, inputs=7, num_joints=1):
        super(Loc_Flow, self).__init__()

        self.num_joints = num_joints
        self.inputs = inputs

        self.layer00 = convrelu(inputs, 20, 3, 1, 1)  # 256
        self.layer0 = convrelu(20, 50, 5, 2, 2)  # 128
        self.layer1 = convrelu(50, 60, 5, 2, 2)  # 64
        self.layer10 = convrelu(60, 70, 5, 2, 1)  # 64
        self.layer11 = convrelu(70, 90, 5, 2, 2)  # 32
        self.layer110 = convrelu(90, 100, 5, 2, 1)  # 32
        self.layer2 = convrelu(100, 120, 5, 2, 2)  # 16
        self.layer20 = convrelu(120, 120, 3, 1, 1)  # 16
        self.layer3 = convrelu(120, 135, 5, 2, 1)  # 16
        self.layer31 = convrelu(135, 150, 5, 2, 2)  # 8
        self.layer4 = convrelu(150, 225, 5, 2, 1)  # 8
        self.layer41 = convrelu(225, 300, 5, 2, 2)  # 4
        self.layer5 = convrelu(300, 400, 5, 2, 1)  # 4
        self.layer51 = convrelu(400, 500, 5, 2, 2)  # 2

        self.token_to_coor = TokenToCoordinate(500, 4)

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        self.flow = RealNVP(nets, nett, masks, prior)

    def forward(self, x, label=None):

        BATCH_SIZE = x.shape[0]

        layer00 = self.layer00(x)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer11 = self.layer11(layer10)
        layer110 = self.layer110(layer11)
        layer2 = self.layer2(layer110)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer31 = self.layer31(layer3)
        layer4 = self.layer4(layer31)
        layer41 = self.layer41(layer4)
        layer5 = self.layer5(layer41)
        layer51 = self.layer51(layer5)


        x = rearrange(layer51, 'B C H W -> B (H W) C')

        output = self.token_to_coor(x)

        coordinate = output

        out_coord = coordinate.reshape(BATCH_SIZE, self.num_joints, 2)

        assert out_coord.shape[2] == 2

        out_sigma = coordinate.reshape(BATCH_SIZE, self.num_joints, -1)

        # (B, N, 2)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid()
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        if label is not None:
            gt_uv = label.reshape(pred_jts.shape)
            bar_mu = (pred_jts - gt_uv) / sigma
            # (B, K, 2)
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)
            nf_loss = torch.log(sigma) - log_phi

        else:
            nf_loss = None

        outputs = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
            nf_loss=nf_loss
        )

        return outputs

def test():
    x = torch.randn((1, 7, 256, 256)).cuda()

    print('==> Building model..')

    model = Loc_Flow()
    model.cuda()

    flops, params = profile(model, (x, ))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

    preds = model(x)
    print(preds)

if __name__ == "__main__":
    test()


