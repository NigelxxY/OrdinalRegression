import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Means(nn.Module):
    def __init__(self):
        super(Means, self).__init__()

    def forward(self, input):
        return torch.mean(input, dim=(1, 2, 3)).unsqueeze(-1)


class ZeroOuts(nn.Module):
    def __init__(self):
        super(ZeroOuts, self).__init__()

    def forward(self, x):
        batchSize = x.size()[0]
        return torch.zeros(batchSize, 4, 1, 1).cuda()


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 relu=True, bn=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.leaky_relu(x, inplace=True)
        return x


class NormalizerLoss(nn.Module):
    def __init__(self, std):
        super(NormalizerLoss, self).__init__()
        self.std = torch.from_numpy(
            np.array(std).astype('float32')
        ).cuda()
        self.lambda_ = 1e-2

    def forward(self, x, y, rainMask, regressionMask):
        se = ((x - y) / self.std) ** 2
        se = torch.matmul(regressionMask.unsqueeze(-2), se.unsqueeze(-1)).squeeze(-1)
        zeros = torch.zeros(se.size()).to(se.device)
        se = torch.cat([zeros, se], dim=1).unsqueeze(-1)
        rainMask = rainMask.unsqueeze(-2)
        loss = torch.matmul(rainMask, se)
        # ae = torch.abs((x - y) / self.std)
        # ae = torch.matmul(softMax, ae.unsqueeze(-1))
        # loss = (se.sum()) / (y > 0.05).sum()
        #        # - self.lambda_ * (
        #     # (ae.sum() / (y > 0.05).sum()) ** 2.0
        # )
        return loss.mean()


class MeanVarLoss(nn.Module):
    def __init__(self, nClass):
        super(MeanVarLoss, self).__init__()
        self.labels = torch.arange(nClass).view(nClass, 1).float().cuda()

    def forward(self, x, y):
        e = torch.matmul(x.unsqueeze(-2), self.labels)
        varLoss = torch.matmul(x.unsqueeze(-2), (self.labels - e) ** 2).squeeze(-1)
        meanLoss = (e.squeeze(-1) - y) ** 2
        return torch.mean(meanLoss), torch.mean(varLoss)


class ordinalLoss(nn.Module):
    def __init__(self, space):
        super(ordinalLoss, self).__init__()
        self.epilson = 1e-10
        self.gamma = 2
        self.alpha = 0.25
        self.space = space

    def forward(self, x, y):
        batchSize = x.size(0)
        gamma = self.gamma
        alpha = self.alpha
        # ordinal ce loss
        ce = - (torch.sum(
            torch.pow((1 - x), gamma) * torch.log(x + self.epilson).mul(y) +
            torch.pow(x, gamma) * torch.log(1 - x + self.epilson).mul(1 - y))
        ) / batchSize
        # mse loss
        # mse = torch.mean((0.5 * (torch.sum(x, dim=1) - torch.sum(y, dim=1))) ** 2)
        # EMD Loss
        x_soft = F.softmax(x, dim=1)
        y_soft = F.softmax(y, dim=1)
        x_cumsum = torch.cumsum(x_soft, dim=1)
        y_cumsum = torch.cumsum(y_soft, dim=1)
        EMD_loss = torch.sum((x_cumsum - y_cumsum) ** 2)
        return ce, EMD_loss


class FocalLoss(nn.Module):
    def __init__(self, nClass, alpha=0.75, gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss, self).__init__()
        self.nClass = nClass
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.nClass, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.nClass
            self.alpha = torch.FloatTensor(alpha).unsqueeze(-1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.nClass, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise NotImplementedError

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
        target = target.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha

        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        logit = F.softmax(input, dim=-1)
        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.nClass).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]

        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


class MeanVarianceNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(MeanVarianceNormalizer, self).__init__()
        self.mean = torch.from_numpy(
            np.array(mean).astype('float32')
        ).cuda()
        self.std = torch.from_numpy(
            np.array(std).astype('float32')
        ).cuda()

    def forward(self, x):
        x = x.squeeze(-1) * self.std + self.mean
        return x


class regressionClassification(nn.Module):
    def __init__(self, nClass):
        super(regressionClassification, self).__init__()
        self.nClass = nClass

        self.conv = nn.Sequential(
            BasicConv2d(37, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 128, 3, bn=True, padding=1),
            BasicConv2d(128, 128, 3, bn=True, padding=1),
        )

        self.downsample = nn.Sequential(
            BasicConv2d(37, 128, 1, bn=True, relu=False, padding=0)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(128 * 1 * 1, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, self.nClass),
        )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv(x)
        x = x + residual
        x = self.pool(x)
        x = self.fc(x)
        return x


class rainFallClassification(nn.Module):
    def __init__(self):
        super(rainFallClassification, self).__init__()

        self.conv = nn.Sequential(
            BasicConv2d(37, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 128, 3, bn=True, padding=1),
            BasicConv2d(128, 128, 3, bn=True, padding=1),
        )

        self.downsample = nn.Sequential(
            BasicConv2d(37, 128, 1, bn=True, relu=False, padding=0)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(128 * 1 * 1, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv(x)
        x = x + residual
        x = self.pool(x)
        x = self.fc(x)
        return x


class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(64, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 32, 1),
            BasicConv2d(32, 32, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.downsampler = nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0),
            nn.BatchNorm2d(32)
        )

        self.ac = nn.ReLU(True)

        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = self.downsampler(x)
        x = self.conv(x)
        x = self.ac(x + identity)
        x = self.pooling(x)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()

        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(37*17*17,1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024,128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128,1)
        )

    def forward(self,x):
        x = self.fc(x)
        return x


class SimpleRegressionModel(nn.Module):
    def __init__(self, nClass):
        super(SimpleRegressionModel, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(64, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 32, 1),
            BasicConv2d(32, 32, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))


        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32,1)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.fc(x)
        return x


class OrdinalRegressionModel(nn.Module):
    def __init__(self, nClass):
        super(OrdinalRegressionModel, self).__init__()
        self.nClass = nClass

        self.boosting = nn.ModuleList()

        for i in range(self.nClass):
            oneClassifier = BaseClassifier()
            self.boosting.append(oneClassifier)

    def forward(self, x):
        outputs = [self.boosting[i](x) for i in range(self.nClass)]
        return torch.cat(outputs, dim=1)


class Regression(nn.Module):
    def __init__(self, nClass):
        super(Regression, self).__init__()
        self.nClass = nClass

        self.subreg_l1 = nn.Sequential(
            # BasicConv2d(64, 64, 3, padding=1),
            # BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(64, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(32 * 4 * 4, 1, bias=False),
        )

        self.subreg_l2 = nn.Sequential(
            # nn.MaxPool2d(2),
            BasicConv2d(64, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(16 * 4 * 4, 1, bias=False),
        )

        self.share_conv_l3_l4 = nn.Sequential(
            BasicConv2d(64, 64, 3, bn=True, padding=1),
        )

        self.subreg_l3 = nn.Sequential(
            BasicConv2d(64, 32, 1),
            BasicConv2d(32, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(16 * 4 * 4, 1, bias=False),
        )

        self.subreg_l4 = nn.Sequential(
            BasicConv2d(64, 32, 1),
            BasicConv2d(32, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(16 * 4 * 4, 1, bias=False),
        )

        self.subreg_l5 = nn.Sequential(
            # nn.MaxPool2d(2),
            BasicConv2d(64, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(16 * 4 * 4, 1, bias=False),
        )

        # self.zeroOut = ZeroOuts()

        # self.multiFc = nn.ModuleList()

        self.multiFc = nn.ModuleList([
            self.subreg_l1,
            self.subreg_l2,
            self.subreg_l5,
            nn.Sequential(
                self.share_conv_l3_l4,
                self.subreg_l3,
            ),
            nn.Sequential(
                self.share_conv_l3_l4,
                self.subreg_l4,
            )
        ])
        # for i in range(nClass):
        #     oneFc = BaseRegression()
        #     self.multiFc.append(oneFc)
        # self.multiFc.append(self.zeroOut)

    def forward(self, x):
        batchSize = x.size()[0]
        # x = self.conv(x)
        # print('aft_conv:', x[:5, -1, :, :])
        outputs = [torch.mean(self.multiFc[i](x), dim=(1, 2, 3)).unsqueeze(-1) for i in range(len(self.multiFc))]
        regressionValues = torch.cat(outputs, dim=1).view(batchSize, len(self.multiFc), 1)
        return regressionValues


class AutoencoderBN(nn.Module):
    def __init__(self, noiseMean, noiseStd):
        super(AutoencoderBN, self).__init__()

        self.noiseMean = noiseMean
        self.noiseStd = noiseStd

        self.encoder = nn.Sequential(
            BasicConv2d(37, 32, 1, bn=True),
            BasicConv2d(32, 32, 3, bn=True, padding=1),
        )
        self.encoderAfterNoise = nn.Sequential(
            nn.MaxPool2d(2),
            # -------------------------------------
            BasicConv2d(32, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 64, 3, bn=True, padding=1),
            # nn.MaxPool2d(2),
            # -------------------------------------
            # BasicConv2d(64, 128, 3, padding=1),
            # BasicConv2d(128, 128, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            BasicConv2d(64, 32, 3, bn=True, padding=1),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(size=(17, 17), mode='bilinear'),
            # BasicConv2d(128, 64, 5, padding=2),
            # nn.Upsample(size=(17, 17), mode='bilinear'),
            BasicConv2d(32, 32, 3, bn=True, padding=1),
            BasicConv2d(32, 37, 1, bn=True),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        if self.training:
            encoder = encoder + torch.normal(mean=torch.zeros(encoder.size()), std=self.noiseStd).to(encoder.device)
        encoder = self.encoderAfterNoise(encoder)
        decoder = self.decoder(encoder)
        return encoder, decoder


class NonLocalSptialBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocalSptialBlock2D, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NonLocalChannelBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocalChannelBlock2D, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                             kernel_size=3, stride=1, padding=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=3, stride=1, padding=1)

        if sub_sample:
            self.theta = nn.Sequential(max_pool_layer, self.theta)
            self.phi = nn.Sequential(max_pool_layer, self.phi)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        # g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        theta_x = self.theta(x).view(batch_size, self.in_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, x.view(batch_size, self.in_channels, -1))
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NonLocalRegression(nn.Module):
    def __init__(self, nClass):
        super(NonLocalRegression, self).__init__()

        self.nClass = nClass

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=37, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # NonLocalSptialBlock2D(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # NonLocalSptialBlock2D(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classificationFc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.convs(x).view(batch_size, -1)
        # outputs = [self.multiFc[i](output) for i in range(len(self.multiFc))]
        return self.classificationFc(output)

        # preds = self.classificationFc(output)
        # return torch.cat(outputs,dim=1).view(-1, len(self.multiFc), 1), preds
        #
        # def getResidual(self, x, height, width):
        #     crop_h = torch.FloatTensor([x.size()[1]]).sub(height).div(-2)
        #     crop_w = torch.FloatTensor([x.size()[2]]).sub(width).div(-2)
        #
        #     return torch.mean(F.pad(x, [crop_w.ceil().int()[0], crop_w.floor().int()[0],
        #                                 crop_h.ceil().int()[0], crop_h.floor().int()[0]]))
