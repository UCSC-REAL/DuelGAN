import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks


class Generator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 z_dim=128,
                 nc=3,
                 ngf=64,
                 embed_dim=256,
                 **kwargs):
        super(Generator, self).__init__()

        self.fc = nn.Linear(z_dim, 4 * 4 * ngf * 8)
        

        self.conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ngf * 4)

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf * 2)

        self.conv3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf)

        self.conv_out = nn.Sequential(nn.Conv2d(ngf, nc, 3, 1, 1), nn.Tanh())

    def forward(self, input):
        out = self.fc(input)
        out = out.view(out.size(0), -1, 4, 4)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return self.conv_out(out)


class Discriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 nc=3,
                 ndf=64,
                 pack_size=1,
                 features='penultimate',
                 **kwargs):

        super(Discriminator, self).__init__()


        self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size, ndf, 3, 1, 1), nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf, 4, 2, 1), nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 1), nn.LeakyReLU(0.1))
        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1), nn.LeakyReLU(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1), nn.LeakyReLU(0.1))
        self.conv6 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1), nn.LeakyReLU(0.1))
        self.conv7 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1), nn.LeakyReLU(0.1))

        self.fc_out = nn.Linear(ndf * 8 * 4 * 4, 1)


    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        out = out.view(out.size(0), -1)
        result = self.fc_out(out)
        result = result.view(result.size(0))
        # result = self.softplus(result)
        assert (len(result.shape) == 1)
        return result


if __name__ == '__main__':
    z = torch.zeros((1, 128))
    g = Generator()
    x = torch.zeros((1, 3, 32, 32))
    d = Discriminator()

    g(z)
    d(g(z))
    d(x)