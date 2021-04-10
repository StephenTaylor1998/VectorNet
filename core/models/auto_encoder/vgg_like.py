import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, in_channel=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):

    def __init__(self, out_channel=3):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.ConvTranspose2d(16, out_channel, 3, stride=2, padding=1, output_padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x


class VGG_Encoder(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, **kwargs):
        super(VGG_Encoder, self).__init__()
        self.encoder = Encoder(in_channel)
        self.decoder = Decoder(out_channel)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class NetG(nn.Module):
    def __init__(self, nc=3, nef=64, ngf=64, nBottleneck=4000, ngpu=1):
        super(NetG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 64 x 64
            nn.Conv2d(nef, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 32 x 32
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 16 x 16
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*4) x 8 x 8
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*8) x 4 x 4
            nn.Conv2d(nef * 8, nBottleneck, 4, bias=False),
            # tate size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(nBottleneck, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class NetG32(nn.Module):
    def __init__(self, nc=3, nef=64, ngf=64, nBottleneck=4000, ngpu=1, **kwargs):
        super(NetG32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 32 => 16
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            # 16 => 8
            nn.Conv2d(nef, nef*2, 4, 2, 1, bias=False), nn.BatchNorm2d(nef*2), nn.LeakyReLU(0.2, inplace=True),
            # 8 => 4
            nn.Conv2d(nef*2, nef * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nef * 4), nn.LeakyReLU(0.2, inplace=True),
            # 4 => 1
            nn.Conv2d(nef * 4, nBottleneck, 4, bias=False), nn.BatchNorm2d(nBottleneck), nn.LeakyReLU(0.2, inplace=True),
            # 1 => 4
            nn.ConvTranspose2d(nBottleneck, ngf * 4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # 4 => 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # 8 => 16
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 1), nn.ReLU(True),
            # 16 => 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()
            # output == (nc) x 32 x 32
        )

    def forward(self, inputs):
        output = self.main(inputs)
        return output


# def get_vgg_encoder(**kwargs):
#     return VGG_Encoder(**kwargs)

def get_vgg_encoder(**kwargs):
    return NetG32(**kwargs)


if __name__ == '__main__':
    input_tensor = torch.randn((4, 3, 32, 32), dtype=torch.float32)
    # model = get_vgg_encoder()
    model = NetG32()
    print(model(input_tensor).shape)

    # encoder = Encoder()
    # out = encoder(input_tensor)
    # print(out.shape)
    #
    # decoder = Decoder()
    # out = decoder(out)
    # print(out.shape)
