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


def get_vgg_encoder(**kwargs):
    return VGG_Encoder(**kwargs)


if __name__ == '__main__':

    input_tensor = torch.randn((4, 3, 32, 32), dtype=torch.float32)
    # model = get_vgg_encoder()
    model = nn.ConvTranspose2d(3, 6, 3, stride=2, padding=1, output_padding=1)
    print(model(input_tensor).shape)


    encoder = Encoder()
    out = encoder(input_tensor)
    print(out.shape)

    decoder = Decoder()
    out = decoder(out)
    print(out.shape)