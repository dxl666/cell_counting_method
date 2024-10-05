import torch
import torch.nn as nn
import copy
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding, \
    LayerNorm
from torch.nn.modules.utils import _pair
from blocks import DA_Module, Block, ChannelAttention, SpatialAttention
import numpy as np
from scipy import ndimage
import logging
import train_utils.configs as configs

logger = logging.getLogger(__name__)


# 使用融合四个尺度特征的多尺度融合模块

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.doubleConv = DoubleConv(in_channels, out_channels)
        self.downSample = Down(out_channels, out_channels)

    def forward(self, x):
        x1 = self.doubleConv(x)
        x2 = self.downSample(x1)
        return x1, x2


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.deConv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deConv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.doubleConv = DoubleConv(in_channels, out_channels * 2)
        self.upSample = Up(out_channels * 2, out_channels)

    def forward(self, x_encoder, x_decoder):
        x = self.doubleConv(x_decoder)
        x = self.upSample(x)
        result = torch.cat((x, x_encoder), dim=1)
        return result


# three times downSample
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(4)]
        self.down1 = DownSample(3, out_channels[0])
        self.down2 = DownSample(out_channels[0], out_channels[1])
        self.down3 = DownSample(out_channels[1], out_channels[2])
        self.down4 = DownSample(out_channels[2], out_channels[3])

    def forward(self, x):
        skip_features = []
        skip_feature, x = self.down1(x)
        skip_features.append(skip_feature)
        skip_feature, x = self.down2(x)
        skip_features.append(skip_feature)
        skip_feature, x = self.down3(x)
        skip_features.append(skip_feature)
        skip_feature, x = self.down4(x)
        skip_features.append(skip_feature)
        return skip_features, x


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        # upSample
        self.encoder = Encoder()
        self.config = config
        image_size = _pair(img_size)  # 224->(224,224)
        if config.patches.get("grid") is not None:  # exists encoder
            # remember train coder add grid = img_size/16
            grid_size = 14
            patch_size = (image_size[0] // 16 // grid_size, image_size[1] // 16 // grid_size)  # 224/8/14 = 2
            n_patches = (grid_size * grid_size)
        else:
            patch_size = config.patches["size"]
            n_patches = (image_size[0] / patch_size[0]) * (img_size[1] / patch_size[1])
        in_channels = 512
        self.embeddingLayer = nn.Conv2d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(config.hidden_size)
        self.positionEmbedding = Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        skip_features, decoder_feature = self.encoder(x)
        decoder_feature = self.embeddingLayer(decoder_feature)
        decoder_feature = decoder_feature + self.sa(decoder_feature)
        decoder_feature = decoder_feature + self.ca(decoder_feature)
        b, hidden, _, _ = decoder_feature.size()
        decoder_feature = decoder_feature.view(b, hidden, -1)
        decoder_feature = decoder_feature.permute(0, 2, 1)
        result = decoder_feature + self.positionEmbedding
        result = self.dropout(result)
        return skip_features, result


# transformer's encoder
class TransEncoder(nn.Module):
    def __init__(self, config, vis):
        super(TransEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer['num_layers']):
            layer = Block(config, self.vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        weights = []
        for layer_block in self.layer:
            weight, x = layer_block(x)
            if self.vis:
                weights.append(weight)
        result = self.encoder_norm(x)
        return weights, result


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embedding = Embeddings(config, img_size, 3)
        self.transEncoder = TransEncoder(config, vis)

    def forward(self, x):
        skip_features, decoder_feature = self.embedding(x)
        weights, decoder_feature = self.transEncoder(decoder_feature)
        return weights, skip_features, decoder_feature


class MSFF(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(MSFF, self).__init__()
        in_channels = in_channels1 + in_channels2
        self.sa1 = SpatialAttention()
        self.ca1 = ChannelAttention(in_channels)
        self.sa2 = SpatialAttention()
        self.ca2 = ChannelAttention(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels1, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels2, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        shape1 = x1.shape[-1]
        shape2 = x2.shape[-1]
        x2_1 = nn.functional.interpolate(x2, (shape1, shape1), mode='bilinear', align_corners=True)
        x1_2 = nn.functional.adaptive_avg_pool2d(x1, (shape2, shape2))
        x1_1 = torch.cat((x1, x2_1), dim=1)
        x2_2 = torch.cat((x2, x1_2), dim=1)
        x1_1 = x1_1 + self.ca1(self.sa1(x1_1))
        x2_2 = x2_2 + self.ca1(self.sa1(x2_2))
        x1 = self.conv1(x1_1)
        x2 = self.conv2(x2_2)
        return x1, x2


class MSFF2(nn.Module):
    def __init__(self, base_dim):
        super(MSFF2, self).__init__()
        channels = [base_dim * 2 ** i for i in range(4)]
        channel1 = channels[0] + channels[1]
        channel2 = channels[2] + channels[3]
        channel3 = channel1 + channel2
        self.sa1 = SpatialAttention()
        self.ca1 = ChannelAttention(channel3)
        self.sa2 = SpatialAttention()
        self.ca2 = ChannelAttention(channel3)
        self.convList = nn.ModuleList()
        self.convList.append(nn.Sequential(
            nn.ConvTranspose2d(channel3, channels[0], 2, 2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        ))
        self.convList.append(nn.Sequential(
            nn.Conv2d(channel3, channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        ))
        self.convList.append(nn.Sequential(
            nn.Conv2d(channel3, channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        ))
        self.convList.append(nn.Sequential(
            nn.Conv2d(channel3, channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True)
        ))

    def forward(self, skip_features):
        target_shape1 = skip_features[1].shape[-1]
        target_shape2 = skip_features[2].shape[-1]
        for i in range(len(skip_features)):
            if i == 0:
                skip_features[i] = nn.functional.adaptive_avg_pool2d(skip_features[i], (target_shape1, target_shape1))
            elif i == 3:
                skip_features[i] = nn.functional.interpolate(skip_features[i], (target_shape2, target_shape2),
                                                             mode='bilinear', align_corners=True)
            else:
                continue
        skip_feature1 = torch.cat((skip_features[0], skip_features[1]), dim=1)
        # skip_feature1 = self.ca1(self.sa1(skip_feature1))
        skip_feature2 = torch.cat((skip_features[2], skip_features[3]), dim=1)
        # skip_feature2 = self.ca2(self.sa2(skip_feature2))
        skip_feature2_1 = nn.functional.interpolate(skip_feature2, (target_shape1, target_shape1), mode='bilinear',
                                                    align_corners=True)
        skip_feature1_1 = nn.functional.adaptive_avg_pool2d(skip_feature1, (target_shape2, target_shape2))
        skip_feature1 = torch.cat((skip_feature1, skip_feature2_1), dim=1)
        skip_feature1 = skip_feature1 + self.ca1(self.sa1(skip_feature1))
        skip_feature2 = torch.cat((skip_feature2, skip_feature1_1), dim=1)
        skip_feature2 = skip_feature2 + self.ca2(self.sa1(skip_feature2))
        for j in range(len(skip_features)):
            if j < 2:
                skip_features[j] = self.convList[j](skip_feature1)
            else:
                skip_features[j] = self.convList[j](skip_feature2)
        return skip_features


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]
        skip_channels = [64, 128, 256]
        self.recoverConv = nn.Conv2d(config.hidden_size, out_channels[3], kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels[3])
        self.relu = nn.ReLU()
        self.up1 = UpSample(out_channels[3], out_channels[3])
        self.up2 = UpSample(out_channels[4], out_channels[2])
        self.up3 = UpSample(out_channels[3], out_channels[1])
        self.up4 = UpSample(out_channels[2], out_channels[0])

    def forward(self, skip_features, decoder_feature):
        b, n_patches, hidden_size = decoder_feature.size()
        h, w = int(np.sqrt(n_patches)), int(np.sqrt(n_patches))
        decoder_feature = decoder_feature.permute(0, 2, 1)
        decoder_feature = decoder_feature.contiguous().view(b, hidden_size, h, w)
        decoder_feature = self.recoverConv(decoder_feature)
        decoder_feature = self.batch_norm(decoder_feature)
        decoder_feature = self.relu(decoder_feature)
        decoder_feature = self.up1(skip_features[3], decoder_feature)
        decoder_feature = self.up2(skip_features[2], decoder_feature)
        decoder_feature = self.up3(skip_features[1], decoder_feature)
        decoder_feature = self.up4(skip_features[0], decoder_feature)
        return decoder_feature


class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Output, self).__init__()
        mid_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class SCTransUNet2(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(SCTransUNet2, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(4)]
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.msff = MSFF2(out_channels[0])
        self.decoder = Decoder(config)
        self.out = Output(out_channels[1], num_classes)
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        weights, skip_features, decoder_feature = self.transformer(x)
        skip_features = self.msff(skip_features)
        # new_skip_features = []
        # x1, x2 = self.msff1(skip_features[0], skip_features[1])
        # new_skip_features.append(x1)
        # new_skip_features.append(x2)
        # x1, x2 = self.msff2(skip_features[2], skip_features[3])
        # new_skip_features.append(x1)
        # new_skip_features.append(x2)
        decoder_feature = self.decoder(skip_features, decoder_feature)
        result = self.out(decoder_feature)
        return result

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            # self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            # self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            # self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.transEncoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            # self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            self.transformer.transEncoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            # posemb_new = self.transformer.embeddings.position_embeddings
            posemb_new = self.transformer.embedding.positionEmbedding
            if posemb.size() == posemb_new.size():
                # self.transformer.embeddings.position_embeddings.copy_(posemb)
                self.transformer.embedding.positionEmbedding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                # self.transformer.embeddings.position_embeddings.copy_(posemb)
                self.transformer.embedding.positionEmbedding.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                # self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
                self.transformer.embedding.positionEmbedding.copy_(np2th(posemb))
            # Encoder whole
            for bname, block in self.transformer.transEncoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname + '/')


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
