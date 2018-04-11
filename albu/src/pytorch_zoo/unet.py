import math
import torch
import torch.nn as nn
from pytorch_zoo.abstract_model import EncoderDecoder, get_slice, Upscale, UnetDecoderBlock, ConvBottleneck, SumBottleneck, UnetBNDecoderBlock, PathAggregationEncoderDecoder, UnetDoubleDecoderBlock, DPEncoderDecoder
import torch.nn.functional as F

class Unet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name):
        if not hasattr(self, 'decoder_type'):
            self.decoder_type = Upscale.upsample_bilinear
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'need_center'):
            self.need_center = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck
        super().__init__(num_classes, num_channels, encoder_name)

    @property
    def first_layer_params_names(self):
        return ['encoder_stages.0.0']


class ClassicUnet(Unet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='classic_unet')

    def get_encoder(self, encoder, layer):
        features = encoder.features
        if layer == 0:
            return nn.Sequential(*get_slice(features, 0, 4))
        if layer == 1:
            return nn.Sequential(*get_slice(features, 4, 9))
        if layer == 2:
            return nn.Sequential(*get_slice(features, 9, 14))
        if layer == 3:
            return nn.Sequential(*get_slice(features, 14, 19))
        if layer == 4:
            return nn.Sequential(*get_slice(features, 19, 24))


class Vgg11bn(Unet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='vgg11_bn')

    def get_encoder(self, encoder, layer):
        features = encoder.features
        if layer == 0:
            return nn.Sequential(*get_slice(features, 0, 3))
        if layer == 1:
            return nn.Sequential(*get_slice(features, 3, 7))
        if layer == 2:
            return nn.Sequential(*get_slice(features, 7, 14))
        if layer == 3:
            return nn.Sequential(*get_slice(features, 14, 20))
        if layer == 4:
            return nn.Sequential(*get_slice(features, 20, 27))

class Vgg16bn(Unet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='vgg16_bn')

    def get_encoder(self, encoder, layer):
        features = encoder.features
        if layer == 0:
            return nn.Sequential(*get_slice(features, 0, 6))
        if layer == 1:
            return nn.Sequential(*get_slice(features, 6, 13))
        if layer == 2:
            return nn.Sequential(*get_slice(features, 13, 23))
        if layer == 3:
            return nn.Sequential(*get_slice(features, 23, 33))
        if layer == 4:
            return nn.Sequential(*get_slice(features, 33, 43))


class Vgg11(Unet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='vgg11')

    def get_encoder(self, encoder, layer):
        features = encoder.features
        if layer == 0:
            return nn.Sequential(*get_slice(features, 0, 2))
        if layer == 1:
            return nn.Sequential(*get_slice(features, 2, 5))
        if layer == 2:
            return nn.Sequential(*get_slice(features, 5, 10))
        if layer == 3:
            return nn.Sequential(*get_slice(features, 10, 15))
        if layer == 4:
            return nn.Sequential(*get_slice(features, 15, 20))


class Resnet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name):
        if not hasattr(self, 'decoder_type'):
            self.decoder_type = Upscale.upsample_bilinear
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'need_center'):
            self.need_center = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck
        super().__init__(num_classes, num_channels, encoder_name)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['conv1']

class Resnet34_transconv(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.transposed_conv
        super().__init__(num_classes, num_channels, encoder_name='resnet34')


class Resnet34_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        super().__init__(num_classes, num_channels, encoder_name='resnet34')

class Resnet34_double(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        self.decoder_block = UnetDoubleDecoderBlock
        self.bottleneck_type = SumBottleneck
        super().__init__(num_classes, num_channels, encoder_name='resnet34')


class Resnet34_embeddings(Resnet):
     def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        super().__init__(num_classes, num_channels, encoder_name='resnet34')

     def make_final_classifier(self, in_filters, num_classes):
         return nn.Conv2d(in_filters, 16, 3, padding=1)

class DilatedNet(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(self.filters[-1], self.filters[-1], 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(self.filters[-1], self.filters[-1], 3, padding=4, dilation=4)
        self.conv3 = nn.Conv2d(self.filters[-1], self.filters[-1], 3, padding=8, dilation=8)
        self.conv4 = nn.Conv2d(self.filters[-1], self.filters[-1], 3, padding=16, dilation=16)

    def forward(self, x):
        fst = F.relu(self.conv1(x))
        snd = F.relu(self.conv2(fst))
        thrd = F.relu(self.conv3(snd))
        fourth = F.relu(self.conv4(thrd))
        return x + fst + snd + thrd + fourth

class DilatedResnet34(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        self.need_center = True
        super().__init__(num_classes, num_channels, encoder_name='resnet34')

    def get_center(self):
        return DilatedNet(self.filters)


class Resnet34_bn_sum(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        self.bottleneck_type = SumBottleneck
        self.decoder_block = UnetBNDecoderBlock
        super().__init__(num_classes, num_channels, encoder_name='resnet34')

class Resnet34_sum(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        self.bottleneck_type = SumBottleneck
        # self.decoder_block = UnetBNDecoderBlock
        super().__init__(num_classes, num_channels, encoder_name='resnet34')



class Resnet34_pixelshuffle(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.pixel_shuffle
        super().__init__(num_classes, num_channels, encoder_name='resnet34')


class Resnet18(Resnet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='resnet18')


class Resnet34SpatialDropout(Resnet):
    #todo
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='resnet34')

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, in_filters // 2, 3, padding=1),
            nn.BatchNorm2d(in_filters // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_filters // 2, in_filters // 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_filters // 2, num_classes, 3, padding=1),
        )


class Resnet50(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        super().__init__(num_classes, num_channels, encoder_name='resnet50')


class Resnet50_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        super().__init__(num_classes, num_channels, encoder_name='resnet50')


class Incv3(EncoderDecoder):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='inceptionv3')

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.Conv2d_1a_3x3,
                encoder.Conv2d_2a_3x3,
                encoder.Conv2d_2b_3x3
            )
        elif layer == 1:
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                encoder.Conv2d_3b_1x1,
                encoder.Conv2d_4a_3x3
            )
        elif layer == 2:
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                encoder.Mixed_5b,
                encoder.Mixed_5c,
                encoder.Mixed_5d
            )
        elif layer == 3:
            return nn.Sequential(
                encoder.Mixed_6a,
                encoder.Mixed_6b,
                encoder.Mixed_6c,
                encoder.Mixed_6d,
                encoder.Mixed_6e
            )
        elif layer == 4:
            return nn.Sequential(
                encoder.Mixed_7a,
                encoder.Mixed_7b,
                encoder.Mixed_7c
            )

    @property
    def first_layer_params_names(self):
        return ['Conv2d_1a_3x3.conv']


class LinkNetIncvRes2(EncoderDecoder):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='inception_resnet_v2')

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                self.inc_res_v2.conv2d_1a,
                self.inc_res_v2.conv2d_2a,
                self.inc_res_v2.conv2d_2b,
            )
        elif layer == 1:
            return nn.Sequential(
                self.inc_res_v2.maxpool_3a,
                self.inc_res_v2.conv2d_3b,
                self.inc_res_v2.conv2d_4a,
            )
        elif layer == 2:
            return nn.Sequential(
                self.inc_res_v2.maxpool_5a,
                self.inc_res_v2.mixed_5b,
                self.inc_res_v2.repeat
            )
        elif layer == 3:
            return nn.Sequential(
                self.inc_res_v2.mixed_6a,
                self.inc_res_v2.repeat_1
            )
        elif layer == 4:
            return nn.Sequential(
                self.inc_res_v2.mixed_7a,
                self.inc_res_v2.repeat_2,
                self.inc_res_v2.block8,
                self.inc_res_v2.conv2d_7b,
            )


class PAResnet34(PathAggregationEncoderDecoder):
    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

class DPNUnet(DPEncoderDecoder):
    def __init__(self, num_classes, num_channels=3, encoder_name='dpn92'):
        # self.decoder_block = UnetDoubleDecoderBlock
        self.bottleneck_type = ConvBottleneck
        super().__init__(num_classes, num_channels, encoder_name)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.blocks['conv1_1'].conv, #conv
                encoder.blocks['conv1_1'].bn, #bn
                encoder.blocks['conv1_1'].act, #relu
            )
        elif layer == 1:
            return nn.Sequential(
                encoder.blocks['conv1_1'].pool, #maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        elif layer == 2:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        elif layer == 3:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        elif layer == 4:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

if __name__ == '__main__':
    images = torch.autograd.Variable(torch.randn(1, 3, 256, 256), volatile=True).cuda()

    # model = UNet11bn(1).cuda()
    # print(ret.data.shape)
    # model = Resnet34(1).cuda()
    # ret = model.forward(images)
    # print(ret.data.shape)
    model = DPNUnet(1, 3)
    print(model.state_dict().keys())
    # ret = model.forward(images)
    # print(ret)
    # assert set((model.first_layer_params + model.layers_except_first_params)) == set(model.parameters())
