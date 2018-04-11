import math
import torch
import torch.nn as nn
from pytorch_zoo.abstract_model import EncoderDecoder, get_slice, Upscale, UnetDecoderBlock, ConvBottleneck, SumBottleneck, UnetBNDecoderBlock
import torch.nn.functional as F


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
            return encoder.mod1
        elif layer == 1:
            return nn.Sequential(
                encoder.pool2,
                encoder.mod2
            )
        elif layer == 2:
            return nn.Sequential(
                encoder.pool3,
                encoder.mod3
            )
        elif layer == 3:
            return nn.Sequential(
                encoder.pool4,
                encoder.mod4
            )
        elif layer == 4:
            return nn.Sequential(
                encoder.pool5,
                encoder.mod5
            )
        elif layer == 5:
            return nn.Sequential(
                encoder.pool6,
                encoder.mod6,
                encoder.mod7
            )

    @property
    def first_layer_params_names(self):
        return ['mod1.conv1']

class WideResnet38(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.first_layer_stride_two = False
        super().__init__(num_classes, num_channels, encoder_name='wideresnet38')

if __name__ == '__main__':
    images = torch.autograd.Variable(torch.randn(16, 3, 256, 256), volatile=True).cuda()

    # model = UNet11bn(1).cuda()
    # print(ret.data.shape)
    # model = Resnet34(1).cuda()
    # ret = model.forward(images)
    # print(ret.data.shape)
    model = Resnet(3, 3, 'wideresnet38').cuda()
    ret = model.forward(images)
    # assert set((model.first_layer_params + model.layers_except_first_params)) == set(model.parameters())
    print(ret)
