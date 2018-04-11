from models.unets import resnet152_fpn, resnet101_fpn, resnet50_fpn, xception_fpn,  densenet_fpn, inception_resnet_v2_fpn


def make_model(network, input_shape):
    if network == 'resnet101_softmax':
        return resnet101_fpn(input_shape,channels=3, activation="softmax")
    elif network == 'resnet152_2':
        return resnet152_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnet101_2':
        return resnet101_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnet50_2':
        return resnet50_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnetv2':
        return inception_resnet_v2_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnetv2_3':
        return inception_resnet_v2_fpn(input_shape, channels=3, activation="sigmoid")
    elif network == 'densenet169':
        return densenet_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'densenet169_softmax':
        return densenet_fpn(input_shape, channels=3, activation="softmax")
    elif network == 'resnet101_unet_2':
        return resnet101_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'xception_fpn':
        return xception_fpn(input_shape, channels=2, activation="sigmoid")
    elif network == 'resnet50_2':
        return resnet50_fpn(input_shape, channels=2, activation="sigmoid")
    else:
        raise ValueError('unknown network ' + network)