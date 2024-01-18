import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Multiply, Conv2D, Activation
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

def conv_block(x, filter_num,dropout,kernel_size=(3, 3),batch_norm=False,regularization = False,is_relu =True):

    if regularization is True:
        conv = layers.Conv2D(filters=filter_num, kernel_size=kernel_size,
                         padding="same",kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                         )(x)
    else:
        conv = layers.Conv2D(filters=filter_num, kernel_size=kernel_size,
                             padding="same")(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    if is_relu:
        conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def dot_product(seg, cls):
    b, h, w, n = k.backend.int_shape(seg)
    seg = tf.reshape(seg, [-1, h * w, n])
    final = tf.einsum("ijk,ik->ijk", seg, cls)
    final = tf.reshape(final, [-1, h, w, n])
    return final

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    # by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape
    # (None, 256,256,6), if specified axis=3 and rep=2.

    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                         arguments={'repnum': rep})(tensor)


def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def A_unet3plus(input_shape, output_channels,dropout=0,batch_norm=True,regularization=True):
    """ UNet3+ base model """
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(
        shape=input_shape,
        name="input_layer"
    )  # 320*320*3

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filter_num=filters[0],dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 320*320*64

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filter_num=filters[1],dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*128

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filter_num=filters[2],dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*256

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filter_num=filters[3],dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 40*40*512

    # block 5
    # bottleneck layer
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    e5 = conv_block(e5, filter_num=filters[4],dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 20*20*1024

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    gating_e4 = gating_signal(e5, 8 * cat_channels, batch_norm)
    att_e4 = attention_block(e4, gating_e4, 8 * cat_channels)

    gating_e3 = gating_signal(e4, 4 * cat_channels, batch_norm)
    att_e3 = attention_block(e3, gating_e3, 4 * cat_channels)

    gating_e2 = gating_signal(e3, 4 * cat_channels, batch_norm)
    att_e2 = attention_block(e2, gating_e2, 4 * cat_channels)

    gating_e1 = gating_signal(e2, 4 * cat_channels, batch_norm)
    att_e1 = attention_block(e1, gating_e1, 4 * cat_channels)

    """ d4 """
    e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(att_e1)  # 320*320*64  --> 40*40*64
    e1_d4 = conv_block(e1_d4, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 320*320*64  --> 40*40*64

    e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(att_e2)  # 160*160*128 --> 40*40*128
    e2_d4 = conv_block(e2_d4, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*128 --> 40*40*64

    e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(att_e3)  # 80*80*256  --> 40*40*256
    e3_d4 = conv_block(e3_d4,  filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*256  --> 40*40*64

    e4_d4 = conv_block(att_e4, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 40*40*512  --> 40*40*64

    e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
    e5_d4 = conv_block(e5_d4, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 20*20*1024  --> 20*20*64

    d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, filter_num=upsample_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 40*40*320  --> 40*40*320

    """ d3 """
    e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(att_e1)  # 320*320*64 --> 80*80*64
    e1_d3 = conv_block(e1_d3, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*64 --> 80*80*64

    e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(att_e2)  # 160*160*256 --> 80*80*256
    e2_d3 = conv_block(e2_d3, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*256 --> 80*80*64

    e3_d3 = conv_block(att_e3, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*512 --> 80*80*64

    e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 40*40*320 --> 80*80*320
    e4_d3 = conv_block(e4_d3, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*320 --> 80*80*64

    e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
    e5_d3 = conv_block(e5_d3, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*320 --> 80*80*64

    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = conv_block(d3, upsample_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 80*80*320 --> 80*80*320

    """ d2 """
    e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(att_e1)  # 320*320*64 --> 160*160*64
    e1_d2 = conv_block(e1_d2, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*64 --> 160*160*64

    e2_d2 = conv_block(att_e2, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*256 --> 160*160*64

    d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
    d3_d2 = conv_block(d3_d2, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*320 --> 160*160*64

    d4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  # 40*40*320 --> 160*160*320
    d4_d2 = conv_block(d4_d2, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*320 --> 160*160*64

    e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
    e5_d2 = conv_block(e5_d2, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*320 --> 160*160*64

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, filter_num=upsample_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*320 --> 160*160*320

    """ d1 """
    e1_d1 = conv_block(att_e1, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 320*320*64 --> 320*320*64

    d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
    d2_d1 = conv_block(d2_d1, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 160*160*320 --> 160*160*64

    d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
    d3_d1 = conv_block(d3_d1, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 320*320*320 --> 320*320*64

    d4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)  # 40*40*320 --> 320*320*320
    d4_d1 = conv_block(d4_d1, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 320*320*320 --> 320*320*64

    e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
    e5_d1 = conv_block(e5_d1, filter_num=cat_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 320*320*320 --> 320*320*64

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, filter_num=upsample_channels,dropout=dropout, batch_norm=batch_norm,regularization = regularization)  # 320*320*320 --> 320*320*320

    # last layer does not have batchnorm and relu
    d = conv_block(d1, filter_num=output_channels,dropout=0, batch_norm=False,regularization = regularization,is_relu=False)

    output = k.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=[output], name='unet3plusmore')
