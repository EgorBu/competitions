from typing import Callable, List, Tuple, Union
import warnings

import numpy as np
import keras
from keras import backend as kbackend
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, \
    TimeDistributed, Input, Dropout, MaxPooling2D, Conv2DTranspose
from keras.models import Model
try:
    import tensorflow as tf
except ImportError:
    warnings.warn("Tensorflow is not installed, dependent functionality is unavailable.")


LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
DATA_FORMAT = "channels_last"


def register_metric(metric: Union[str, Callable]) -> Union[str, Callable]:
    """
    Decorator function to register the metrics in the METRICS constant.

    :param metric: name of the tensorflow metric or custom function metric.
    :return: the metric.
    """
    assert isinstance(metric, str) or callable(metric)
    METRICS.append(metric)
    return metric


def add_output_layer(hidden_layer: tf.Tensor) -> keras.layers.wrappers.TimeDistributed:
    """
    Applies a Dense layer to each of the timestamps of a hidden layer, independently.
    The output layer has 1 sigmoid per character which predicts if there is a space or not
    before the character.

    :param hidden_layer: hidden layer before the output layer.
    :return: output layer.
    """
    norm_input = BatchNormalization()(hidden_layer)
    return TimeDistributed(Dense(1, activation="sigmoid"))(norm_input)


def add_conv(X: tf.Tensor, filters: List[int], kernel_sizes: List[int],
             output_n_filters: int, rate: float=0.5, padding="same") -> tf.Tensor:
    """
    Builds a single convolutional layer.

    :param X: input layer.
    :param filters: number of output filters in the convolution.
    :param kernel_sizes: list of lengths of the 1D convolution window.
    :param output_n_filters: number of 1D output filters.
    :return: output layer.
    """
    # normalize the input
    # X = BatchNormalization()(X)
    X = Dropout(rate)(X)

    # add convolutions
    convs = []
    for n_filters, kernel_size in zip(filters, kernel_sizes):
        conv = Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding,
                      activation="relu")
        convs.append(conv(X))

    # concatenate all convolutions
    if len(convs) > 1:
        conc = Concatenate(axis=-1)(convs)
        # conc = BatchNormalization()(conc)
        conc = Dropout(rate)(conc)
    else:
        conc = convs[0]

    # dimensionality reduction
    conv = Conv2D(filters=output_n_filters, kernel_size=1, padding="same", activation="relu")
    return conv(conc)


def build_unet(filters: List[int]=[32, 32], output_n_filters: int=16,
               kernel_sizes: List[int]=[3, 4], optimizer: str="rmsprop", width: int=101,
               height: int=101, channels: int=1, rate: float=0.3) -> keras.engine.training.Model:
    """
    Builds a CNN model with the parameters specified as arguments.

    :param filters: number of output filters in the convolution.
    :param output_n_filters: number of 1d output filters.
    :param kernel_sizes: list of lengths of the 1D convolution window.
    :param optimizer: algorithm to use as an optimizer for the CNN.
    :param width: width of input image in pixels.
    :param height: height of input image in pixels.
    :param channels: number of channels in image.
    :param rate: dropout rate.
    :return: compiled CNN model.
    """
    def increase_sizes(sizes, ntimes):
        return list(map(lambda x: int(x * ntimes), sizes))

    # prepare the model
    input_layer = Input(shape=(width, height, channels))
    hidden_layer = input_layer

    # h1
    h1 = add_conv(hidden_layer, filters=filters, kernel_sizes=kernel_sizes,
                  output_n_filters=output_n_filters, rate=rate)
    p1 = MaxPooling2D((2, 2))(h1)
    print("p1.shape", p1.shape)
    # 50 * 50

    # h2
    h2 = add_conv(p1, filters=increase_sizes(filters, 2), kernel_sizes=kernel_sizes,
                  output_n_filters=int(output_n_filters * 2), rate=rate)
    p2 = MaxPooling2D((2, 2))(h2)
    print("p2.shape", p2.shape)
    # 25 * 25

    # h3
    h3 = add_conv(p2, filters=increase_sizes(filters, 4), kernel_sizes=kernel_sizes,
                  output_n_filters=int(output_n_filters * 4), rate=rate)
    p3 = MaxPooling2D((2, 2))(h3)
    print("p3.shape", p3.shape)
    # 12 * 12

    # middle
    middle = add_conv(p3, filters=increase_sizes(filters, 4), kernel_sizes=kernel_sizes,
                      output_n_filters=output_n_filters, rate=rate)

    # new_cols = (cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1]
    # 25 = (12 - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1]
    # strides 2, kernel_size 4, padding 1, output_padding 1

    # d1
    d1 = Conv2DTranspose(filters=output_n_filters * 4, strides=[2, 2], padding="valid",
                         kernel_size=[4, 4], activation="relu")(middle)
    d1 = add_conv(d1, filters=[output_n_filters * 4], kernel_sizes=[2],
                  output_n_filters=output_n_filters * 4, rate=rate, padding="valid")
    print("d1.shape", d1.shape)
    conc = Concatenate(axis=-1)([d1, h3])

    # d2
    d2 = Conv2DTranspose(filters=output_n_filters * 2, strides=[2, 2], padding="same",
                         kernel_size=[3, 3])(conc)
    print("d2.shape", d2.shape)
    conc = Concatenate(axis=-1)([d2, h2])
    conc = add_conv(conc, filters=increase_sizes(filters, 2), kernel_sizes=kernel_sizes,
                    output_n_filters=output_n_filters, rate=rate)

    # d3
    d3 = Conv2DTranspose(filters=output_n_filters, strides=[2, 2], padding="VALID",
                         kernel_size=[4, 4])(conc)
    d3 = add_conv(d3, filters=[output_n_filters], kernel_sizes=[2],
                  output_n_filters=output_n_filters, rate=rate, padding="valid")
    print("d2.shape", d2.shape)
    conc = Concatenate(axis=-1)([d3, h1])

    # output
    unet_output = Dropout(rate)(conc)
    output = Conv2D(1, 1, activation='sigmoid')(unet_output)

    # compile the model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
    return model


# @register_metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        kbackend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return kbackend.mean(kbackend.stack(prec), axis=0)
