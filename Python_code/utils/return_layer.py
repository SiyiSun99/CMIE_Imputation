#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   return the layer
@Author      :   siyi.sun
@Time        :   2025/02/21 01:47:26
"""
import tensorflow as tf
from tensorflow.python.keras.layers import (
    Dense,
    Dropout,
    Activation,
    LayerNormalization,
)


def return_layer(
    layer_input, output_size, norm=False, dropout=False, activation="relu"
):
    output = Dense(
        output_size,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(layer_input)
    if norm:
        output = LayerNormalization()(output)
    if activation:
        output = Activation(activation=activation)(output)
    if dropout:
        output = Dropout(rate=0.5)(output)

    return output
