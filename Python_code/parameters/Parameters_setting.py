import numpy as np
import tensorflow as tf


class parameters_setting:
    def __init__(self, model_name, data_shape, epoch=100):
        """
        Common parameters for all deep learning models
        """
        self.model = model_name  # model's name
        self.data_shape = data_shape  # Input data shape
        self.cross_validation = None  # Whether to do the cross-validation or not
        self.epoch = epoch  # Num of epochs
        tf.compat.v1.disable_eager_execution()

    def return_parameters(self):
        return self.return_gain_parameters()

    def return_layer_size(self):
        return self.return_gain_layer_size()

    def return_placeholder(self):
        return self.return_gain_placeholder()

    ###############################################         GAIN          ###################################################################

    def return_gain_parameters(self):
        """
        Define the GAIN's structure info
        """
        cross_validation = self.cross_validation  # whether to do cross-validation
        # L_g = l + alpha * c_g
        loss_mode = "mse_masked"  # detail loss calculation for l
        d_loss_mode = "log_masked"  # c_d
        g_loss_mode = "log_masked"  # c_g
        epoch = self.epoch  # num of epochs
        alpha = 10.0  # alpha as balance parameters
        loss_balance = 1.0
        p_hint = 0.8
        noise_high_limit = 1e-1  # noise range
        return (
            cross_validation,
            loss_mode,
            d_loss_mode,
            g_loss_mode,
            epoch,
            alpha,
            loss_balance,
            p_hint,
            noise_high_limit,
        )

    def return_gain_layer_size(self):
        """
        Define the GAIN's layers for G and D.
        """
        network_layer_G = [
            self.data_shape * 2,
            self.data_shape * 1,
            self.data_shape * 1,
        ]
        network_layer_D = [
            self.data_shape * 2,
            self.data_shape * 1,
            self.data_shape * 1,
        ]
        return network_layer_G, network_layer_D

    def return_gain_placeholder(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.data_shape])
        m = tf.compat.v1.placeholder(tf.float32, shape=[None, self.data_shape])
        h = tf.compat.v1.placeholder(tf.float32, shape=[None, self.data_shape])
        return x, m, h
