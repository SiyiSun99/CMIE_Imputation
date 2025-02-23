#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   define the GAIN model
@Author      :   siyi.sun
@Time        :   2025/02/21 01:03:10
"""
import numpy as np
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from utils.return_layer import return_layer
from utils.data_shuffle_noise import data_shuffle_noise
from utils.Performance_store import Performance_store
from utils.Model_test import Model_test
from parameters.Parameters_setting import parameters_setting
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from utils.process_data import processed_data
from sklearn.model_selection import KFold
from tqdm import tqdm


class GAIN(object):

    # The embedding size should be larger than or equal to 2. This is very important.
    # The num of fold should be 5, or 10. For 5, the fold_index should be picked from 0-4, for 10 should be 0-9.

    def __init__(
        self,
        base_path,
        cohort,
        miss_method,
        miss_ratio,
        index_file,
        batch_num,
        epoch,
        sampletest=False,
        index_pick="continuous_first",
    ):

        # Define paths
        self.base_path = base_path
        # Define Dataset
        self.cohort = cohort
        self.miss_method = miss_method
        self.miss_ratio = miss_ratio
        self.index_file = index_file
        self.batch_size = batch_num
        self.index_pick = index_pick

        (
            self.data,
            self.column_name,
            self.column_location,
            self.label_reverse,
            self.df_original,
            self.df_miss,
            self.label_ori,
            self.mask_raw,
        ) = processed_data(
            self.base_path,
            self.cohort,
            self.miss_method,
            self.miss_ratio,
            self.index_file,
            mode="one_hot",
            sampletest=sampletest,
        )
        self.mask = 1.0 - np.isnan(self.data)

        self.para = parameters_setting(
            model_name="GAIN", data_shape=self.data.shape[1], epoch=epoch
        )
        (
            self.cross_validation,
            self.loss_mode,
            self.d_loss_mode,
            self.g_loss_mode,
            self.epoch,
            self.alpha,
            self.loss_balance,
            self.p_hint,
            self.noise_high_limit,
        ) = self.para.return_parameters()
        self.dsn = data_shuffle_noise(
            mode="one_hot", noise_zero=False, high=self.noise_high_limit
        )
        self.model_estimate = Model_test(
            label_reverse=self.label_reverse,
            label_ori=self.label_ori,
            column_location=self.column_location,
            column_name=self.column_name,
            mode="one_hot",
        )
        self.ps = Performance_store(
            self.base_path,
            self.cohort,
            self.miss_method,
            self.miss_ratio,
            self.index_file,
            self.label_reverse,
            self.label_ori,
            self.column_location,
            self.column_name,
            name="GAIN",
            mode="one_hot",
            index_pick=self.index_pick,
        )
        self.network_layer_G, self.network_layer_D = self.para.return_layer_size()

    def return_generative_network(self):
        input_ = Input(shape=(self.network_layer_G[0],))

        for i in range(len(self.network_layer_G) - 1):

            if i == 0:
                output = return_layer(
                    layer_input=input_, output_size=self.network_layer_G[i + 1]
                )
            elif i == len(self.network_layer_G) - 2:
                output = return_layer(
                    layer_input=output,
                    output_size=self.network_layer_G[i + 1],
                    dropout=True,
                    activation="sigmoid",
                )
            else:
                output = return_layer(
                    layer_input=output, output_size=self.network_layer_G[i + 1]
                )

        model = Model(inputs=input_, outputs=output)

        # Return the constructed generative network model
        return model

    def return_discriminator_network(self):
        input_ = Input(shape=(self.network_layer_D[0],))

        for i in range(len(self.network_layer_D) - 1):

            if i == 0:
                output = return_layer(
                    layer_input=input_, output_size=self.network_layer_D[i + 1]
                )
            elif i == len(self.network_layer_D) - 2:
                output = return_layer(
                    layer_input=output,
                    output_size=self.network_layer_D[i + 1],
                    dropout=True,
                    activation="sigmoid",
                )
            else:
                output = return_layer(
                    layer_input=output, output_size=self.network_layer_D[i + 1]
                )

        model = Model(inputs=input_, outputs=output)

        return model

    def loss(self, gen_x, x, m):
        if self.loss_mode == "log_mse_masked":
            for index, i in enumerate(self.column_location):
                if self.label_reverse[index][0] == "con":
                    if index == 0:
                        loss = tf.reduce_sum(
                            (gen_x[:, 0:i] * m[:, 0:i] - x[:, 0:i] * m[:, 0:i]) ** 2
                        )
                    else:
                        loss = loss + tf.reduce_sum(
                            (
                                gen_x[:, self.column_location[index - 1] : i]
                                * m[:, self.column_location[index - 1] : i]
                                - x[:, self.column_location[index - 1] : i]
                                * m[:, self.column_location[index - 1] : i]
                            )
                            ** 2
                        )
                else:
                    if index == 0:
                        loss_target = (
                            -x[:, 0:i] * m[:, 0:i] * tf.math.log(gen_x[:, 0:i] + 1e-8)
                        )
                        loss = self.loss_balance * tf.reduce_sum(loss_target)
                    else:
                        loss_target = (
                            -x[:, self.column_location[index - 1] : i]
                            * m[:, self.column_location[index - 1] : i]
                            * tf.math.log(
                                gen_x[:, self.column_location[index - 1] : i] + 1e-8
                            )
                        )
                        loss = loss + self.loss_balance * tf.reduce_sum(loss_target)

            loss = loss / (tf.reduce_sum(m) + 1e-8)

        else:
            loss = tf.reduce_sum((gen_x * m - x * m) ** 2) / (tf.reduce_sum(m) + 1e-8)
        return loss

    def d_loss(self, m, gen_m):
        if self.d_loss_mode == "log_masked":
            loss = -tf.reduce_mean(
                m * tf.math.log(gen_m + 1e-8)
                + (1.0 - m) * tf.math.log(1.0 - gen_m + 1e-8)
            )
        else:
            loss = tf.reduce_mean((m - gen_m) ** 2)
        return loss

    def g_loss(self, m, gen_m):
        if self.g_loss_mode == "log_masked":
            loss = -tf.reduce_mean((1.0 - m) * tf.math.log(gen_m + 1e-8))
        elif self.g_loss_mode == "log_complete_masked":
            loss = -tf.reduce_mean(
                (1.0 - m) * tf.math.log(gen_m + 1e-8) + m * tf.math.log(gen_m + 1e-8)
            )
        return loss

    def return_defined_network_for_mode(self):
        """
        Defines and returns the generative and discriminator networks along with their solvers.

        Returns:
            G_solver: The optimizer for the generative network.
            D_solver: The optimizer for the discriminator network.
            gen_xpre: The output of the generative network before masking.
            x: Placeholder for input data.
            m: Placeholder for mask data.
            h: Placeholder for hint data.
        """
        x, m, h = self.para.return_placeholder()
        gen_model = self.return_generative_network()
        InputG = tf.concat(values=[x, m], axis=1)
        gen_xpre = gen_model(InputG)
        gen_x = gen_xpre * (1.0 - m) + x * m
        dis_model = self.return_discriminator_network()
        InputD = tf.concat(values=[gen_x, h], axis=1)
        gen_m = dis_model(InputD)
        d_loss = self.d_loss(m=m, gen_m=gen_m)
        g_loss_p1 = self.g_loss(m=m, gen_m=gen_m)
        g_loss_p2 = self.loss(gen_x=gen_xpre, x=x, m=m)
        g_loss = g_loss_p1 + self.alpha * g_loss_p2
        G_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            g_loss, var_list=gen_model.trainable_weights
        )
        D_solver = tf.compat.v1.train.AdamOptimizer().minimize(
            d_loss, var_list=dis_model.trainable_weights
        )

        return G_solver, D_solver, gen_xpre, x, m, h

    def return_hint_of_mask(self, train_m_batch):
        mask_hint = np.random.uniform(
            size=(train_m_batch.shape[0], self.mask.shape[1]), low=0.0, high=1.0
        )
        mask_res = np.float32(mask_hint > 1 - self.p_hint)
        return mask_res * train_m_batch

    def return_Kfold(self, tr, te):
        data = self.data.copy()
        mask = self.mask.copy()

        train_d = data[tr]
        test_d = data[te]
        train_m = mask[tr]
        test_m = mask[te]
        return (
            train_d,
            train_m,
            test_d,
            test_m,
        )

    def train_process(self):
        G_solver, D_solver, gen_x, x, m, h = self.return_defined_network_for_mode()
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        for _ in tqdm(range(self.epoch)):

            train_d, train_m = self.dsn.data_shuffle(
                train_data=self.data.copy(), train_mask=self.mask.copy()
            )
            for iteration in range(int(train_d.shape[0] / self.batch_size)):

                train_d_batch = train_d[
                    iteration * self.batch_size : (iteration + 1) * self.batch_size
                ]
                train_m_batch = train_m[
                    iteration * self.batch_size : (iteration + 1) * self.batch_size
                ]
                train_h_batch = self.return_hint_of_mask(train_m_batch=train_m_batch)

                _ = sess.run(
                    [G_solver],
                    feed_dict={
                        x: train_d_batch,
                        m: train_m_batch,
                        h: train_h_batch,
                    },
                )
                _ = sess.run(
                    [D_solver],
                    feed_dict={
                        x: train_d_batch,
                        m: train_m_batch,
                        h: train_h_batch,
                    },
                )

        data_noised = self.dsn._add_noise_(
            train_data=self.data.copy(), train_mask=self.mask.copy()
        )
        data_imputed = sess.run(gen_x, feed_dict={x: data_noised, m: self.mask})
        data_imputed = data_imputed * (1.0 - self.mask) + data_noised * self.mask

        self.ps.save_results(imputed_data=data_imputed, mask_df=self.mask_raw)
        sess.close()
        tf.keras.backend.clear_session()

    def train_process_sample(self):
        G_solver, D_solver, gen_x, x, m, h = self.return_defined_network_for_mode()
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        con_list, cat_list = [], []
        for _ in tqdm(range(self.epoch)):

            train_d, train_m = self.dsn.data_shuffle(
                train_data=self.data.copy(), train_mask=self.mask.copy()
            )
            for iteration in range(int(train_d.shape[0] / self.batch_size)):

                train_d_batch = train_d[
                    iteration * self.batch_size : (iteration + 1) * self.batch_size
                ]
                train_m_batch = train_m[
                    iteration * self.batch_size : (iteration + 1) * self.batch_size
                ]
                train_h_batch = self.return_hint_of_mask(train_m_batch=train_m_batch)

                _ = sess.run(
                    [G_solver],
                    feed_dict={
                        x: train_d_batch,
                        m: train_m_batch,
                        h: train_h_batch,
                    },
                )
                _ = sess.run(
                    [D_solver],
                    feed_dict={
                        x: train_d_batch,
                        m: train_m_batch,
                        h: train_h_batch,
                    },
                )

            data_noised = self.dsn._add_noise_(
                train_data=self.data.copy(), train_mask=self.mask.copy()
            )
            data_imputed = sess.run(gen_x, feed_dict={x: data_noised, m: self.mask})
            data_imputed = data_imputed * (1.0 - self.mask) + data_noised * self.mask
            con_loss, cat_accuracy = self.model_estimate.model_test(
                data=data_imputed, mask=self.mask.copy(), df_original=self.df_original
            )
            con_list.append(con_loss)
            cat_list.append(cat_accuracy)

        index_re = self.ps.select_best_index(
            continuous_metrics=con_list, categorical_accuracies=cat_list
        )
        sess.close()
        tf.keras.backend.clear_session()

        return index_re
