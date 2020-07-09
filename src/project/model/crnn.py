import numpy as np
import os
import tensorflow.keras.backend as k
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     MaxPooling2D, Reshape, Dense, LSTM,
                                     add, concatenate, Dropout, Lambda,
                                     Flatten)
from tensorflow.keras.models import Model

from project.model.config import Config
from project.model.image_processor import ImageProcessor
from project.model.stn import SpatialTransformer


def ctc_lambda_func(args):
    iy_pred, ilabels, iinput_length, ilabel_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    return k.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)


def loc_net(input_shape):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    w = np.zeros((64, 6), dtype='float32')
    weights = [w, b.flatten()]
    loc_input = Input(input_shape)
    loc_conv_1 = \
        Conv2D(16, (5, 5), padding='same', activation='relu')(loc_input)
    loc_conv_2 = \
        Conv2D(32, (5, 5), padding='same', activation='relu')(loc_conv_1)
    loc_fla = Flatten()(loc_conv_2)
    loc_fc_1 = Dense(64, activation='relu')(loc_fla)
    loc_fc_2 = Dense(6, weights=weights)(loc_fc_1)

    output = Model(inputs=loc_input, outputs=loc_fc_2)

    return output


class CRNN_STN:
    """ CRRN with Spatial transformer class
    Implements a CRNN. Borrowed from [1] but with a more OOP approach. Still
    the mayor part of the base code is the same
    References
    ----------
    .. [1]  https://github.com/kurapan/CRNN/blob/master/models.py
    """
    def __init__(self):
        self.cfg = Config()
        self.__set_gpus()
        inputs = Input((self.cfg.width, self.cfg.height, self.cfg.nb_channels))
        c_1 = Conv2D(self.cfg.conv_filter_size[0], (3, 3), activation='relu',
                     padding='same', name='conv_1')(inputs)
        c_2 = Conv2D(self.cfg.conv_filter_size[1], (3, 3), activation='relu',
                     padding='same', name='conv_2')(c_1)
        c_3 = Conv2D(self.cfg.conv_filter_size[2], (3, 3), activation='relu',
                     padding='same', name='conv_3')(c_2)
        bn_3 = BatchNormalization(name='bn_3')(c_3)
        p_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(bn_3)

        c_4 = Conv2D(self.cfg.conv_filter_size[3], (3, 3), activation='relu',
                     padding='same', name='conv_4')(p_3)
        c_5 = Conv2D(self.cfg.conv_filter_size[4], (3, 3), activation='relu',
                     padding='same', name='conv_5')(c_4)
        bn_5 = BatchNormalization(name='bn_5')(c_5)
        p_5 = MaxPooling2D(pool_size=(2, 2), name='maxpool_5')(bn_5)

        c_6 = Conv2D(self.cfg.conv_filter_size[5], (3, 3), activation='relu',
                     padding='same', name='conv_6')(p_5)
        c_7 = Conv2D(self.cfg.conv_filter_size[6], (3, 3), activation='relu',
                     padding='same', name='conv_7')(c_6)
        bn_7 = BatchNormalization(name='bn_7')(c_7)

        bn_7_shape = bn_7.get_shape()
        loc_input_shape = (int(bn_7_shape[1]), int(bn_7_shape[2]),
                           int(bn_7_shape[3]))
        stn = SpatialTransformer(
            localization_net=loc_net(loc_input_shape),
            output_size=(loc_input_shape[0], loc_input_shape[1])
        )(bn_7)

        reshape = Reshape(
            target_shape=(
                int(bn_7_shape[1]),
                int(bn_7_shape[2] * bn_7_shape[3])
            ),
            name='reshape'
        )(stn)

        fc_9 = Dense(
            self.cfg.lstm_nb_units,
            activation='relu',
            name='fc_9'
        )(reshape)

        lstm_10 = LSTM(self.cfg.lstm_nb_units, kernel_initializer="he_normal",
                       return_sequences=True, name='lstm_10')(fc_9)
        lstm_10_back = LSTM(
            self.cfg.lstm_nb_units,
            kernel_initializer="he_normal",
            go_backwards=True,
            return_sequences=True,
            name='lstm_10_back'
        )(fc_9)
        lstm_10_add = add([lstm_10, lstm_10_back])

        lstm_11 = LSTM(self.cfg.lstm_nb_units, kernel_initializer="he_normal",
                       return_sequences=True, name='lstm_11')(lstm_10_add)
        lstm_11_back = LSTM(
            self.cfg.lstm_nb_units,
            kernel_initializer="he_normal",
            go_backwards=True,
            return_sequences=True,
            name='lstm_11_back'
        )(lstm_10_add)
        lstm_11_concat = concatenate([lstm_11, lstm_11_back])
        do_11 = Dropout(self.cfg.dropout_rate, name='dropout')(lstm_11_concat)

        fc_12 = Dense(len(self.cfg.characters), kernel_initializer='he_normal',
                      activation='softmax', name='fc_12')(do_11)

        self.prediction_model = Model(inputs=inputs, outputs=fc_12)

        labels = Input(
            name='labels',
            shape=[self.cfg.label_len],
            dtype='float32'
        )
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [fc_12, labels, input_length, label_length])

        self.training_model = Model(
            inputs=[inputs, labels, input_length, label_length],
            outputs=[ctc_loss]
        )

        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, self.cfg.model_path)
        self.training_model.load_weights(abs_file_path)

    def predict_text(self, img_name) -> str:
        processed_img = ImageProcessor(img_name, self.cfg).process()
        y_pred = self.prediction_model.predict(
            processed_img[np.newaxis, :, :, :]
        )

        shape = y_pred[:, 2:, :].shape
        ctc_decode = k.ctc_decode(
            y_pred[:, 2:, :],
            input_length=np.ones(shape[0]) * shape[1]
        )[0][0]
        ctc_out = k.get_value(ctc_decode)[:, :self.cfg.label_len]
        result_str = ''.join([self.cfg.characters[c] for c in ctc_out[0]])
        result_str = result_str.replace('-', '')
        return result_str

    def __set_gpus(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpus)[1:-1]
