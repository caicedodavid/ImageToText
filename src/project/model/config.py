import os
import string


class Config:
    """ Config class that will contain all of the hyperparameters of the
    CRNN
    """
    def __init__(self, data=None):
        """ Config class constructor
        :param data: dict of hyperparameters that can override defaults and
        environment values
        """
        if data is None:
            data = {}

        self.model_path = os.environ.get(
            'MODEL_PATH',
            'prediction_model.hdf5'
        )
        self.data_path = os.environ.get('DATA_PATH', '')
        self.gpus = os.environ.get('GPUS', [0])
        self.characters = os.environ.get(
            'CHARACTERS',
            '0123456789' + string.ascii_lowercase+'-'
        )
        self.label_len = os.environ.get('LABEL_LEN', 16)
        self.nb_channels = os.environ.get('NB_CHANNELS', 1)
        self.width = os.environ.get('WIDTH', 200)
        self.height = os.environ.get('HEIGHT', 31)
        self.model = 'CRNN_STN'
        self.conv_filter_size = [64, 128, 256, 256, 512, 512, 512]
        self.lstm_nb_units = os.environ.get('LSTM_NB_UNITS', 128)
        self.timesteps = os.environ.get('TIMESTEPS', 50)
        self.dropout_rate = os.environ.get('DROPOUT_RATE', 0)

        for key, value in data:
            self.__setattr__(key, value)
