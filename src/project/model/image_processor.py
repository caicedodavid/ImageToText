import numpy as np
import cv2


class ImageProcessor:
    """ Image processing before being fed to the CRNN
    Borrowed from [1] but with a more OOP approach. Still
    the mayor part of the base code is the same
    References
    ----------
    .. [1]  https://github.com/kurapan/CRNN/blob/master/eval.py
    """
    def __init__(self, img_path, cfg):
        """ ImageProcessor constructor
        :param img_path: the path of the image
        :param cfg: configuration object
        """
        self.img_path = img_path
        self.cfg = cfg

    def process(self):
        img = self.__load_image()
        return self.__preprocess_image(img)

    def __preprocess_image(self, img):
        """ Preprocesses the image by resizing, padding and other
        transformations so it will have the dimensions required by the model
        :param img:
        :return: image
        """
        if img.shape[1] / img.shape[0] < 6.4:
            img = self.__pad_image(img, (self.cfg.width, self.cfg.height),
                                   self.cfg.nb_channels)
        else:
            img = self.__resize_image(img, (self.cfg.width, self.cfg.height))
        if self.cfg.nb_channels == 1:
            img = img.transpose([1, 0])
        else:
            img = img.transpose([1, 0, 2])
        img = np.flip(img, 1)
        img = img / 255.0
        if self.cfg.nb_channels == 1:
            img = img[:, :, np.newaxis]
        return img

    def __load_image(self):
        """Loads the image from filesystem
        :return: image
        """
        if self.cfg.nb_channels == 1:
            return cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imread(self.img_path)

    def __pad_image(self, img, img_size, nb_channels):
        # img_size : (width, height)
        # loaded_img_shape : (height, width)
        img_reshape = cv2.resize(img, (
            int(img_size[1] / img.shape[0] * img.shape[1]),
            img_size[1]
        ))
        if nb_channels == 1:
            padding = np.zeros((img_size[1], img_size[0] - int(
                img_size[1] / img.shape[0] * img.shape[1])), dtype=np.int32)
        else:
            padding = np.zeros((img_size[1], img_size[0] - int(
                img_size[1] / img.shape[0] * img.shape[1]), nb_channels),
                               dtype=np.int32)
        img = np.concatenate([img_reshape, padding], axis=1)
        return img

    def __resize_image(self, img, img_size):
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        img = np.asarray(img)
        return img
