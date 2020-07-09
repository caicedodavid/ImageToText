import os
import requests
from tempfile import NamedTemporaryFile


class Image:
    """ Creates a temporary file from an image url
    """
    def __init__(self, image_url):
        os.makedirs("tmp", exist_ok=True)
        response = requests.get(image_url, allow_redirects=True)
        if not response.ok:
            raise Exception('Not a valid url')

        extension = get_file_extension(image_url)
        self.temp_file = NamedTemporaryFile(dir='tmp', suffix='.' + extension)
        self.temp_file.write(response.content)

    def get_name(self) -> str:
        """ Gets the name of the temporary file
        :return: str
        """
        return self.temp_file.name


def get_file_extension(file_name) -> str:
    """Get the extension of a file name
    :param file_name: str
    :return: str
    """
    split_name = file_name.split('.')
    if len(split_name) < 2:
        return ''

    return split_name[-1]
