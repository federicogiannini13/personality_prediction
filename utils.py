import os


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_dir_file(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
