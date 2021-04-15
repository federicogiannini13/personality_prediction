import os
import yaml
from settings import ROOT_DIR


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_dir_file(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_yaml_config(config, path):
    if os.path.exists(path):
        with open(path) as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        for k in config_yaml.keys():
            setattr(config, k, config_yaml[k])
    with open(path, "w") as f:
        yaml.dump(config.__dict__, f)
    if "OUTPUTS_DIR" in config.__dict__.keys():
        if config.OUTPUTS_DIR is None or config.OUTPUTS_DIR == "":
            config.OUTPUTS_DIR = ROOT_DIR
    return config


class Config:
    pass
