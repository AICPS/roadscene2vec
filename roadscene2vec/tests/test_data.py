import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from util import config_parser
from data import dataset


def get_parser(yml_path):
  return config_parser.configuration(yml_path)

def load_carla_data(yml_path):
  return dataset.GroundTruthDataset(get_parser(yml_path))

def load_raw_data(yml_path):
  return dataset.RawImageDataset(get_parser(yml_path))

def check_empty(obj):
  for k, v in obj.__dict__.items():
    assert v is not None, f'{obj} with instance variable {k} cannot have a value of {v}'

def test_carla_data(yml_path='./resources/custom_datasets/data_config.yaml'):
  # example of an incorrect yaml file error message should be more verbose in the dataset class as to why dataset object errored!
  dataset = load_carla_data(yml_path)
  check_empty(dataset)