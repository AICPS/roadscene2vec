import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from util import config_parser


# TODO: save version of each model and upload them to resources/pretrained_models with appropriate names
def get_parser(yml_path):
  return config_parser.configuration(yml_path)

def load_model():
  pass

def test_model():
  pass