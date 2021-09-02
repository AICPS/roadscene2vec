import os
import sys

import pytest

sys.path.append(os.path.dirname(sys.path[0]))
from util import config_parser


def get_parser(yml_path):
  return config_parser.configuration(yml_path)


resources = ['./resources/custom_hyperparams/learning_config.yaml', 
            './resources/custom_relations/scene_graph_config.yaml']

@pytest.mark.parametrize("yml_path", resources)
def test_parser(yml_path):
  # check if config includes at least 2 field
  keys = get_parser(yml_path).__dict__.keys()
  assert len(keys) > 1


  