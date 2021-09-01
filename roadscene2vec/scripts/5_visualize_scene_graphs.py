import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from util.config_parser import configuration
from util.visualizer import visualize


if __name__ == '__main__':
  extraction_config = configuration(sys.argv[1:])
  visualize(extraction_config)
  