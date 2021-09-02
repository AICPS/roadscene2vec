import sys
from pathlib import Path
sys.path.append(str(Path("../../")))
from abc import ABC

'''Abstract base class used to create CarlaPreprocessor and RealPreprocessor'''
class Preprocessor(ABC):
    def __init__(self, config):
        self.conf = config
        self.dataset = None

