import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), "config"))
from argparse import ArgumentParser
import yaml
from pathlib import Path
import pickle 


class configuration:
    def __init__(self, args, from_function = False):
        if type(args) != list:
            from_function = True
        if not(from_function):
            ap = ArgumentParser(description='The parameters for use-case 2.')
            ap.add_argument('--yaml_path', type=str, default="./IP-NetList.yaml", help="The path of yaml config file.")
            args_parsed = ap.parse_args(args)
            for arg_name in vars(args_parsed):
                self.__dict__[arg_name] = getattr(args_parsed, arg_name)
                self.yaml_path = Path(self.yaml_path).resolve()
        
        if from_function:
            self.yaml_path = Path(args).resolve()
        with open(self.yaml_path, 'r') as f:
            yaml_configs = yaml.safe_load(f)
            self.args = yaml_configs
            for arg_name, arg_value in yaml_configs.items():
                self.__dict__[arg_name] = arg_value
    @staticmethod
    def parse_args(yaml_path):
        return configuration(yaml_path,True)




if __name__ == "__main__":
    configuration(sys.argv[1:])