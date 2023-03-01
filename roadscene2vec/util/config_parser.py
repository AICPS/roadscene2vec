import sys

from argparse import ArgumentParser
import yaml
from pathlib import Path

class configuration:
    def __init__(self, args, from_function = False):
        if type(args) != list:
            from_function = True
        if not(from_function):
            ap = ArgumentParser(description='The parameters for use-case 2.')
            ap.add_argument('--yaml_path', type=str, default="./IP-NetList.yaml", help="The path of yaml config file.")
            ap.add_argument('--model', type=str, default = None, help="model override.")
            ap.add_argument('--input_path', type=str, default = None, help="input_path override.")
            ap.add_argument('--task_type', type=str, default = None, help="task type override.")
            args_parsed = ap.parse_args(args)
            for arg_name in vars(args_parsed):
                self.__dict__[arg_name] = getattr(args_parsed, arg_name)
                self.yaml_path = Path(self.yaml_path).resolve()
            #handle command line overrides.
            if self.model != None:
                self.model_configuration['model'] = self.model
            if self.input_path != None:
                self.location_data['input_path'] = self.input_path
            if self.task_type != None:
                self.training_configuration['task_type'] = self.task_type
                    
        
        if from_function:
            self.yaml_path = Path(args).resolve()
        with open(self.yaml_path, 'r') as f:
            self.args = yaml.safe_load(f)
            for arg_name, arg_value in self.args.items():
                self.__dict__[arg_name] = arg_value


        
    @staticmethod
    def parse_args(yaml_path):
        return configuration(yaml_path,True)




if __name__ == "__main__":
    configuration(sys.argv[1:])