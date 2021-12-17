import importlib
import os
import torch.nn as nn

MODEL_REGISTRY = {}

def build_model(args):
    return MODEL_REGISTRY[args.model].build_model(args)


def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model {}'.format(name))
        if not issubclass(cls, nn.Module):
            raise ValueError('Model {} must extend {}'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file[0].isalpha():
        module = file[:file.find('.py')]
        importlib.import_module('models.' + module)












