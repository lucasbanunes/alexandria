import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from ..utils import makedir

def build_model(model_name, layers_config):

    layers = dict()
    model_inputs = list()
    for layer_class, layer_info in layers_config.items():
        layer = getattr(keras.layers, layer_class).from_config(layer_info['layer_config'])
        layers[layer.name] = {
            'layer': layer,
            'connects_to': layer_info['connects_to'],
        }
        if layer_class == 'InputLayer':
            model_inputs.append(layer)
    
    model_outputs = list()
    for layer_name, layer_info in layers.items():
        if len(layer_info['connects_to']) == 0:
            model_outputs.append(layer_info['layer'])
        else:
            for layer_name in layer_info['connects_to']:
                layers[layer_name]['layer'](layer_info['layer'])
    
    if len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    
    if len(model_outputs) == 1:
        model_outputs = model_outputs[0]

    model = keras.Model(model_inputs, model_outputs, name=model_name)
    return model

class LayersConfigs(object):

    def __init__(self, srcdir):

        self.srcdir = srcdir
        self.configs = {config_file[:-5]: None for config_file in os.listdir(srcdir)
                                if config_file.endswith('.json')}
    
    def __getitem__(self, key):

        if not key in self.configs.keys():
            raise KeyError(f'There is no template for {key}')
        
        if self.configs[key] is None:
            with open(os.path.join(self.srcdir, f'{key}.json'), 'r') as json_file:
                self.configs[key] = json.load(json_file)
        
        return self.configs[key]