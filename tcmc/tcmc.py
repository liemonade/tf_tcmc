from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras




class MarkovTransition(layers.Layer):

  def __init__(self, output_shape, **kwargs):
    self.output_shape = output_shape
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    

  def call(self, inputs):
    # evaluate

  def get_config(self):
    base_config = super(MarkovTransition, self).get_config()
    base_config['output_shape'] = self.output_shape
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)