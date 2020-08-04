import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from lib.util import dump_yaml


class SimpleCNN(tf.keras.Model):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = layers.Conv2D(32, (3, 3),
                                    activation='relu', input_shape=(32, 32, 3))
        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.layer2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(99)

    def call(self, inputs):
        model = self.layer1(inputs)
        model = self.maxpool1(model)
        model = self.layer2(model)
        model = self.maxpool1(model)
        # model = self.layer2(model)
        # model = self.maxpool1(model)
        model = layers.Flatten()(model)
        model = self.dense1(model)
        model = self.dense2(model)
        return model
