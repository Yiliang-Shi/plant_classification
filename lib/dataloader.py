#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Contains functions related to dataloading.

Each dataset has a specific function, which returns a tf2 generator
"""

import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class PlantData:
    """Short summary.

    Attributes
    ----------
    train_data : type
        Description of attribute `train_data`.

    """

    def __init__(self, config):
        """Initialize generator for all image data.

        Returns None
        """
        self.image_generator = ImageDataGenerator(
            rotation_range=45,
            rescale=1./255,
            validation_split=0.8
        )
        self.config = config

    def get_train_data(self):
        """Training set from generator.

        Returns Iterator
        -------
        type
            Iterator containing trainingset

        """
        data_file = os.path.join(self.config['base']['root'],
                                 self.config['data']['annotations'])
        image_dir = os.path.join(self.config['base']['root'],
                                 self.config['data']['img_folder'])
        batch_size = 10
        random_seed = 42
        train = pd.read_csv(data_file)
        train['id'] = train['id'].astype(str)+'.jpg'
        train_data = self.image_generator.flow_from_dataframe(
            dataframe=train, subset='training',
            directory=image_dir, x_col='id',
            y_col='species', seed=random_seed,
            batch_size=batch_size, shuffle=True,
            class_mode="categorical", target_size=(32, 32)
        )
        return train_data

    def get_val_data(self):
        """Val set from generator.

        Returns Iterator
        -------
        type
            Iterator containing trainingset

        """
        data_file = os.path.join(self.config['base']['root'],
                                 self.config['data']['annotations'])
        image_dir = os.path.join(self.config['base']['root'],
                                 self.config['data']['img_folder'])
        batch_size = 10
        random_seed = 42
        val = pd.read_csv(data_file)
        val['id'] = val['id'].astype(str)+'.jpg'
        val_data = self.image_generator.flow_from_dataframe(
            dataframe=val, subset='validation',
            directory=image_dir, x_col='id',
            y_col='species', seed=random_seed,
            batch_size=batch_size, shuffle=True,
            class_mode="categorical", target_size=(32, 32)
        )
        return val_data
