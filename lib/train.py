#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Functions related to model training."""

from lib.models import SimpleCNN
import tensorflow as tf
from lib.util import dump_yaml
import os
from pathlib import Path


def train(config, train_generator, val_generator=None):
    """Trains model given in configuration with data provided.

    Saves checkpoint and history as specified by config.
    If directory does not exist, creates directory.

    Parameters
    ----------
    config : dict
        Dictionary containing all parameters related to model.
    train_generator : Iterator
        Generator for training data.
    val_generator : Iterator
        Generator for validation data.

    Returns
    -------
    dict
        Returns tf2 generated history of training in dict format

    """
    model = eval(config['train']['model'])()
    log_dir = os.path.join(config['base']['root'],
                           config['train']['log_dir'])
    # Creates log directory if necessary
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # Compiles model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    history = model.fit(
        train_generator,
        verbose=1,
        steps_per_epoch=500,
        epochs=1,
        validation_steps=100,
        validation_data=val_generator)
    # Saves history as yaml file
    dump_yaml(history, os.path.join(log_dir, 'history.yaml'))
    return history
