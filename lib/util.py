#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Contains variety of utility function not specific to tensorflow."""
import yaml


def load_yaml(file_path):
    """Python configuration file.

    Parameters
    ----------
    file_path : str
        Configuration file loaction

    Returns
    -------
    Dict
        Dictionary containing all parameters describing an experiment

    """
    with open(file_path, 'r') as f:
        config = yaml.load(f)
    return config


def dump_yaml(obj, file_path):
    """Python configuration file.

    Parameters
    ----------
    file_path : str
        Configuration file loaction

    Returns : None

    """
    with open(file_path, '+w') as f:
        yaml.dump(obj, f)
