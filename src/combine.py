#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combines workout files into one.

Author:   
    Erik Johannes Husom

Created:  
    2020-10-29

"""
import os
import sys

import numpy as np
from codecarbon import track_emissions

from config import config
from pipelinestage import PipelineStage
from preprocess_utils import find_files


@track_emissions(project_name="combine")
class CombineStage(PipelineStage):
    """Combine data from multiple input files into one dataset.

    Args:
        dir_path (str): Path to directory containing files.

    """

    def __init__(self):
        super().__init__(stage_name="combine")

    def run(self):

        filepaths = find_files(config.DATA_SEQUENTIALIZED_PATH, file_extension=".npz")

        train_inputs = []
        train_outputs = []
        test_inputs = []
        test_outputs = []

        for filepath in filepaths:
            infile = np.load(filepath)

            X = infile["X"]
            y = infile["y"]

            if X.size == 0 or y.size == 0:
                print(f"Skipped {filepath} because it is emtpy")
                continue

            if "train" in filepath:
                train_inputs.append(X)
                train_outputs.append(y)
            elif "test" in filepath:
                test_inputs.append(X)
                test_outputs.append(y)

        X_train = np.concatenate(train_inputs)
        y_train = np.concatenate(train_outputs)
        X_test = np.concatenate(test_inputs)
        y_test = np.concatenate(test_outputs)

        np.savez(config.DATA_COMBINED_PATH / "train.npz", X=X_train, y=y_train)
        np.savez(config.DATA_COMBINED_PATH / "test.npz", X=X_test, y=y_test)

def main():
    CombineStage().run()

if __name__ == "__main__":
    main()
