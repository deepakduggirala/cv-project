#!/bin/bash

python train.py --epochs=1 \
                --epochs2=1 \
                --save_freq=1 \
                --params=hyperparameters/inception.json \
                --data_dir=/Users/deepakduggirala/Documents/ELPephant-cropped \
                --additional_data_dir=/Users/deepakduggirala/Documents/Elephants-dataset-cropped-png-1024
