#!/bin/bash
echo "Start Experiment: resnet18_MAE_detection"
python train_detection.py
echo "Start Experiment: resnext50_MAE_detection"
python train_detection_resnext50.py


