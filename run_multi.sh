#!/usr/bin/env bash
set -e

# MNIST fast, default epochs/batch
python run_experiment.py --dataset mnist --fast "$@"

# CIFAR10 full with overridden epochs/batch
python run_experiment.py --dataset cifar10 --epochs 40 --batch 256 "$@"

# CIFAR100 full with custom epochs/batch and only bottleneck/deep scenarios
python run_experiment.py --dataset cifar100 --epochs 60 --batch 192 --scenarios bottleneck deep "$@"
