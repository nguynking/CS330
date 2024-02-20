#!/bin/bash

ARG_DEVICE=--device=${1:-'cpu'}
ARG_BATCH_SIZE=--batch_size=${2:-16}
ARG_COMPILE=--${3:-'no-compile'}
ARG_BACKEND=--backend=${4:-'inductor'}

# Protonet experiment
python protonet.py $ARG_DEVICE $ARG_BATCH_SIZE $ARG_COMPILE $ARG_BACKEND

# Maml experiments
# Experiment 1
python maml.py $ARG_DEVICE $ARG_BATCH_SIZE

# Experiment 2
python maml.py --inner_lr 0.04 $ARG_DEVICE $ARG_BATCH_SIZE

# Experiment 3
python maml.py --inner_lr 0.04 --num_inner_steps 5 $ARG_DEVICE $ARG_BATCH_SIZE

# Experiment 4
python maml.py --learn_inner_lrs $ARG_DEVICE $ARG_BATCH_SIZE