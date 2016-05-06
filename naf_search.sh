#!/usr/bin/env bash

python naf.py --gym_record ${1}_reference $*
python naf.py --noise exp_decay --gym_record ${1}_noise_exp $*
python naf.py --noise fixed --fixed_noise 0.1 --gym_record ${1}_noise_fixed0.1 $*
python naf.py --gamma 0.99 --gym_record ${1}_gamma0.99 $*
python naf.py --tau 0.01 --gym_record ${1}_tau0.01 $*
python naf.py --activation relu --gym_record ${1}_relu $*
python naf.py --batch_norm --gym_record ${1}_batchnorm $*
python naf.py --batch_size 1000 --gym_record ${1}_batchsize1000 $*
python naf.py --hidden_size 200 --gym_record ${1}_hidden200 $*
python naf.py --layers 2 --gym_record ${1}_layers2 $*
