#!/usr/bin/env bash

python naf.py --gym_record ${1}_reference $*
#python naf.py --batch_norm --gym_record ${1}_batchnorm $*
python naf.py --optimizer_lr 0.01 --gym_record ${1}_lr0.01 $*
python naf.py --optimizer_lr 0.0001 --gym_record ${1}_lr0.0001 $*
python naf.py --noise exp_decay --gym_record ${1}_noise_exp $*
python naf.py --noise fixed --noise_scale 0.1 --gym_record ${1}_noise_fixed0.1 $*
python naf.py --noise fixed --noise_scale 0.01 --gym_record ${1}_noise_fixed0.01 $*
python naf.py --noise covariance --noise_scale 0.001 --gym_record ${1}_noise_cov $*
#python naf.py --batch_size 1000 --gym_record ${1}_batchsize1000 $*
python naf.py --tau 1 --gym_record ${1}_tau1 $*
python naf.py --tau 0.1 --gym_record ${1}_tau0.1 $*
python naf.py --tau 0.01 --gym_record ${1}_tau0.01 $*
python naf.py --activation relu --gym_record ${1}_relu $*
python naf.py --layers 1 --gym_record ${1}_layers1 $*
python naf.py --hidden_size 50 --gym_record ${1}_hidden50 $*
python naf.py --gamma 0.995 --gym_record ${1}_gamma0.995 $*
python naf.py --unit_norm --gym_record ${1}_unitnorm $*
python naf.py --max_norm 1 --gym_record ${1}_maxnorm1 $*
python naf.py --max_norm 5 --gym_record ${1}_maxnorm5 $*
