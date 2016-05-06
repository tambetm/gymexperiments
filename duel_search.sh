#!/usr/bin/env bash

python duel.py --gym_record ${1}_reference $*
python duel.py --advantage max --gym_record ${1}_advmax $*
python duel.py --advantage avg --gym_record ${1}_advavg $*
python duel.py --gamma 0.995 --gym_record ${1}_gamma0.995 $*
python duel.py --tau 0.01 --gym_record ${1}_tau0.01 $*
python duel.py --activation relu --gym_record ${1}_relu $*
python duel.py --batch_norm --gym_record ${1}_batchnorm $*
python duel.py --exploration 0.05 --gym_record ${1}_exploration0.05 $*
python duel.py --batch_size 1000 --gym_record ${1}_batchsize1000 $*
python duel.py --hidden_size 200 --gym_record ${1}_hidden200 $*
python duel.py --layers 2 --gym_record ${1}_layers2 $*
