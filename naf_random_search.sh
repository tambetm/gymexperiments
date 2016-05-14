#!/usr/bin/env bash

python random_search.py python naf.py $* \
  --batch_norm,--no_batch_norm \
  --optimizer_lr 0.01,0.001,0.0001 \
  --noise linear_decay,exp_decay,fixed,covariance \
  --noise_scale 0.1,0.01,0.001 \
  --batch_size 100,200,1000 \
  --tau 1,0.1,0.01,0.001 \
  --activation tanh,relu \
  --layers 0,1,2 \
  --hidden_size 50,100,200 \
  --gamma 0.9,0.99,0.995 \
  "--unit_norm,--max_norm 1,--max_norm 5," \
  --l2_reg 0.001,0.0001,0.00001 \
  --train_repeat 1,5,10
