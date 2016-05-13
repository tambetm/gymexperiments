# OpenAI Gym experiments

My implementations of [normalized advantage functions](http://arxiv.org/abs/1603.00748) (NAF) for continuous actions spaces and [dueling network architecture](http://arxiv.org/abs/1511.06581) (DUEL) for discrete action spaces. 

Example results with NAF:
 * [InvertedPendulum-v1](https://gym.openai.com/evaluations/eval_CzoNQdPSAm0J3ikTBSTCg)
 * [Pendulum-v0](https://gym.openai.com/evaluations/eval_IU3wehAQQRuJRbzMjy26QQ)
 
Example results with DUEL:
 * [CartPole-v0](https://gym.openai.com/evaluations/eval_sOUmkzSy26GIWJ5IIQeA)
 * [MountainCar-v0](https://gym.openai.com/evaluations/eval_nAU6XkQhSuKrVNNZdQ5xQ)
 * [Acrobot-v0](https://gym.openai.com/evaluations/eval_TkAOrmYAQ9eoiTujoCgw)

## Prerequisites

You will need:
 * Python 2.7
 * [OpenAI Gym](https://gym.openai.com/)
 * [Keras](http://keras.io/)
 * [Numpy](http://www.numpy.org/)
 * [Scikit-Learn](http://scikit-learn.org) (if using imagination rollouts)
 
In Ubuntu that would be:

```
sudo apt-get install python-numpy python-sklearn
pip install --user gym keras
```

If you want to run Mujoco environments, you also need to acquire [trial key](https://www.roboti.us/trybuy.html) and [install the binaries](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key). Then you can install Mujoco support for OpenAI Gym:

```
pip install --user gym[mujoco]
```

## Running the code

There are three main starting points:
 * `python duel.py <envid>` - run DUEL against environment with discrete action space,
 * `python naf.py <envid>` - run NAF against environment with continuous action space,
 * `python nag_ir.py <envid>` - run NAF with imagination rollouts.

You can override default hyperparameters with command-line options, use `-h` to see them or check out the code.

Some other utility scipts:
 * `python test.py <envid>` - test script to run random actions against the environment,
 * `python naf_search.sh` - example how to run crude hyperparameter search for NAF,
 * `python duel_search.sh` - example how to run crude hyperparameter search for DUEL.
