# SMERL
This repository contains code for [One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL](https://arxiv.org/abs/2010.14484). 

# Installation
1. In the rlkit/ directory, copy `config_template.py` to `config.py`:
```
cp rlkit/launchers/config_template.py rlkit/launchers/config.py
```
2. Install and use the included Ananconda environment
```
$ conda env create -f environment/[linux-cpu|linux-gpu|mac]-env.yml
$ source activate rlkit
```
Choose the appropriate `.yml` file for your system.
These Anaconda environments use MuJoCo 1.5 and gym 0.10.5.
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

DISCLAIMER: the mac environment has only been tested without a GPU.

For an even more portable solution, try using the docker image provided in `environment/docker`.
The Anaconda env should be enough, but this docker image addresses some of the rendering issues that may arise when using MuJoCo 1.5 and GPUs.
The docker image supports GPU, but it should work without a GPU.
To use a GPU with the image, you need to have [nvidia-docker installed](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

## Using a GPU
You can use a GPU by calling
```
import rlkit.torch.pytorch_util as ptu
ptu.set_gpu_mode(True)
```
before launching the scripts.

If you are using `doodad` (see below), simply use the `use_gpu` flag:
```
run_experiment(..., use_gpu=True)
```

# Training
 
### HalfCheetah-Goal

#### SAC:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --seed 2

#### DIAYN:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --seed 2

#### SAC+DIAYN:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env HalfCheetahGoalEnv-v1 --subopt-return-threshold -10000000.0 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env HalfCheetahGoalEnv-v1 --subopt-return-threshold -10000000.0 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env HalfCheetahGoalEnv-v1 --subopt-return-threshold -10000000.0 --seed 2

#### SMERL:
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --subopt-return-threshold -100.0 --seed 0
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --subopt-return-threshold -100.0 --seed 1
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env HalfCheetahGoalEnv-v1 --subopt-return-threshold -100.0 --seed 2


### Walker-Velocity

#### SAC:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --seed 2

#### DIAYN:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --seed 2

#### SAC+DIAYN:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env Walker2dVelocityEnv-v1 --subopt-return-threshold -10000000.0 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env Walker2dVelocityEnv-v1 --subopt-return-threshold -10000000.0 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env Walker2dVelocityEnv-v1 --subopt-return-threshold -10000000.0 --seed 2

#### SMERL:
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --subopt-return-threshold 790 --seed 0
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --subopt-return-threshold 790 --seed 1
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env Walker2dVelocityEnv-v1 --subopt-return-threshold 790 --seed 2


### Hopper-Velocity

#### SAC:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env HopperVelocityEnv-v1 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env HopperVelocityEnv-v1 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.0 --num_skills 5 --env HopperVelocityEnv-v1 --seed 2

#### DIAYN:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env HopperVelocityEnv-v1 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env HopperVelocityEnv-v1 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 0.0 --unsupervised-reward-weight 1.0 --num_skills 5 --env HopperVelocityEnv-v1 --seed 2

#### SAC+DIAYN:
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env HopperVelocityEnv-v1 --subopt-return-threshold -10000000.0 --seed 0
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env HopperVelocityEnv-v1 --subopt-return-threshold -10000000.0 --seed 1
python examples/url/run_smerl_experiment.py --algo diayn --environment-reward-weight 1.0 --unsupervised-reward-weight 0.5 --num_skills 5 --env HopperVelocityEnv-v1 --subopt-return-threshold -10000000.0 --seed 2

#### SMERL:
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env HopperVelocityEnv-v1 --subopt-return-threshold 600 --seed 0
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env HopperVelocityEnv-v1 --subopt-return-threshold 600 --seed 1
python examples/url/run_smerl_experiment.py --environment-reward-weight 1.0 --unsupervised-reward-weight 10.0 --num_skills 5 --env HopperVelocityEnv-v1 --subopt-return-threshold 600 --seed 2


# Evaluation

Plotting scripts for reproducing plots coming soon.
