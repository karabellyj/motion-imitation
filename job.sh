#!/bin/sh
#
#$ -S /bin/bash
#$ -N MotionImitation
#$ -o /homes/eva/xkarab03/$JOB_NAME.$JOB_ID.out
#$ -e /homes/eva/xkarab03/$JOB_NAME.$JOB_ID.err
#$ -q all.q@@servers,all.q@@blade,all.q@@stable
#
# PE_name    CPU_Numbers_requested
#$ -pe smp  32
#


# git clone ?

#wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
#tar -xzvf mujoco210-linux-x86_64.tar.gz
#export MUJOCO_PY_MUJOCO_PATH=./mujoco210

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run.py --save_path=run --n_envs=$NSLOTS