#!/bin/env bash

# apt update

WORKSPACE=/workspace
cd $WORKSPACE

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh -b -p $WORKSPACE/miniconda3
# ./Miniconda3-latest-Linux-x86_64.sh -p $WORKSPACE/miniconda3

source $WORKSPACE/miniconda3/bin/activate

export TMP=$WORKSPACE/tmp.tmp
export TEMP=$WORKSPACE/tmp.temp
export PIP_CACHE_DIR=$WORKSPACE/tmp.pip
export XDG_CACHE_HOME=$WORKSPACE/tmp.xdg
export PIP_NO_USER=true
export PIP_USER=false
export CONDA_PKGS_DIRS=/workspace/conda-pkgs
export PYTHONNOUSERSITE=1
mkdir -p $TMP $TEMP $PIP_CACHE_DIR

conda config --set plugins.auto_accept_tos yes
conda create -y --name myenv2 python=3.12

conda activate myenv2

# conda info
# conda config --set env_dirs $WORKSPACE/miniconda3/envs

pip install --prefer-binary --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/mideind/lestrarflaekja
cd lestrarflaekja
pip3 install -r requirements.txt

# python3 prepare_data.py $DATA_PREP_ARGS
# python3 train.py

# SSH_ARGS=$1

# # move to remote machine
# scp myscript.sh $ADDR:~/
# # execute on remote machine
# ssh -t $COMMAND $COMMAND_ARGS
