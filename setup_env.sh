#!/bin/env bash

apt update
# apt install 

WORKSPACE=/workspace
cd $WORKSPACE

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh -b -p $WORKSPACE/miniconda3

source $WORKSPACE/miniconda3/bin/activate

conda config --set plugins.auto_accept_tos yes
conda create -y -n mycondaenv python=3.13

conda activate mycondaenv

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/mideind/lestrarflaekja
cd lestrarflaekja
pip3 install -r requirements.txt

python3 prepare_data.py $DATA_PREP_ARGS
python3 train.py

# SSH_ARGS=$1

# # move to remote machine
# scp myscript.sh $ADDR:~/
# # execute on remote machine
# ssh -t $COMMAND $COMMAND_ARGS
