#!/bin/env bash

SSH_ARGS=$1

apt update
# apt install 
cd $HOME

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

source $HOME/miniconda3/bin/activate
conda create -n mycondaenv python=3.12
conda activate mycondaenv

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/mideind/lestrarflaekja
cd lestrarflaekja
pip3 install -r requirements.txt

python3 prepare_data.py $DATA_PREP_ARGS
python3 train.py

# # move to remote machine
# scp myscript.sh $ADDR:~/
# # execute on remote machine
# ssh -t $COMMAND $COMMAND_ARGS
