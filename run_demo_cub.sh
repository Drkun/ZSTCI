#!/usr/bin/env bash

DATASET=cub
GPU=1
NET=resnet18
LR=1e-5
EPOCH=51
SAVE=50
LOSS=triplet #triplet
TASK=10 #6
BASE=20 #17
SEED=1993

for Method in Finetuning #Finetuning LwF EWC MAS
do
for Tradeoff in 0 # 0 1 1e7 1e6
do

NAME=${Method}_${Tradeoff}_${DATASET}_${LOSS}_${NET}_${LR}_${EPOCH}epochs_task${TASK}_base${BASE}_seed${SEED}_base${BASE}

python train.py -base ${BASE} -seed ${SEED} -task ${TASK} -data ${DATASET} -tradeoff ${Tradeoff} -exp ${Tradeoff} -net ${NET} -method ${Method} \
-lr ${LR} -dim 512  -num_instances 8 -BatchSize 32 -loss ${LOSS}  -epochs ${EPOCH} -log_dir ${DATASET}_seed${SEED}_final/${NAME}  \
-save_step ${SAVE} -gpu ${GPU}


python test.py  -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU}  -log_dir ${DATASET}_seed${SEED}_final/${NAME}


python test_SDC.py -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU} -log_dir ${DATASET}_seed${SEED}_final/${NAME}

python test_new.py -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU}  -log_dir ${DATASET}_seed${SEED}_final/${NAME}



done
done

