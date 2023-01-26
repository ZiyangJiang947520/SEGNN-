# model=sage
model=gcn
dataset=cora
output_dir=./ckpts

CUDA_VISIBLE_DEVICES=$1 python ./train.py \
    --config ./config/${model}.yaml \
    --dataset ${dataset} \
    --output_dir ${output_dir}