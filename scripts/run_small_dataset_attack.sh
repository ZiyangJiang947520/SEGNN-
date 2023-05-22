
output_dir=attack/ckpts/ablations
# output_dir=./ckpts

for dataset in cora; do  ##cora flickr reddit2 amazoncomputers amazonphoto coauthorcs coauthorphysics yelp arxiv products
for model in gcn gcn_mlp; do  ## gcn sage mlp gcn_mlp sage_mlp
    if ! [ -d "./${output_dir}/${dataset}" ]; then
        mkdir -p "./${output_dir}/${dataset}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./train.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --attack \
        --attack_ratio 0.5 \
        --attack_class 0 \
        --output_dir ${output_dir} 2>&1 | tee ${output_dir}/${dataset}/${model}.log
done
wait
done
# python ./train.py --config ./config/gcn.yaml --dataset flickr --output_dir ./ckpts