
output_dir=./ckpts
# output_dir=./rebuttal/ckpts

# flickr amazoncomputers amazonphoto coauthorcs
for dataset in flickr amazoncomputers amazonphoto coauthorcs; do  ##cora flickr reddit2 amazoncomputers amazonphoto coauthorcs coauthorphysics yelp arxiv products
    for model in sage_mlp; do  ## gcn sage mlp gcn_mlp sage_mlp sign_mlp
    if ! [ -d "./${output_dir}/${dataset}" ]; then
        mkdir -p "./${output_dir}/${dataset}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./train.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} 2>&1 | tee ${output_dir}/${dataset}/${model}.log
done
wait
done
# python ./train.py --config ./config/gcn.yaml --dataset flickr --output_dir ./ckpts
