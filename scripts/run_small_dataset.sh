
output_dir=./ckpts
# output_dir=./rebuttal/ckpts

# flickr amazoncomputers amazonphoto coauthorcs
for dataset in cora; do  ##cora flickr reddit2 amazoncomputers amazonphoto coauthorcs coauthorphysics yelp arxiv products
    for model in gcn_mlp; do  ## gcn sage mlp gcn_mlp sage_mlp sign_mlp gat gat_mlp gin gin_mlp
    if ! [ -d "./${output_dir}/${dataset}" ]; then
        mkdir -p "./${output_dir}/${dataset}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./train.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --use_betweenness_centrality True\
        --use_closeness_centrality False\
        --use_eigenvector_centrality False\
        --output_dir ${output_dir} 2>&1 | tee ${output_dir}/${dataset}/${model}.log


done
wait
done
# python ./train.py --config ./config/gcn.yaml --dataset flickr --output_dir ./ckpts
