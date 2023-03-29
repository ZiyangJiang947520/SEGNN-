
output_dir=./ckpts
# for model in gcn sage; do
# for dataset in cora citeseer pubmed; do
#     if ! [ -d "./${output_dir}/${dataset}" ]; then
#         mkdir "./${output_dir}/${dataset}"
#     fi
#     CUDA_VISIBLE_DEVICES=$1 python ./train.py \
#         --config ./config/${model}.yaml \
#         --dataset ${dataset} \
#         --output_dir ${output_dir} 2>&1 | tee ${output_dir}/${dataset}/${model}.log
# done
# done

for dataset in yelp; do  ##cora flickr reddit2 amazoncomputers amazonphoto coauthorcs coauthorphysics yelp arxiv products
for model in sage; do  ## gcn sage mlp gcn_mlp sage_mlp
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