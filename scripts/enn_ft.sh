save_dir_root=./ckpts
output_dir=./ckpts/enn_ft
criterion=wrong2correct


for dataset in cora amazoncomputers amazonphoto coauthorcs arxiv; do
for model in gcn sage gat gin; do
    if ! [ -d "./${output_dir}/${dataset}" ]; then
        mkdir -p "./${output_dir}/${dataset}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./enn_ft.py \
        --config ./config/${model}.yaml \
        --alg_config ./alg_config/enn.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} \
        --saved_model_path ${save_dir_root}/${dataset} 2>&1 | tee ${output_dir}/${dataset}/${model}.log
done
done
