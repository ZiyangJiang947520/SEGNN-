save_dir_root=./ckpts
output_dir=./finetune_abl_samples
criterion=wrong2correct


# for model in mlp; do
# for dataset in cora flickr reddit2 arxiv; do
#     if ! [ -d "./${output_dir}/${dataset}" ]; then
#         mkdir -p "./${output_dir}/${dataset}"
#     fi
#     CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
#         --config ./config/${model}.yaml \
#         --dataset ${dataset} \
#         --output_dir ${output_dir} \
#         --saved_model_path ${save_dir_root}/${dataset} \
#         --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${model}_${criterion}_eval.log
# done
# done

## cora flickr reddit2 arxiv amazoncomputers amazonphoto coauthorcs coauthorphysics yelp products

for manner in GD; do    ### GD GD_Diff Ada_GD_Diff
for dataset in cora amazoncomputers amazonphoto coauthorcs flickr reddit2; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto wikics yelp products
for samples in 5 10 15 20 30 40; do
for model in  gcn_mlp sage_mlp; do ###gcn sage mlp gcn_mlp sage_mlp
    if ! [ -d "./${output_dir}/${samples}/${dataset}" ]; then
        mkdir -p "./${output_dir}/${samples}/${dataset}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir}/${samples} \
        --saved_model_path ${save_dir_root}/${dataset} \
        --manner ${manner} \
        --num_samples ${samples} \
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${samples}/${dataset}/${model}_${criterion}_eval.log
done
done
wait
done
done