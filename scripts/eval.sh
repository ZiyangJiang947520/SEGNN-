save_dir_root=./ckpts
output_dir=./finetune_new_std
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
for dataset in arxiv cora amazoncomputers amazonphoto coauthorcs; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto wikics yelp products
# for dataset in cora; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto wikics yelp products
for model in  gcn sage gcn_mlp sage_mlp; do ###gcn sage mlp gcn_mlp sage_mlp
    if ! [ -d "./${output_dir}/${dataset}/${manner}" ]; then
        mkdir -p "./${output_dir}/${dataset}/${manner}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} \
        --saved_model_path ${save_dir_root}/${dataset} \
        --manner ${manner} \
        --num_mixup_training_samples 100 \
        --finetune_between_edit False \
        --stop_edit_only False \
        --stop_full_edit False \
        --half_half False \
        --iters_before_stop 0 \
        --full_edit False \
        --mixup_k_nearest_neighbors False \
        --incremental_batching False \
        --sliding_batching 0 \
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${manner}/${model}_${criterion}_eval.log
done
done
done
