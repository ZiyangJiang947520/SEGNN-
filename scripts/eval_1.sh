save_dir_root=./ckpts/enn_ft
output_dir=./finetune_new_std/enn_ft
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
for dataset in products; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto wikics yelp products
for model in  gin; do ###gcn sage mlp gcn_mlp sage_mlp
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
        --stop_full_edit True \
        --half_half False \
        --half_half_ratio_mixup 0.75 \
        --iters_before_stop 0 \
        --full_edit 0 \
        --mixup_k_nearest_neighbors False \
        --incremental_batching True \
        --sliding_batching 0 \
        --pure_egnn 0 \
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${manner}/${model}_${criterion}_eval.log
done
wait
done
done
