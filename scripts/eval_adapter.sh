save_dir_root=./rebuttal/ckpts
output_dir=./rebuttal/adapter_w_finetune
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
for dataset in cora flickr amazoncomputers amazonphoto coauthorcs; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto wikics yelp products
# for dataset in cora; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto wikics yelp products
for model in  gcn; do ###gcn sage mlp gcn_mlp sage_mlp
    if ! [ -d "./${output_dir}/${dataset}/${manner}" ]; then
        mkdir -p "./${output_dir}/${dataset}/${manner}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./eval_adapter.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} \
        --saved_model_path ${save_dir_root}/${dataset} \
        --manner ${manner} \
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${manner}/${model}_${criterion}_eval.log
done
done
done