save_dir_root=./attack/ckpts
output_dir=./finetune_attack_new/
attack_class=0



for manner in GD; do    ### GD GD_Diff Ada_GD_Diff
for dataset in cora; do ### cora flickr reddit2 arxiv amazoncomputers amazonphoto wikics yelp products
for model in gcn_mlp; do ###gcn sage mlp gcn_mlp sage_mlp
    if ! [ -d "./${output_dir}/${dataset}/${manner}" ]; then
        mkdir -p "./${output_dir}/${dataset}/${manner}"
    fi

    MODEL=$(echo "${model}" | tr '[:lower:]' '[:upper:]')

    CUDA_VISIBLE_DEVICES=$1 python ./eval_attack.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} \
        --saved_model_path ${save_dir_root}/${dataset} \
        --manner ${manner} \
        --attack_indices_path ./attack/ckpts/${dataset}/${MODEL}_attack_indices.npy \
        --attack_class ${attack_class} \
        --num_samples 50 2>&1 # | tee ${output_dir}/${dataset}/${manner}/${model}_eval.log
done
wait
done
done