save_dir_root=./ckpts
###
 # @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @Date: 2023-02-01 16:15:09
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2023-02-06 20:27:44
 # @FilePath: /edit_gnn/scripts/eval.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
output_dir=./finetune
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
# for manner in GD_Diff Ada_GD_Diff; do    ### GD GD_Diff Ada_GD_Diff
manner=GD_Diff
for hyper_Diff in 0. 0.1 1.0 10.0 50.0; do
for dataset in yelp products; do ### cora flickr reddit2 arxiv
for model in gcn sage mlp gcn_mlp sage_mlp; do ###gcn sage mlp gcn_mlp sage_mlp
    if ! [ -d "./${output_dir}/${dataset}/${manner}" ]; then
        mkdir -p "./${output_dir}/${dataset}/${manner}"
    fi
    CUDA_VISIBLE_DEVICES=$1 python ./eval.py \
        --config ./config/${model}.yaml \
        --dataset ${dataset} \
        --output_dir ${output_dir} \
        --saved_model_path ${save_dir_root}/${dataset} \
        --manner ${manner} \
        --hyper_Diff ${hyper_Diff} \
        --criterion ${criterion} 2>&1 | tee ${output_dir}/${dataset}/${manner}/${model}_${criterion}_eval_hyper_Diff=${hyper_Diff}.log
done
wait
done
done