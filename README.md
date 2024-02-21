# How to reproduce the experiments results

* Run ```pip install -e .``` at the proj root directory

**Warning** : some depencies will raise error so we need to install these dependencies first before proceeding with ```pip install -e .```. We also need to determine torch version that is compatible with your GPU's CUDA version (commented out line in requirements.txt). 

* Then, train a model under inductive setting. 
```
bash scripts/run_small_dataset.sh $GPU_NUMBER
```

GPU_NUMBER is the index of GPU to use. For example, ```bash scripts/run_small_dataset.sh 0``` means using GPU  to train the models.

* Then, perform the edit on pretrained model.

```
bash scripts/eval.sh $GPU_NUMBER
bash scripts/eval_edg.sh $GPU_NUMBER
bash scripts/eval_edg_plus.sh $GPU_NUMBER
```
Here is the brief descriptions of all the flags for editing the pretrained models. They are not very loosely coupled so we need to carefully turn them to avoid any unexpected errors:

```
        --num_mixup_training_samples 100 \
        --finetune_between_edit False \
        --stop_edit_only False \
        --stop_full_edit True \
        --half_half True \
        --half_half_ratio_mixup 0.5 \
        --iters_before_stop 0 \
        --full_edit 0 \
        --mixup_k_nearest_neighbors True \
        --incremental_batching True \
        --sliding_batching 0 \
        --pure_egnn 0 \
        --wrong_ratio_mixup 0.5 \
```
* ```--num_mixup_training_samples```: this flag is used to indicate how many mixup samples we would like to have in the editing batch. 
* ```--finetune_between_edit```: this flag is used to indicate if we would like to finetune the MLP after editing one target.
* ```--stop_edit_only```: this flag is used to indicate if we would like to stop the editing process immediately when the current editing target is corrected, regardless of other samples in the editing batch. If this is true, ```--stop_full_edit``` needs to be false.
* ```--stop_full_edit```: this flag is used to indicate if we would like to stop the editing process immediately when **all** editing targets in the batch are corrected, regardless of other samples in the editing batch. If this is true, ```--stop_edit_only``` needs to be false.
* If both ```--stop_edit_only``` and ```--stop_full_edit``` are false, the model use ```batch_stop``` setting, i.e. the editing process only stops if all samples in the batch are predicted correctly.
* ```--half_half```: this flag is used to indicate if we would like to mix any nearest neighbors (NN) into the mixup training samples. If this is true, the ```--half_half_ratio_mixup``` flag is used to indicate the fraction of NN in the mixup samples in the batch. ```--mixup_k_nearest_neighbors``` also needs to be true in order to mixup random training samples + NN. If this is false, and ```--mixup_k_nearest_neighbors``` is true, only NN are used as mixup samples. If this is false and ```--mixup_k_nearest_neighbors``` is false, only random samples are used as mixup samples.
* ```--iters_before_stop k```: this flag is used to indicate if we would like to run k more iterations after the editing process is stopped.
* ```--full_edit k```: this flag is used to indicate if we would like to update both the GNN & MLP during the first k editing steps. By default, it is 0, meaning only the MLP is updated.
* ```--incremental_batching```: this flag is used to indicate if we would like to batch all editing targets from 1 to i when we are editing target i. If this is false, we are using ```sequential_edit``` setting, i.e. only the current editing target is in the batch (+ mixup if there is any).
* ```--sliding_batching```: this flag is used if we would like to have sliding window over which editing targets to batch.
* ```--wrong_ratio_mixup```: this flaf is used to indicate if we would like to also mixup wrong NN in the batch.

**Note**: the editing result would be "batch_edit" in the output json files.

For example: if we would like to edit the pretrained model under SEGNN setting (Incremental Batch Edit + Training Data Mixup (25 NN /75 training) + All Edit Stop), we would turn the flag as follows:
```
        --num_mixup_training_samples 100 \
        --finetune_between_edit False \
        --stop_edit_only False \
        --stop_full_edit True \
        --half_half True \
        --half_half_ratio_mixup 0.25 \
        --iters_before_stop 0 \
        --full_edit 0 \
        --mixup_k_nearest_neighbors True \
        --incremental_batching True \
        --sliding_batching 0 \
        --pure_egnn 0 \
        --wrong_ratio_mixup 0.0 \
```
