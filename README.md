# How to reproduce the experiments results

* Run ```pip install -e .``` at the proj root directory

* Then, train a model under inductive setting. 
```
bash scripts/run_small_dataset.sh $GPU_NUMBER
```

GPU_NUMBER is the index of GPU to use. For example, ```bash scripts/run_small_dataset.sh 3``` means using GPU 3 to train the models.

* Then, perform the edit on pretrained model.

```
bash scripts/eval.sh $GPU_NUMBER
bash scripts/eval_edg.sh $GPU_NUMBER
bash scripts/eval_edg_plus.sh $GPU_NUMBER
```