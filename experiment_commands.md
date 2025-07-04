Fine-tuning and Evaluation of TerraMind (v1_large) on HLSBurnScars
```
HYDRA_FULL_ERROR=1 torchrun --nnodes=1 --nproc_per_node=1 --master-port=29501 pangaea/run.py --config-name=train dataset=hlsburnscars encoder=terramind_large decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy task=segmentation
```