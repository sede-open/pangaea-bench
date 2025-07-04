## HLSBurnScars
Fine-tuning (UpperNet Decoder) and Evaluation of TerraMind (v1_large) on HLSBurnScars
```
HYDRA_FULL_ERROR=1 torchrun --nnodes=1 --nproc_per_node=1 --master-port=29501 pangaea/run.py --config-name=train dataset=hlsburnscars encoder=terramind_large decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy task=segmentation
```
Full Fine-tuning (both Encoder and UpperNet Decoder) and Evaluation of TerraMind (v1_large) on HLSBurnScars
```
HYDRA_FULL_ERROR=1 torchrun --nnodes=1 --nproc_per_node=1 --master-port=29501 pangaea/run.py --config-name=train dataset=hlsburnscars encoder=terramind_large decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy task=segmentation finetune=True
```

## Sen1Floods11
Fine-tuning (UpperNet Decoder) and Evaluation of TerraMind (v1_large) on Sen1Floods11
```
HYDRA_FULL_ERROR=1 torchrun --nnodes=1 --nproc_per_node=1 --master-port=29501 pangaea/run.py --config-name=train dataset=sen1floods11 encoder=terramind_large decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy task=segmentation 
```
Full Fine-tuning (both Encoder and UpperNet Decoder) and Evaluation of TerraMind (v1_large) on Sen1Floods11
```
HYDRA_FULL_ERROR=1 torchrun --nnodes=1 --nproc_per_node=1 --master-port=29501 pangaea/run.py --config-name=train dataset=sen1floods11 encoder=terramind_large decoder=seg_upernet preprocessing=seg_default criterion=cross_entropy task=segmentation finetune=True
```