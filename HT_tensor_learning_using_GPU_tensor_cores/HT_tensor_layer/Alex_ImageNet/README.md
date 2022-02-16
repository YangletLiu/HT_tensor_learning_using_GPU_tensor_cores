
Run the code by
```bash
CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.launch --nproc_per_node=3 main.py --arch alexnet # Conventional layer
CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.launch --nproc_per_node=3 main.py --arch ht # HT tensor layer
```
