```bash
$ python main.py --arch alexnet --pretrained --evaluate
$ CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.launch --nproc_per_node=3 main.py --arch alexnet 
```
