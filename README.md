# HT_tensor_learning_using_GPU_tensor_cores
This repository contains applications for HT tensor learning using GPU tensor cores, including HT tensor decompostion, HT tensor layer and TTN algorithm. 

## File structure

> third_order_tensor_decompsition
>> CUDA
>>> Baseline ----- unoptimized <br>
>>> Optimized ---------- optimized <br>
>>> TSQR -------- using TSQR algorithm <br>
>>> ablation experiment
>>>> eig2svd ----- parallel Eigenvalue decomposition <br>
>>>> matrix_free ----- matricization-free access <br>
>>>> only_tensor_core ----- only usee tensor core <br>

>> Python
>>> HT_TensorLy.py  ---- implementation using TensorLy <br>
>>> HT_TensorNetwork.py ---- implementation using TensorNetwork <br>


> fourth_order_tensor_decomposition
>> multiple_GPUs ----- using shard mode
>>> 2_GPUs ----- using 2 GPUs <br>
>>> 8_GPUs ----- using 8 GPUs <br>

>> single GPU
>>> CUDA
>>>> Baseline ----- unoptimized <br>
>>>> Optimized ---------- optimized <br>

>>> Python
>>>> HT_4d_TensorLy.py  ---- implementation using TensorLy <br>
>>>> HT_4d_TensorNetwork.py ---- implementation using TensorNetwork <br>


> HT_tensor_layer
>> FC_MINST
>>> mnist.py ---------- fully connect <br>
>>> mnist_ht.py ------- HT tensor layer <br>
>>> mnist_half.py ----- HT tensor layer using tensor core <br>


>> Alex_ImageNet
>>> model ----- define the model <br> 
>>> main.py ----- the AlexNet in ImageNet (--arch to choose alexnet/ht) <br>
>>> main_TC.py ----- the AlexNet in ImageNet using tensor core <br>

> TTN
>> CUDA 
>>> Baseline -------------------unoptimized <br>
>>> Optimized -------- optimized <br>

>> Python
>>> groundstate_example.py -------- TensorNetwork-JAX <br>


## Running

### CUDA Code
Running the makefile with
```
$ make clean
$ make
```
### Python Code
The AlexNet model is run with

```
$ python main.py --arch alexnet
```
The AlexNet model using HT tensor layer is run with
```
$ python main.py --arch ht
```
The HT tensor layer on AlexNet using GPU tensor cores with
```
$ python main_TC.py -- arch ht
```

