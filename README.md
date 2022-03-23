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


>> HT_tensor_layer
>>> FC_MINST
>>>> mnist.py ---------- fully connect <br>
>>>> mnist_ht.py ------- HT tensor layer <br>
>>>> mnist_half.py ----- HT tensor layer using apex <br>


>>> Alex_ImageNet
>>>> main.py ----- the AlexNet in ImageNet (--arch to choose alexnet/ht) <br>

>> TTN
>>> cuda_baseline -------------------unoptimized <br>
>>> cuda_unitree_tensorcore --------optimized <br>
>>> groundstate_example.py --------TensorNetwork-JAX <br>
