# HT_tensor_learning_using_GPU_tensor_cores
This repository contains applications for HT tensor learning using GPU tensor cores, including HT tensor decompostion, HT tensor layer and TTN algorithm. 

## File structure

> HT_tensor_learning_using_GPU_tensor_cores
>> third_order_tensor_decompsition
>>> baseline ----- unoptimized \r\n
>>> opt      ---- optimized
>>> large    ----- using TSQR algorithm
>> fourth_order_tensor_decomposition
>>> single GPU
>>>> baseline ---- unoptimized
>>>> opt       ---- optimized
>>> multiple_GPUs    ----- using shard mode
>> HT_tensor_layer
>>> mnist.py  ----- fully connect
>>> mnist_ht.py ----- HT tensor layer
>>> mnist_half.py ----- HT tensor layer using apex
>> TTN
>>> cuda_baseline            ----- unoptimized
>>> cuda_unitree_tensorcore  ----- optimized
>>> groundstate_example.py   ----- TensorNetwork-JAX
