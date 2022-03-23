# HT_tensor_learning_using_GPU_tensor_cores
This repository contains applications for HT tensor learning using GPU tensor cores, including HT tensor decompostion, HT tensor layer and TTN algorithm. 

## File structure

```
HT_tensor_learning_using_GPU_tensor_cores
│ 
├── third_order_tensor_decompsition
│   ├── CUDA
│   	├── Baseline
│   	├── Optimized
│   	├── TSQR
│   	└── ablation experiment	
│
│   ├── Python
│   	├── HT_TensorLy.py
│   	└── HT_TensorNetwork.py
│ 
│ 
│ 
├── fourth_order_tensor_decomposition
│   ├── multiple_GPUs
│       ├── 2_GPUs      
│       └── 8_GPUs      
│       
│    ├── single_GPU
│        ├── CUDA 
│            ├── Baseline
│            └── Optimized
│ 
│        ├── Python
│            ├── HT_4d_TensorLy.py
│            └── HT_4d_TensorNetwork.py 
│ 
│ 
├── HT_tensor_layer 
│   ├── FC_MNIST
│       ├── mnist.py
│       ├── mnist_ht.py 
│       └── mnist_half.py
│ 
│   ├── Alex_ImageNet
│       ├── model (define the model)
│       ├── main.py (using CUDA cores)
│       └── main_TC.py (using GPU tensor cores)
│
│
├── TTN
│   ├── CUDA
│       ├── Baseline
│       └── Optimized
│    
│   ├── Python
│       └── groundstate_example.py (using TensorNetwork with JAX)
│
│ 
├── RESULT
│   	
└── README.md
```


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
## Result
<div style="float:left"><img width="580" height="470" src="https://raw.githubusercontent.com/XiaoYangLiu-FinRL/HT_tensor_learning_using_GPU_tensor_cores/main/RESULT/3d_runT.png"/></div>

<div style="float:left"><img width="580" height="470" src="https://raw.githubusercontent.com/XiaoYangLiu-FinRL/HT_tensor_learning_using_GPU_tensor_cores/main/RESULT/4d_runT.png"/></div>

