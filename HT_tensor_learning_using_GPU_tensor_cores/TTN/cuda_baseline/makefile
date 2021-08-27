#cc=/usr/local/cuda-10.1/bin/nvcc
cc = nvcc
des=tree
source = unitree_init.cpp data_gen.cu tool_function.cu opt_tree.cu opt_enery_layer.cu h_mpo.cpp
 
link = -lcublas -lcusolver -lcurand -lcutensor -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) -L${CUTENSOR_ROOT}/lib/11.0/ -I${CUTENSOR_ROOT}/include $(link)

clean:
	rm -rf $(des)
