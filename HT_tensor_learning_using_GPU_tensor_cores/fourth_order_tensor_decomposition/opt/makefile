cc=/usr/local/cuda-10.1/bin/nvcc
des=ht
 
source = ht_init.cu  htd4_tensorcore.cu kernel.cu ttm.cu 
#source = ht_init.cu htd.cu kernel.cu  
#source = ht_init.cu htd4.cu kernel.cu ttm.cu

link = -lcublas -lcusolver -lcurand -lcufft -lnvidia-ml -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link) -L /lib64

clean:
	rm -rf $(des)