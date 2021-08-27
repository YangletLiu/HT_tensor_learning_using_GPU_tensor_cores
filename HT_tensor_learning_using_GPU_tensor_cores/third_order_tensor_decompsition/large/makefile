cc=/usr/local/cuda-10.1/bin/nvcc
des=ht
 
source = ht_init.cu  htd_tensorcore.cu kernel.cu ttm.cu
#source = ht_init.cu htd_batch.cu kernel.cu ttm.cu

link = -lcublas -lcusolver -lcurand -lcufft -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)