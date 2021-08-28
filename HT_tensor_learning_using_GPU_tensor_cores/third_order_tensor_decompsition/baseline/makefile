cc= nvcc
des=ht
 
source = ht_init.cu htd_unopt.cu kernel.cu  
 
link = -lcublas -lcusolver -lcurand -lcufft -lnvidia-ml -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link) -L /lib64

clean:
	rm -rf $(des)