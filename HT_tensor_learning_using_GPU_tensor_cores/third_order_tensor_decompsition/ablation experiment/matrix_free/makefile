cc= nvcc
des=ht
 
source = ht_init.cu htd_unopt.cu kernel.cu ttm.cu  
 
link = -lcublas -lcusolver -lcurand -lcufft -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)