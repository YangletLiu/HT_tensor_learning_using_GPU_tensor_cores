cc=nvcc
des=omp
  
source = omp_init.cpp  kernel.cu 

link = -lcublas -lcusolver -lcurand -lcufft -std=c++11

$(des):$(source)
	$(cc) -Xcompiler -fopenmp -o $(des) $(source) $(link)

clean:
	rm -rf $(des)