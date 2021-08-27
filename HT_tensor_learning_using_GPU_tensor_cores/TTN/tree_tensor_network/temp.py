A = [1,2,3,4]
B = [1,2,3,4]
C = []
def F(A,B,C):
	A[0] - 1
	C.append(A[0] - B[0])
	return C

C = F(A,B,C)
print(A)
print(C)