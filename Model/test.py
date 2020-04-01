import numpy as np
m=2
M=8
B = np.zeros((m * M, m * M))
for indexIn in range(m):
    for noderow in range(M):
        for nodecol in range(M):
            if (nodecol <= noderow):
                B[indexIn * M + noderow, indexIn * M + nodecol] = 1
print(np.zeros(0))
x=np.array([1,2,3,4,5,6,7,8])
xm = x[1:-1]
print(xm)

xm_m1 = x[:-2]
print(xm_m1)

xm_p1 = x[2:]
print(xm_p1)

aa=np.array([[1,2],[3,4]])
bb=np.array([[5,6],[7,8]])
print(aa@bb)
print(np.dot(aa ,bb))