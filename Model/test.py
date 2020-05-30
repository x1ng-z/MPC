import numpy as np
import sys
if __name__=='__main__':
    d = {'1': 'one', '3': 'three', '2': 'two', '5': 'five', '4': 'four'}
    print("100" in d)
    dmvmin = np.zeros((2* 3, 1))
    print(dmvmin.reshape(-1))
    x = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    x0 = np.asarray(x)
    y_0P = np.zeros((1 * 135, 1))
    will = y_0P + np.dot(self.A, deltu)
    gradient = 2 * np.dot(np.dot(-1 * self.A.transpose(), self.Q), will) + 2 * np.dot(self.R, deltu)

    print(H)


    mmm=np.array([[1],[2]])
    nnn=np.array([1,2])
    for index,needcheckdmv in np.ndenumerate(nnn):
        print(index,needcheckdmv)
        print(nnn[index])
    tmmm=nnn.reshape(-1)


    k=2*4000 // 8000
    aa=np.zeros((3,1))
    bb=np.ones((1,3))
    cc=aa.all()<=bb
    for i in range(0, len(sys.argv)):
        m=2
        M=3
        B = np.zeros((m * M, m * M))
        for indexIn in range(m):
            for noderow in range(M):
                for nodecol in range(M):
                    if (nodecol <= noderow):
                        B[indexIn * M + noderow, indexIn * M + nodecol] = 1
        print(B)
        strs = sys.argv[i]
        print(i)
        if i==2:
            print(int(strs))
