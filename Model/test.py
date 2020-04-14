import numpy as np
import sys
if __name__=='__main__':
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
