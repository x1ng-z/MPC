import numpy as np

class Tools:
    def __init__(self):
        pass
    '''工具类'''
    def buildU(self,U,m,M):
        Um=np.zeros((m*M,1))
        for indexIn in range(m):
            for nodein in range(M):
                Um[indexIn * M + nodein, 0] = U[indexIn]
        return Um

    def buildY(self,Y,p,P):
        Yp = np.zeros((p * P, 1))
        for indexIn in range(p):
            for nodein in range(P):
                Yp[indexIn * P + nodein, 0] = Y[indexIn, 0]
        return Yp

    def buildY0(self,p,N,y0):
        Y0=np.zeros((p * N, 1))
        for outi in range(p):
            for step in range(N):
                Y0[outi*N+step,0]=y0[outi]
        return Y0

    def biuldWi(self,p,P,wi):
        W_i = np.zeros((p * P, 1))
        for loop_ri in range(P * p):
            W_i[loop_ri, 0] = wi[int(loop_ri / P)]

        return W_i

