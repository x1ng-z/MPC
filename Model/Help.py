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

