import numpy as np
class DMC:
    def __init__(self,A_model,uwt,ywt,M,P,m,p,N):
        self.A=A_model
        self.uwt=uwt
        self.ywt=ywt
        self.M=M
        self.P=P
        self.m=m
        self.p=p
        self.N=N

    def compute(self):
        '''得到R矩阵'''
        R_t=np.eye((self.M*self.m))
        for loop_ini in range(self.m):
            R_t[self.M*loop_ini:self.M*(loop_ini+1),:]=self.uwt[loop_ini]*R_t[self.M*loop_ini:self.M*(loop_ini+1),:]

        print(R_t)

        print()




        pass