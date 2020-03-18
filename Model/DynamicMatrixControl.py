import numpy as np
class DMC:
    '''A_model是输入阶跃响应
       R_t R矩阵
       Q 矩阵
       M控制域长度
       P预测域长度
       m输入长度
       p输出长度
       B_model前馈响应
       numv前馈数量
    '''
    def __init__(self,A_model,R_t,Q,M,P,m,p):
        self.a_model_series=A_model
        self.R_t=R_t
        self.Q=Q
        self.M=M
        self.P=P
        self.m=m
        self.p=p
    def compute(self):


        '''得到响应矩阵A'''
        A=np.zeros((self.P*self.p,self.M*self.m))
        for loopouti in range(self.p):
            for loopini in range(self.m):
                for loopmi in range(self.M):
                    A[self.P*loopouti+loopmi:self.P*(loopouti+1),self.M*loopini+loopmi]= self.a_model_series[loopouti, loopini, 0:self.P - loopmi]




        deltau=np.dot(np.dot(np.linalg.inv(np.dot(np.dot(A.transpose(), self.Q), A) + self.R_t), A.transpose()), self.Q)
        RESULTS=   {'deltau':deltau, 'A':A}
        return RESULTS