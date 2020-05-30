import numpy as np


class DMC:

    def __init__(self, step_respon, R, Q, M, P, m, p):
        '''
            function:
                预测控制
            Args:
                :arg step_respon 是输入阶跃响应 shape(p,m,N)
                :arg R矩阵diag(m*M,m*M)
                :arg Q 矩阵diag(p*P,p*P)
                :arg M 控制域长度
                :arg P预测域长度
                :arg m 输入长度
                :arg p 输出长度
            Returns:

        '''

        self.step_respon = step_respon
        self.R = R
        self.Q = Q
        self.M = M
        self.P = P
        self.m = m
        self.p = p

    def compute(self):
        '''得到响应矩阵A'''
        dynamic_matrix = np.zeros((self.P * self.p, self.M * self.m))
        for loopouti in range(self.p):
            for loopini in range(self.m):
                for loopmi in range(self.M):
                    dynamic_matrix[self.P * loopouti + loopmi:self.P * (loopouti + 1),
                    self.M * loopini + loopmi] = self.step_respon[loopouti, loopini, 0:self.P - loopmi]

        control_vector = np.dot(
            np.dot(np.linalg.pinv(np.dot(np.dot(dynamic_matrix.transpose(), self.Q), dynamic_matrix) + self.R),
                   dynamic_matrix.transpose()), self.Q)
        return control_vector, dynamic_matrix
