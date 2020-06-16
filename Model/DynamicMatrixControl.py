import numpy as np


class DMC:

    def __init__(self, step_respon, N, R, Q, M, P, m, p, alphe):
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
                :arg alphe 柔化参数，用于保留最终误差，弱化初期误差shape=(,p),如[0.8,0.7]
            Returns:

        '''

        self.step_respon = step_respon
        self.N = N
        self.R = R
        self.Q = Q
        self.M = M
        self.P = P
        self.m = m
        self.p = p
        self.alphe = alphe

    def compute(self):
        '''得到响应矩阵A'''
        alphecoeff = np.zeros((self.p * self.P, 1))
        for indexp in range(self.p):
            alphecoeff[self.P * indexp:self.P * (indexp + 1)] = np.array(
                [1 - self.alphe[indexp] ** i for i in range(1, self.P + 1)]).reshape(-1,1)
        alphediag_2 = np.power(np.diagflat(alphecoeff),2)#p*P
        dynamic_matrix_P = np.zeros((self.P * self.p, self.M * self.m))  # P预测域内的响应矩阵动态矩阵
        dynamic_matrix_N = np.zeros((self.N * self.p, self.M * self.m))  # N全部响应序列的响应动态矩阵
        for loopouti in range(self.p):
            for loopini in range(self.m):
                for loopmi in range(self.M):
                    dynamic_matrix_P[self.P * loopouti + loopmi:self.P * (loopouti + 1),
                    self.M * loopini + loopmi] = self.step_respon[loopouti, loopini, 0:self.P - loopmi]

                    dynamic_matrix_N[self.N * loopouti + loopmi:self.N * (loopouti + 1),
                    self.M * loopini + loopmi] = self.step_respon[loopouti, loopini, 0:self.N - loopmi]

        justQ = np.dot(self.Q, alphediag_2)#调整的Q系数
        control_vector = np.dot(
            np.dot(np.linalg.pinv(np.dot(np.dot(dynamic_matrix_P.transpose(), justQ), dynamic_matrix_P) + self.R),
                   dynamic_matrix_P.transpose()), justQ)
        return control_vector, dynamic_matrix_P, dynamic_matrix_N
