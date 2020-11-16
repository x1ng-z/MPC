import numpy as np


class DMC:

    def __init__(self, step_respon, N, R, Q, M, P, m, p, alphe,alphemethod):
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
                :arg self.alphemethod 柔化系数方法
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
        self.alphemethod=alphemethod

    def compute(self):
        '''得到响应矩阵A'''
        alphecoeff = np.zeros((self.p * self.P, 1))
        for indexp in range(self.p):
            if(self.alphemethod[indexp]=='before'):
                alphecoeff[self.P * indexp:self.P * (indexp + 1)]= np.array(
                [1 - self.alphe[indexp] ** i for i in range(1, self.P + 1)]).reshape(-1,1)
            if(self.alphemethod[indexp]=='after'):
                alphecoeff[self.P * indexp:self.P * (indexp + 1)] = np.flipud(np.array(
                    [1 - self.alphe[indexp] ** i for i in range(1, self.P + 1)]).reshape(-1, 1))
        alphediag_2 = np.power(np.diagflat(alphecoeff),2)#p*P
        dynamic_matrix_P = np.zeros((self.P * self.p, self.M * self.m))  # P预测域内的响应矩阵动态矩阵
        dynamic_matrix_N = np.zeros((self.N * self.p, self.M * self.m))  # N全部响应序列的响应动态矩阵
        for indexpi in range(self.p):
            for indexmi in range(self.m):
                for indexMi in range(self.M):
                    dynamic_matrix_P[self.P * indexpi + indexMi:self.P * (indexpi + 1),
                    self.M * indexmi + indexMi] = self.step_respon[indexpi, indexmi, 0:self.P - indexMi]

                    dynamic_matrix_N[self.N * indexpi + indexMi:self.N * (indexpi + 1),
                    self.M * indexmi + indexMi] = self.step_respon[indexpi, indexmi, 0:self.N - indexMi]

        justQ = np.dot(self.Q, alphediag_2)#调整的Q系数
        control_vector = np.dot(
            np.dot(np.linalg.pinv(np.dot(np.dot(dynamic_matrix_P.transpose(), justQ), dynamic_matrix_P) + self.R),
                   dynamic_matrix_P.transpose()), justQ)
        return control_vector, dynamic_matrix_P, dynamic_matrix_N
