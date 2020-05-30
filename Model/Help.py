import numpy as np


class Tools:
    def __init__(self):
        pass

    '''工具类'''

    def buildU(self, U, m, M):
        Um = np.zeros((m * M, 1))
        for indexIn in range(m):
            for nodein in range(M):
                Um[indexIn * M + nodein, 0] = U[indexIn]
        return Um

    def buildY(self, Y, p, P):
        Yp = np.zeros((p * P, 1))
        for indexIn in range(p):
            for nodein in range(P):
                Yp[indexIn * P + nodein, 0] = Y[indexIn, 0]
        return Yp

    def buildY0(self, p, N, y0):
        Y0 = np.zeros((p * N, 1))
        for outi in range(p):
            for step in range(N):
                Y0[outi * N + step, 0] = y0[outi]
        return Y0

    def biuldWi(self, p, P, wi):
        W_i = np.zeros((p * P, 1))
        for loop_ri in range(P * p):
            W_i[loop_ri, 0] = wi[int(loop_ri / P)]

        return W_i

    '''
               function:
                     构建漏斗
               Args:
                   :arg wp sp值
                   :arg deadZones 死区
                   :arg funelInitValues 漏斗初始值
                   :arg N 阶跃响应数据点数量
                   :arg p pv数量
               Returns:
                    返回的漏斗数据为shape为(2,p*N)
                   funels=[ up1  up2..upN  pv的高限制都在这一行
                   donw1 donw2...donwN   pv的低限制都在这一行 ]
    '''
    def buildFunel(self, wp, deadZones, funelInitValues, N, p):

        funels = np.zeros((2, p * N))
        leftUpPointsY = wp + deadZones + funelInitValues
        leftDownPointsY = wp - deadZones - funelInitValues
        rightUpProintsY = wp + deadZones
        rightdDownProintsY = wp - deadZones

        for pvi in range(p):
            for lineNum in range(2):
                Upki = 0
                Upbi = 0
                if lineNum == 0:
                    Upki = np.true_divide(leftUpPointsY[pvi] - rightUpProintsY[pvi], (1 - N))
                    Upbi = leftUpPointsY[pvi]
                elif lineNum == 1:
                    Upki = np.true_divide(leftDownPointsY[pvi] - rightdDownProintsY[pvi], (1 - N))
                    Upbi = leftDownPointsY[pvi]

                for lineLimti in range(N):
                    funels[lineNum, pvi * N + lineLimti] = Upki * lineLimti + Upbi
        return funels
