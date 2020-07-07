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

    def biuldWi(self, p, P, wi, yreal, alph):
        W_i = np.zeros((p * P, 1))
        # for loop_ri in range(P * p):
        #     W_i[loop_ri, 0] = wi[int(loop_ri / P)]
        for indexp in range(p):
            for indexP in range(P):
                tempwp = (1 - alph ** (indexP + 1)) * wi[indexp] + alph ** (indexP + 1) * yreal[indexp]
                W_i[indexp * P + indexP, 0] = tempwp
        return W_i

    def biuldWiByFunel(self, p, P, N, y0, funels):
        '''
        :param p: pv数量
        :param P: 预测域大小
        :param N 响应序列长度
        :param y0: y0预测序列
        :param funels: shape(2,N*p)[ up1  up2..upN  pv的高限制都在这一行
                         donw1 donw2...donwN   pv的低限制都在这一行 ]
        :return: W_i shape=(p * P, 1)
        '''

        W_i = np.zeros((p * P, 1))

        for indexp in range(p):
            '''这里，响应和漏斗都是N的，要截取为P的'''
            upfunel = funels[0, indexp * N:(indexp) * N + P].copy()
            downfunel = funels[1, indexp * N:(indexp ) * N + P].copy()
            # 判断超过上funel，超过则获取 funle,不然则获取y0的
            y0P_test = y0[indexp * N:(indexp) * N + P, 0].copy()
            y0P_rec = y0[indexp * N:(indexp) * N + P, 0].copy()

            y0P_rec[((y0P_test <= downfunel) + (y0P_test >= upfunel))]=0#截取y0部分
            downfunel[downfunel < y0P_test] = 0#截取下漏斗部分
            upfunel[y0P_test < upfunel] = 0#截取上漏斗部分
            W_i[indexp * P:(indexp+1)*P, 0] = y0P_rec+downfunel+upfunel
        return W_i

    def buildFunel(self, wp, deadZones, funelInitValues, N, p,funneltype,maxfunnelvale,minfunnelvale):
        '''
                     function:
                           构建漏斗
                     Args:
                         :arg wp sp值
                         :arg deadZones 死区
                         :arg funelInitValues 漏斗初始值
                         :arg N 阶跃响应数据点数量
                         :arg p pv数量
                         :arg funneltype漏斗类型
                         :arg maxfunnelvale上漏斗最大值
                         ;:arg minfunnelvale 下漏斗最小值
                     Returns:
                         :return originalfunnels 原始全漏斗数据为shape为(2,p*N)
                         funels=[ up1  up2..upN  pv的高限制都在这一行
                         donw1 donw2...donwN   pv的低限制都在这一行 ]

                          :return decoratefunnels 根据漏斗类型修饰的漏斗数据为shape为(2,p*N)
                         funels=[ up1  up2..upN  pv的高限制都在这一行
                         donw1 donw2...donwN   pv的低限制都在这一行 ]

          '''
        originalfunnels = np.zeros((2, p * N))
        decoratefunnels=np.zeros((2, p * N))
        leftUpPointsY = wp + deadZones + funelInitValues
        leftDownPointsY = wp - deadZones - funelInitValues
        rightUpProintsY = wp + deadZones
        rightdDownProintsY = wp - deadZones
        funnelmaxminmatrix=np.array([maxfunnelvale,minfunnelvale])
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
                    originalfunnels[lineNum, pvi * N + lineLimti] = Upki * lineLimti + Upbi
                    decoratefunnels[lineNum, pvi * N + lineLimti]=(Upki * lineLimti + Upbi)*funneltype[pvi,lineNum]*funnelmaxminmatrix[lineNum]
        return originalfunnels,decoratefunnels
