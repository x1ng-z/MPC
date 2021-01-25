import numpy as np
# import matplotlib.pyplot as plt
# import apc
# import Help
import sys
import requests
import json
import time

import gc
import os
import sys
import traceback

from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import SR1
from scipy.optimize import Bounds


class MinJ:
    '''
    deltav前馈增量
    :arg alphe 柔化参数，用于保留最终误差，弱化初期误差shape=(,p),如[0.8,0.7]
    '''
    def __init__(self,wp,y0,u0,A,Q,R,M,P,m,p,Umin,Umax,Ymin,Ymax,alphe,alphemethod):
        self.wp=wp
        self.y0=y0
        self.u0=u0
        self.dynamix_matrix=A
        self.Q=Q
        self.R=R
        self.M=M
        self.P=P
        self.Umin=Umin
        self.Umax=Umax
        self.Ymin=Ymin
        self.Ymax=Ymax
        self.m=m
        self.p=p
        self.Dumin=0
        self.Dumax=0
        self.alphe=alphe
        self.alphemethod=alphemethod
        #self.deltav=deltav
        #self.B_N=B_timeserial#数据矩阵形式(输出个数*响应序列长度，前馈个数)
        '''B矩阵，用于计算M个增量的变化'''
        self.B = np.zeros((m * M, m * M))
        for indexIn in range(m):
            for noderow in range(M):
                for nodecol in range(M):
                    if (nodecol <= noderow):
                        self.B[indexIn * M + noderow, indexIn * M + nodecol] = 1

        self.alphecoeff = np.zeros((self.p * self.P, 1))
        for indexp in range(self.p):
            if (self.alphemethod[indexp] == 'before'):
                self.alphecoeff[self.P * indexp:self.P * (indexp + 1)] = np.array(
                    [1 - self.alphe[indexp] ** i for i in range(1, self.P + 1)]).reshape(-1, 1)
            if (self.alphemethod[indexp] == 'after'):
                self.alphecoeff[self.P * indexp:self.P * (indexp + 1)] = np.flipud(np.array(
                    [1 - self.alphe[indexp] ** i for i in range(1, self.P + 1)]).reshape(-1, 1))
        self.alphediag =np.diagflat(self.alphecoeff)
        self.alphediag_2=np.power(np.diagflat(self.alphecoeff), 2)# p*P

    def setDumin(self,Dumin):
        self.Dumin=Dumin

    def setDumax(self,Dumax):
        self.Dumax=Dumax

    def sety0(self,y0):
        self.y0 = y0
    def setwp(self,wp):
        self.wp = wp

    def setu0(self,u0):
        self.u0 = u0

    def setUmin(self,Umin):
        self.Umin = Umin

    def setUmax(self,Umax):
        self.Umax = Umax

    def setYmin(self,Ymin):
        self.Ymin = Ymin

    def setYmax(self,Ymax):
        self.Ymax = Ymax

    def J(self, dmv):
        e=(self.wp - (self.y0 + np.dot(self.dynamix_matrix, dmv))).reshape(-1,1)
        juste=e*self.alphecoeff
        aa=np.dot(np.dot(juste.transpose(), self.Q), juste) + np.dot(np.dot(dmv.transpose(), self.R), dmv)
        # print("loss=",aa[0][0])
        return aa[0][0]

    # def J_with_Balance(self, delu):
    #
    #     e = (self.wp - (self.y0 + np.dot(self.A, delu))).transpose()
    #     aa = np.dot(np.dot(e.transpose(), self.Q), e) + np.dot(np.dot(delu.transpose(), self.R), delu)
    #     # print("cost")
    #     # print(aa)
    #     cost=aa[0][0]+np.std(np.add(self.u0, delu))*self.Balance
    #     return cost

    def gradientJ(self,deltu):
        will=self.wp-(self.y0 + np.dot(self.dynamix_matrix, deltu))
        # print(will[0,:].transpose())
        justQ=np.dot(self.Q, self.alphediag_2)
        gradient= np.dot(np.dot(-2*self.dynamix_matrix.transpose(), justQ), will[0, :]) +np.dot(2*self.R, deltu)
        return gradient

    def hesionJ(self,deltu):
        justQ = np.dot(self.Q, self.alphediag_2)
        H=(2 * np.dot(np.dot(self.dynamix_matrix.transpose(), justQ), self.dynamix_matrix) + 2 * self.R)
        # print(H)
        return H.transpose()


    def comput(self):
        '''
        The bound constraints 0≤x0≤1 and −0.5≤x1≤2.0 are defined using a Bounds object.
        bounds = Bounds([0, -0.5], [1.0, 2.0])#增量不能超过dmvmax
        :return:
        '''
        bounds = Bounds( -1*self.Dumax.reshape(-1), self.Dumax.reshape(-1))
        ##lb <= A.dot(x) <= ub #累计增量+mv初始值不能超过umax umin
        linear_constraintu = LinearConstraint(self.B , (self.Umin-self.u0).transpose()[0,:],(self.Umax-self.u0).transpose()[0,:])
        #linear_constrainty = LinearConstraint(self.A,(self.Ymin-self.y0).transpose()[0,:],(self.Ymax-self.y0).transpose()[0,:])
        x0=np.zeros(self.M*self.m).transpose()
        res = minimize(self.J, x0, method='trust-constr',
                       jac=self.gradientJ,
                       hess=self.hesionJ,
                       # jac="2-point",#在求一次导比较难时使用
                       # hess=SR1(),#在求二阶倒数比较难时使用
                       constraints=[linear_constraintu],
                       #constraints=[linear_constraintu,linear_constrainty],
                       options={'verbose': 1,'disp': True},
                       bounds=bounds
                       )

        return res

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
                         :arg maxfunnelvale上漏斗最大值 近似正无穷
                         ;:arg minfunnelvale 下漏斗最小值 近似负无穷
                     Returns:
                         :return originalfunnels 原始全漏斗数据为shape为(2,p*N)
                         funels=[ up1  up2..upN  pv的高限制都在这一行
                         donw1 donw2...donwN   pv的低限制都在这一行 ]

                          :return decoratefunnels 根据漏斗类型修饰的漏斗数据为shape为(2,p*N)
                         funels=[ up1  up2..upN  pv的高限制都在这一行
                         donw1 donw2...donwN   pv的低限制都在这一行 ]

          '''
        undecoratefunnels = np.zeros((2, p * N))
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
                    undecoratefunnels[lineNum, pvi * N + lineLimti] = Upki * lineLimti + Upbi
                    decoratefunnels[lineNum, pvi * N + lineLimti]=(Upki * lineLimti + Upbi)+funneltype[pvi,lineNum]*funnelmaxminmatrix[lineNum]
        return undecoratefunnels,decoratefunnels


    def build_B_respond(self,origionB):
        p=origionB.shape[0]
        f=origionB.shape[1]#feedforwardNum
        N=origionB.shape[2]
        B_step_response_sequence = np.zeros((p * N, f))
        for outi in range(p):
            for ini in range(f):
                B_step_response_sequence[outi * N:(outi + 1) * N, ini] = origionB[outi, ini]

        return B_step_response_sequence
    def build_A_respond(self,origionA):
        p=origionA.shape[0]
        m=origionA.shape[1]
        N=origionA.shape[2]
        origion_A_step_response_sequence = np.zeros((p * N, m))
        for loop_outi in range(p):
            for loop_ini in range(m):
                origion_A_step_response_sequence[N * loop_outi:N * (loop_outi + 1),
                loop_ini] = origionA[loop_outi, loop_ini, :]
        return origion_A_step_response_sequence


    def getFirstY0Position(self,matrixPvMvMapping):
            '''
            根据映射矩阵，将y0转换为映射前数量的y0
            '''
            p = matrixPvMvMapping.shape[0]
            m = matrixPvMvMapping.shape[1]
            y0Positon=np.zeros(p)
            indexmapping=0
            for indexp in range(p):
                isNewRow=True
                for indexm in range(m):
                    if (matrixPvMvMapping[indexp][indexm] == 1):
                        if isNewRow:
                            y0Positon[indexp]=indexmapping
                            isNewRow=False
                        indexmapping=indexmapping+1

            return y0Positon

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

class apc:

    def __init__(self, P, p, M, m, N, outStep, feedforwardNum, A, B, qi, ri, pvusemv, alphe, funneltype,alphemethod):
        '''
                    function:
                        预测控制
                    Args:
                           :arg P 预测时域长度
                           :arg p PV数量
                           :arg M mv计算后续输出几步
                           :arg m mv数量
                           :arg N 阶跃响应序列个数
                           :arg outStep 输出间隔
                           :arg feedforwardNum 前馈数量
                           :arg A mv对pv的阶跃响应
                           :arg B ff对pv的阶跃响应
                           :arg qi 优化控制域矩阵，用于调整sp与预测的差值，在滚动优化部分
                           :arg ri 优化时间域矩阵,用于约束调整dmv的大小，在滚动优化部分
                           :arg pvusemv 一个矩阵，标记pv用了哪些mv
                           :arg alphe 柔化系数
                           :arg alphemethod 柔化系数方法 目前支持before after两种
                           :arg funneltype 漏斗类型shape=(pv数量，2)：如pv数量为2 [[0,0],[1,0],[0,1]],[0,0]全漏斗，[1,0]下漏斗，[0,1]上漏斗
                    '''

        '''预测时域长度'''
        self.P = P

        '''输出个数'''
        self.p = p

        '''控制时域长度'''
        self.M = M

        '''输入个数'''
        self.m = m

        '''建模时域'''
        self.N = N

        '''输出间隔'''
        self.outStep = outStep

        '''前馈数量'''
        self.feedforwardNum = feedforwardNum

        '''pv 使用mv的标记矩阵'''
        self.pvusemv = pvusemv

        self.alphe = alphe

        self.alphemethod=alphemethod

        '''mv 对 pv 的阶跃响应'''
        self.A_step_response_sequence = np.zeros((p * N, m))
        for loop_outi in range(p):
            for loop_ini in range(m):
                self.A_step_response_sequence[N * loop_outi:N * (loop_outi + 1), loop_ini] = A[loop_outi, loop_ini, :]

        '''ff 对 pv 的阶跃响应'''
        self.B_step_response_sequence = []

        '''前馈数量为0 则不需要初始化前馈响应B_step_response_sequence'''
        if feedforwardNum != 0:
            self.B_step_response_sequence = np.zeros((p * N, feedforwardNum))
            for outi in range(p):
                for ini in range(feedforwardNum):
                    self.B_step_response_sequence[outi * N:(outi + 1) * N, ini] = B[outi, ini]

        '''控制优化区域Q矩阵 shape=(p*P,p*P),是一个对角矩阵diag'''
        self.Q = np.eye(p * P)
        for indexp in range(p):
            self.Q[P * indexp:P * (indexp + 1), :] = qi[indexp] * self.Q[P * indexp:P * (indexp + 1), :]

        '''得到R矩阵 优化时间区域 shape=(m*M,m*M),是一个diag对角矩阵'''
        self.R = np.eye((M * m))
        for indexm in range(m):
            self.R[M * indexm:M * (indexm + 1), :] = ri[indexm] * self.R[M * indexm:M * (indexm + 1), :]

        '''H Matrix反馈矫正系数矩阵
        '''
        self.H = np.zeros((p * N, p))  # [输出引脚*阶跃时序长度，输出引脚]

        '''build矫正 H matrix'''
        for indexp in range(p):
            for indexn in range(N):
                if indexn == 0:
                    self.H[indexn + N * indexp, indexp] = 1  # hi[loop_outi]
                else:
                    self.H[indexn + N * indexp, indexp] = 0.7  # hi[loop_outi]

        '''位移矩阵'''
        self.S = np.zeros((p * N, p * N))  # [输出引脚*阶跃时序长度，输出引脚*阶跃时序长度]

        '''构造计算位移矩阵S'''
        for loop_outi in range(p):
            for loop_stepi in range(0, N - 1):
                self.S[loop_outi * N + loop_stepi, loop_outi * N + loop_stepi + 1] = 1
            self.S[loop_outi * N + N - 1, loop_outi * N + N - 1] = 1

        '''算法运行时间'''
        self.costtime = 0

        self.solver_dmc = DMC(A, self.N, self.R, self.Q, self.M, self.P, self.m, self.p,
                                                   self.alphe,self.alphemethod)
        self.control_vector, self.dynamic_matrix_P, self.dynamic_matrix_N = self.solver_dmc.compute()

        self.solver_qp = MinJ(0, 0, 0, self.dynamic_matrix_P, self.Q, self.R, self.M, self.P, self.m, self.p, 0, 0,
                                 0, 0, self.alphe,self.alphemethod)

        self.help = Tools()

        self.PINF = 2 ** 200  # 正无穷
        self.NINF = -2 ** 200  # 负无穷
        self.funneltype = funneltype

        pass

    def predictive_control(self, y0, dmv):
        '''
            function:
                预测控制
            Args:
                :arg dmv mv增量shape(m,1)
                :arg y0 pv预测值shape(p*P,1)

            Returns:
                返回再作用
            '''
        return np.dot(self.dynamic_matrix_N, dmv) + y0

    def rolling_optimization(self, wp, y0, ypv, mv, limitmv, limitdmv, funels):
        '''
               function:
                    滚动优化
                    求解出来的dmv是会检查是否在-1*dmvmax<=dmv<=dmvmax,同时如果dmv如果比dmvmin还小，那么dmv会被赋值为0
                    也检查了mvmin<=mv+dmv<=mvmax

               Args:
                    :arg wp sp设定值shape(m,1)
                    :arg y0 对pv预测值(p*N,1)
                    :arg ypv pv的反馈值
                    :arg mv 当前的mv数值shape(m,1)
                    :arg dmv 求解器求出来的mv增量shape(m*M,1)
                    :arg limitmv shape=(m,2)[ [m1_min,m1.max],
                                              [m2.min,m2.max],
                                              .......
                                            ]

                    :arg limitdmv shape(m,2)数据排布形式如limitmv
                    :arg funels 漏斗值
                    :arg alph 柔化系数（参考轨迹系数）

               Returns:

               '''

        '''将sp值向量构建为shape=(p*P,1)'''
        # WP = self.help.biuldWi(self.p, self.P, wp,ypv,alph)
        WP = self.help.biuldWiByFunel(self.p, self.P, self.N, y0, funels)
        '''提取每个pv的前P个预测值'''
        y_0P = np.zeros((self.p * self.P, 1))
        for indexp in range(self.p):
            y_0P[indexp * self.P:(indexp + 1) * self.P, 0] = y0[indexp * self.N:indexp * self.N + self.P, 0]

        deltay = np.zeros((self.p * self.P, 1))
        deltay[:, 0] = WP.transpose() - y_0P[:, 0]

        # fig, ax = plt.subplots()
        # X = np.arange(0, self.P, 1)
        # ax.plot(X, WP[:, 0], 'k-')
        # ax.plot(X, funels[0,0:self.P ], 'g-')
        # ax.plot(X, funels[1, 0:self.P], 'r-')
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # X = np.arange(0, self.P, 1)
        # ax.plot(X, y_0P[:, 0], 'y-')
        # ax.plot(X, funels[0, 0:self.P], 'g-')
        # ax.plot(X, funels[1, 0:self.P], 'r-')
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # X = np.arange(0, self.P, 1)
        # ax.plot(X, deltay[:, 0], 'y-')
        # plt.show()

        '''
        1、先用动态矩阵求解器求解
        1.1、动态矩阵求解后，检查数据是否超过限制
        1.2检查是否超过限制：mv上下限、dmv上下限
        1.3如果超过则采用QP求解器进行求解
        2、求解器求解完成后，对dmv进行检查是否有超过限制        
        '''

        '''计算得到m个输入的M个连续的输出的deltaU'''
        dmv = np.zeros((self.m * self.M, 1))
        dmv[:, 0] = np.dot(self.control_vector, deltay[:, 0])

        isinlimit, mvmin, mvmax, dmvmin, dmvmax = self.checklimit(mv, dmv, limitmv, limitdmv)
        if (isinlimit):
            '''dmc求解器成功'''
            self.costtime = 0.1
            pass

        else:
            '''qp求解器开始运行'''
            self.solver_qp.setu0(self.help.buildU(mv, self.m, self.M))
            self.solver_qp.setwp(WP.transpose())
            self.solver_qp.sety0(y_0P[:, 0])
            self.solver_qp.setUmin(mvmin)
            self.solver_qp.setUmax(mvmax)
            self.solver_qp.setDumin(dmvmin)  # 最小值先不用,在qp求解中，使用-1*dmvmax作为最大减动幅度,最小值仅在输出时，如果abs(dmv)<dmvmin就将其累加到mv上
            self.solver_qp.setDumax(dmvmax)
            res = self.solver_qp.comput()
            dmv[:, 0] = res.x
            self.costtime = res.execution_time
            pass

        comstraindmv, firstdmv, originfristdmv = self.mvconstraint(mv, dmv, limitmv, limitdmv)

        return comstraindmv, firstdmv, originfristdmv

    def feedback_correction(self, yreal, y0, lastmvfb, thistimemvfb, lastfffb, thistimefffb, ffdependregion):
        '''
                   function:
                        反馈矫正

                   Args:
                       :arg lastmvfb 上一次mv的反馈值shape(1，m)
                       :arg thistimemvfb 本次mv的反馈值shape(1，m,)
                       :arg y0 pv预测初始化值
                       :arg yreal pv的实际值
                       :arg lastfffb 上一次前馈ff值
                       :arg thistimefffb 本次前馈反馈值
                       :arg ffdependregion 前馈ff是否在上下限内

                   Returns:
                       dff 前馈的增量
                   '''

        '''作用完成后，做预测数据计算'''
        deltay = np.dot(self.A_step_response_sequence, (thistimemvfb - lastmvfb).reshape(-1, 1))
        '''根据反馈计算出上一次mv作用后的预测曲线'''
        y_predictionN = y0 + deltay.reshape(self.p * self.N, 1)
        '''等待到下一次将要输出时候，获取实际值，并与预测值的差距'''
        '''K矩阵 只取本次预测值'''
        K = np.zeros((self.p, self.p * self.N))
        for indexp in range(self.p):
            K[indexp, indexp * self.N] = 1

        frist_y_predictionN = np.dot(K, y_predictionN)  # 提取上一个作用deltau后，第一个预测值
        # yreal[:, 0] = np.array(opcModleData['y0'])  # firstNodePredict.transpose()
        y_Real = np.zeros((self.p, 1))
        y_Real[:, 0] = yreal
        e = y_Real - frist_y_predictionN

        y_Ncor = np.zeros((self.p * self.N, 1))
        y_Ncor = y_predictionN + np.dot(self.H, e)

        y_0N = np.zeros((self.p * self.N, 1))
        y_0N = np.dot(self.S, y_Ncor)

        dff = np.zeros(1)
        if self.feedforwardNum != 0:
            dff = thistimefffb - lastfffb
            y_0N = y_0N + np.dot(self.B_step_response_sequence,
                                 ((thistimefffb - lastfffb) * ffdependregion).reshape(-1, 1))

        return e, y_0N, dff

    def feedback_correction_for_simulate(self, Origionapc, matrixPvMvMapping, Y0Position, yreal, y0, lastmvfb, thistimemvfb, lastfffb, thistimefffb, ffdependregion,funnels):
        '''
                   function:
                        反馈矫正

                   Args:
                       :arg origionA 原始未拆分的响应矩阵(p*N,1*m)
                       :arg origionB 原始未拆分的响应矩阵(p*N,1*f) or []
                       :arg matrixPvMvMapping pv与mv的映射矩阵
                       :arg Y0Position 根据映射矩阵1的数量的y0矩阵(mappingcount*N,1),每行第一个pv的位置
                       :arg p 原始pv数量
                       :arg m 原始mv数量
                       :arg N 响应序列长度
                       :arg yreal 改值为原始pv实时值(p,)
                       :arg y0 映射后的pv预测序列(mappingcount*N,1)
                       :arg lastmvfb 上一次原始mv的反馈值shape(1，m)
                       :arg thistimemvfb 本次原始mv的反馈值shape(1，m,)
                       :arg lastfffb 上一次前馈ff值
                       :arg thistimefffb 本次前馈反馈值
                       :arg ffdependregion 前馈ff是否在上下限内
                       :arg funnels 映射分解后的pv全漏斗数据为shape为(2,mampping*N)
                         funels=[ up1  up2..upN  pv的高限制都在这一行
                         donw1 donw2...donwN   pv的低限制都在这一行 ]

                   Returns:
                       dff 前馈的增量
                   '''

        '''作用完成后，做预测数据计算'''
        origionA=Origionapc.getOrigion_A_step_response_sequence()
        p=Origionapc.p
        m=Origionapc.m
        N=Origionapc.N
        origiony0=np.zeros((p*N,1))
        originalfunnels=np.zeros((2,p*N))
        indexp=0
        for position in Y0Position.astype(np.int32):
            origiony0[indexp*N:(indexp+1)*N,0]=y0[position*N:(position+1)*N,0]
            originalfunnels[:,indexp*N:(indexp+1)*N]=funnels[:,position*N:(position+1)*N]
            indexp=indexp+1

        deltay = np.dot(origionA, (thistimemvfb - lastmvfb).reshape(-1, 1))
        y_predictionN=origiony0+deltay
        '''根据反馈计算出上一次mv作用后的预测曲线'''
        #y_predictionN= y0 + deltay.reshape(self.p * self.N, 1)
        '''等待到下一次将要输出时候，获取实际值，并与预测值的差距'''
        '''K矩阵 只取本次预测值'''
        K = np.zeros((p, p * N))
        for indexp in range(p):
            K[indexp, indexp * N] = 1

        frist_y_predictionN = np.dot(K, y_predictionN)  # 提取上一个作用deltau后，第一个预测值
        # yreal[:, 0] = np.array(opcModleData['y0'])  # firstNodePredict.transpose()
        y_Real = np.zeros((p, 1))
        y_Real[:, 0] = yreal
        e = y_Real - frist_y_predictionN

        y_Ncor = np.zeros((p * N, 1))
        y_Ncor = y_predictionN + np.dot(Origionapc.H, e)

        y_0N = np.zeros((p * N, 1))
        y_0N = np.dot(Origionapc.S, y_Ncor)

        dff = np.zeros(1)
        if (Origionapc.isffNumzero==False):
            dff = thistimefffb - lastfffb
            y_0N = y_0N + np.dot(Origionapc.origion_B_step_response_sequence,
                                 ((thistimefffb - lastfffb) * ffdependregion).reshape(-1, 1))
        '''将预测的pv曲线按照映射的关系矩阵展开'''
        y0_copy=y0.copy()
        indexmapping=0
        for indexp in range(p):
            for indexm in range(m):
                if (matrixPvMvMapping[indexp][indexm] == 1):
                    y0_copy[indexmapping * N:(indexmapping + 1) * N, 0] = y_0N[indexp * N:(indexp + 1) * N, 0]
                    indexmapping = indexmapping + 1

        return e, y0_copy,y_0N, dff,originalfunnels


    def mvconstraint(self, mv, dmv, limitmv, limitdmv):
        '''
                      function:
                           检查是否有超过限制
                           1、mv上下限
                           2、dmv上下限 其中dmv下限暂时先用-1*dmvmax代替，dmvmin只是作为当dmv绝对值小于他时不进行累加到mv上，放弃本次调节

                      Args:
                          :arg mv 当前的mv数值shape(m,1)
                          :arg dmv 求解器求出来的mv增量shape(m*M,1)
                          :arg limitmv shape=(m,2)[ [m1_min,m1.max],
                                                    [m2.min,m2.max],
                                                    .......
                                                  ]

                          :arg limitdmv shape(m,2)数据排布形式如limitmv
                      Returns:
                          True mv 加上增量后还在mv上下限范围内 and dmv也在该范围内
                          False 上述条件不满足
                          origfristonedmv 经过修改的各个第一步dmv的量
                      '''
        '''dmv累加矩阵'''

        '''L矩阵 只取即时控制增量'''
        L = np.zeros((self.m, self.M * self.m))
        for indexm in range(self.m):
            L[indexm, indexm * self.M] = 1

        '''本次要输出的dmv'''
        firstonedmv = np.dot(L, dmv)
        origfristonedmv = firstonedmv.copy()
        for index, needcheckdmv in np.ndenumerate(firstonedmv):
            '''检查下dmv是否在限制之内'''
            if (np.abs(needcheckdmv) > limitdmv[index[0], 1]):
                firstonedmv[index[0], 0] = limitdmv[index[0], 1] if (firstonedmv[index[0]] > 0) else (
                            -1 * limitdmv[index[0], 1])
            '''dmv是否小于最小调节量，如果小于，则不进行调节'''
            if (np.abs(needcheckdmv) <= limitdmv[index[0], 0]):
                firstonedmv[index[0], 0] = 0
            '''mv叠加dmv完成以后是否大于mvmax'''
            if ((mv[index[0]] + firstonedmv[index[0]]) >= limitmv[index[0], 1]):
                firstonedmv[index[0], 0] = limitmv[index[0], 1] - mv[index[0]]
            '''mv叠加dmv完成以后是否小于于mvmmin'''
            if ((mv[index[0]] + firstonedmv[index[0]]) <= limitmv[index[0], 0]):
                firstonedmv[index[0], 0] = limitmv[index[0], 0] - mv[index[0]]

        return dmv, firstonedmv, origfristonedmv

    def origionmvconstraint(self, mv, thistimedmv, limitmv, limitdmv):
        '''
                      function:
                           检查是否有超过限制
                           1、mv上下限
                           2、dmv上下限 其中dmv下限暂时先用-1*dmvmax代替，dmvmin只是作为当dmv绝对值小于他时不进行累加到mv上，放弃本次调节

                      Args:
                          :arg mv 当前的mv数值shape(m,1)
                          :arg dmv 求解器求出来的mv增量shape(m,1)
                          :arg limitmv shape=(m,2)[ [m1_min,m1.max],
                                                    [m2.min,m2.max],
                                                    .......
                                                  ]

                          :arg limitdmv shape(m,2)数据排布形式如limitmv
                      Returns:
                          True mv 加上增量后还在mv上下限范围内 and dmv也在该范围内
                          False 上述条件不满足
                          origfristonedmv 经过修改的各个第一步dmv的量
                      '''
        '''dmv累加矩阵'''

        '''本次要输出的dmv'''
        firstonedmv = thistimedmv
        origfristonedmv = firstonedmv.copy()
        for index, needcheckdmv in np.ndenumerate(firstonedmv):
            '''检查下dmv是否在限制之内'''
            if (np.abs(needcheckdmv) > limitdmv[index[0], 1]):
                firstonedmv[index[0], 0] = limitdmv[index[0], 1] if (firstonedmv[index[0]] > 0) else (
                            -1 * limitdmv[index[0], 1])
            '''dmv是否小于最小调节量，如果小于，则不进行调节'''
            if (np.abs(needcheckdmv) <= limitdmv[index[0], 0]):
                firstonedmv[index[0], 0] = 0
            '''mv叠加dmv完成以后是否大于mvmax'''
            if ((mv[index[0]] + firstonedmv[index[0]]) >= limitmv[index[0], 1]):
                #        增量            =        高限        -       当前值
                firstonedmv[index[0], 0] = limitmv[index[0], 1] - mv[index[0]]
            '''mv叠加dmv完成以后是否小于mvmin'''
            if ((mv[index[0]] + firstonedmv[index[0]]) <= limitmv[index[0], 0]):
                #        增量            =        低限        -       当前值
                firstonedmv[index[0], 0] = limitmv[index[0], 0] - mv[index[0]]

        return firstonedmv, origfristonedmv

    def checklimit(self, mv, dmv, limitmv, limitdmv):
        '''
                      function:
                           检查是否有超过限制
                           1、mv上下限
                           2、dmv上下限 其中dmv下限暂时先用-1*dmvmax代替，dmvmin只是作为当dmv绝对值小于他时不进行累加到mv上，放弃本次调节

                      Args:
                          :arg mv 当前的mv数值shape(m)
                          :arg dmv 求解器求出来的mv增量shape(m*M,1)
                          :arg limitmv shape=(m,2)[ [m1_min,m1.max],
                                                    [m2.min,m2.max],
                                                    .......
                                                  ]

                          :arg limitdmv shape(m,2)数据排布形式如limitmv
                      Returns:
                          True mv 加上增量后还在mv上下限范围内 and dmv也在该范围内
                          False 上述条件不满足
                      '''
        '''dmv累加矩阵'''
        coe_accumdmv = np.zeros((self.m * self.M, self.m * self.M))
        for indexm in range(self.m):
            for noderow in range(self.M):
                for nodecol in range(self.M):
                    if (nodecol <= noderow):
                        coe_accumdmv[indexm * self.M + noderow, indexm * self.M + nodecol] = 1

        accumdmv = np.dot(coe_accumdmv, dmv.reshape(self.m * self.M, 1))

        '''叠加了增量后的mv'''
        accummv = self.help.buildU(mv, self.m, self.M) + accumdmv

        '''分解为mvmin和mvmax'''
        mvmin = np.zeros((self.m * self.M, 1))
        mvmax = np.zeros((self.m * self.M, 1))

        dmvmin = np.zeros((self.m * self.M, 1))
        dmvmax = np.zeros((self.m * self.M, 1))
        for indexm in range(self.m):
            for indexM in range(self.M):
                mvmin[indexm * self.M + indexM, 0] = limitmv[indexm, 0]
                mvmax[indexm * self.M + indexM, 0] = limitmv[indexm, 1]

                dmvmin[indexm * self.M + indexM, 0] = limitdmv[indexm, 0]
                dmvmax[indexm * self.M + indexM, 0] = limitdmv[indexm, 1]

        '''检查增量下界上界'''
        '''检查mv上下限'''
        if (((mvmin) <= accummv).all() and (mvmax >= accummv).all() and (np.abs(dmv) <= dmvmax).all()):
            return True, mvmin, mvmax, dmvmin, dmvmax
        else:
            return False, mvmin, mvmax, dmvmin, dmvmax

def main(input_data, context):
    OUT={}
    if(input_data['msgtype']=='build'):
        if 'MPC' not in context:
            context['tools']=Tools()
            context['MPC'] = apc(input_data["P"], input_data["p"], input_data["M"], input_data["m"], input_data["N"], input_data["APCOutCycle"], input_data["fnum"],
                          np.array(input_data["A"]), (np.array(input_data["B"]) if ("B" in input_data) else []), np.array(input_data["Q"]),
                          np.array(input_data["R"]), np.array(input_data['pvusemv']), np.array(input_data['alphe']),
                          np.array(input_data['funneltype']), input_data['alphemethod'])
            OUT={
                'msgtype':input_data['msgtype']
            }

    elif(input_data['msgtype']=='compute'):
            if 'y0' not in context:
                y0 = np.zeros((context['MPC'].p * context['MPC'].N, 1))
                for indexp in range(context['MPC'].p):
                    for indexn in range(context['MPC'].N):
                        y0[indexp * context['MPC'].N + indexn, 0] = np.array(input_data['y0'])[indexp]
                context['y0']=y0
                context['ypv'] = np.array(input_data['y0'])
                context['limitmv'] = np.array(input_data['limitU'])
                context['limitdmv'] = np.array(input_data['limitDU'])  # ([[0.1,0.2],[0.1,0.2]])
                context['mv'] = np.array(input_data['U'])
                context['mvfb'] = np.array(input_data['UFB'])
                context['ff'] = np.array(input_data['FF']) if ('FF' in input_data) else []  # 前馈值
                context['ffdependregion'] = np.array(input_data['FFLmt']) if (
                        'FFLmt' in input_data) else []  # 前馈置信区间，不在这个区间内的ff,不可以用
                context['wp'] = np.array(input_data['wi'])
                context['deadZones'] = np.array(input_data['deadZones'])
                context['funelInitValues'] = np.array(input_data['funelInitValues'])
                OUT = {
                    'mv': context['mv'].reshape(-1).tolist(),
                     'dmv': np.zeros_like(context['mv']).reshape(-1).tolist(),
                     'e': np.zeros_like(context['ypv']).reshape(-1).tolist(),
                    'dff': np.zeros_like(context['ff']).reshape(-1).tolist(),
                    'predict': context['y0'].reshape(-1).tolist(),
                    'funelupAnddown': np.zeros((2, context['MPC'].p * context['MPC'].N)).tolist(),
                    'msgtype': input_data['msgtype']
                }
            elif 'y0' in context:
                originalfunnels, decoratefunnels = context['tools'].buildFunel(context['wp'], context['deadZones'], context['funelInitValues'], context['MPC'].N, context['MPC'].p,
                                                                    context['MPC'].funneltype, context['MPC'].PINF, context['MPC'].NINF)
                comstraindmv, firstdmvs, originfristdmv = context['MPC'].rolling_optimization(context['wp'], context['y0'], context['ypv'], context['mv'], context['limitmv'], context['limitdmv'],
                                                                                   decoratefunnels)
                predicty0 = context['MPC'].predictive_control(context['y0'], comstraindmv)
                '''新增加死区时间和漏斗初始值'''
                writemv = []

                '''这里说明下，循环遍历每一个pv的漏斗
                如果出现不符合的则需要更新对应影响他的mv
                '''
                updatemvmat = np.zeros(context['MPC'].m)  # 需改更新的mv
                for indexp in range(context['MPC'].p):
                    if ((decoratefunnels[0, indexp * context['MPC'].N:(indexp + 1) * context['MPC'].N] >= context['y0'][
                                                                                    indexp * context['MPC'].N:(indexp + 1) * context['MPC'].N,
                                                                                    0]).all() and (
                            decoratefunnels[1, indexp * context['MPC'].N:(indexp + 1) * context['MPC'].N] <= context['y0'][indexp * context['MPC'].N:(
                                                                                                                 indexp + 1) * context['MPC'].N,
                                                                                       0]).all()):
                        pass
                    else:
                        updatemvmat = updatemvmat + context['MPC'].pvusemv[indexp]
                selectmvmat = updatemvmat > 0  # 大于0说明PV不在漏斗内，需要更新
                writemv = context['mv'] + firstdmvs.reshape(-1) * selectmvmat.astype(int).reshape(-1)


                # 这里要改成输出周期结束以后再取读取反馈值
                # time.sleep((MPC.outStep - MPC.costtime) if ((MPC.outStep - MPC.costtime) >= 0) else 0)

                e, y_0N, dff = context['MPC'].feedback_correction(np.array(input_data['y0']), context['y0'], context['mvfb'],
                                                       np.array(input_data["UFB"]),
                                                       context['ff'],
                                                       (np.array(input_data["FF"]) if (
                                                                   'FF' in input_data) else []),
                                                       (np.array(input_data["FFLmt"]) if (
                                                               'FFLmt' in input_data) else []))
                context['y0'] = y_0N

                context['limitmv'] = np.array(input_data['limitU'])  # mv限制
                context['limitdmv'] = np.array(input_data['limitDU'])  # dmv限制
                context['mv'] = np.array(input_data['U'])  # mv值
                context['mvfb'] = np.array(input_data['UFB'])  # mv反馈
                context['ff'] = np.array(input_data['FF']) if ('FF' in input_data) else []  # 前馈值
                context['ffdependregion'] = np.array(input_data['FFLmt']) if (
                        'FFLmt' in input_data) else []  # 前馈置信区间，不在这个区间内的ff,不可以用
                context['wp'] = np.array(input_data['wi'])  # sp值
                context['ypv'] = np.array(input_data['y0'])
                # realvalidekey = input_data['validekey']
                '''模型更新退出运算'''
                OUT = {
                            'mv': writemv.reshape(-1).tolist()
                            , 'dmv': firstdmvs.reshape(-1).tolist()
                            , 'e': e.reshape(-1).tolist()
                            , 'dff': dff.reshape(-1).tolist()
                            , 'predict': context['y0'].reshape(-1).tolist()
                            , 'funelupAnddown': originalfunnels.tolist()
                            ,'msgtype': input_data['msgtype']
                     }
                # write_resp = requests.post("http://localhost:8080/AILab/python/updateModleData.do", data=payload)
    return OUT
# pyinstaller -F E:\WX_PUSH_3\WX_Push\WX_Clinet_SalePush.py
