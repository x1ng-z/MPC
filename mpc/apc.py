import numpy as np
import DynamicMatrixControl
import QP
import Help
#import matplotlib.pyplot as plt


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

        self.solver_dmc = DynamicMatrixControl.DMC(A, self.N, self.R, self.Q, self.M, self.P, self.m, self.p,
                                                   self.alphe,self.alphemethod)
        self.control_vector, self.dynamic_matrix_P, self.dynamic_matrix_N = self.solver_dmc.compute()

        self.solver_qp = QP.MinJ(0, 0, 0, self.dynamic_matrix_P, self.Q, self.R, self.M, self.P, self.m, self.p, 0, 0,
                                 0, 0, self.alphe,self.alphemethod)

        self.help = Help.Tools()

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
