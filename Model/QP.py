import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import SR1
from scipy.optimize import Bounds
class MinJ:
    '''
    deltav前馈增量
    '''
    def __init__(self,wp,y0,u0,A,Q,R,M,P,m,p,Umin,Umax,Ymin,Ymax):
        self.wp=wp
        self.y0=y0
        self.u0=u0
        self.A=A
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
        #self.deltav=deltav
        #self.B_N=B_timeserial#数据矩阵形式(输出个数*响应序列长度，前馈个数)
        '''B矩阵，用于计算M个增量的变化'''
        self.B = np.zeros((m * M, m * M))
        for indexIn in range(m):
            for noderow in range(M):
                for nodecol in range(M):
                    if (nodecol <= noderow):
                        self.B[indexIn * M + noderow, indexIn * M + nodecol] = 1
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

    def J(self,delu):

        e=(self.wp - (self.y0 + np.dot(self.A, delu))).transpose()
        aa=np.dot(np.dot(e.transpose(), self.Q), e) + np.dot(np.dot(delu.transpose(), self.R), delu)
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
        will=self.y0+np.dot(self.A,deltu)
        gradient=2 * np.dot(np.dot(-1 * self.A.transpose(), self.Q), will) + 2 * np.dot(self.R, deltu)
        return gradient

    def hesionJ(self,deltu):
        H=(2 * np.dot(np.dot(self.A.transpose(), self.Q), self.A) + 2 * self.R)
        return H.transpose()


    def comput(self):
        #bounds = Bounds([-10, 10], [-10, 10])
        #print("delUmin")
        #print((self.Umin - self.u0).transpose()[0,:].shape)
        linear_constraintu = LinearConstraint(self.B , (self.Umin-self.u0).transpose()[0,:],(self.Umax-self.u0).transpose()[0,:])
        #linear_constrainty = LinearConstraint(self.A,(self.Ymin-self.y0).transpose()[0,:],(self.Ymax-self.y0).transpose()[0,:])
        x0=np.zeros(self.M*self.m).transpose()
        res = minimize(self.J, x0, method='trust-constr',
                       # jac=self.gradientJ,
                       # hess=self.hesionJ,
                       jac="2-point", hess=SR1(),
                       constraints=[linear_constraintu],
                       #constraints=[linear_constraintu,linear_constrainty],
                       options={'verbose': 1,'disp': True}
                       )

        return res
