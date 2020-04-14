import numpy as np
import matplotlib.pyplot as plt
import DynamicMatrixControl
import Help
import QP


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
ap=x0[:-1]
a=np.eye(4)
print(a[0:2,:])

'''预测时域长度'''
P=12

'''控制时域长度'''
M=6

'''输入个数'''
m=2

'''输出个数'''
p=2

'''建模时域'''
N=40

'''采样时间'''
delta=1

'''滞后时间'''
Tao=[0,0,0,0]#[5,2,12,20]

'''前馈数量'''
feedforwardNum=2
'''前馈的响应'''
B_time_series=0
'''前馈变动赋值'''
delta_v=np.zeros((feedforwardNum,100))
delta_v[0,:]=np.arange(0,100)*0.001#仅用于测试
delta_v[1,:]=np.arange(0,100)*0.001#仅用于测试
'''多mv输出平衡'''
balance=0



'''时序域 Matrix'''
qi=np.array([3, 1])
'''控制域 Matrix'''
ri=np.array([400, 300])
'''H Matrix'''
hi=np.array([1, 1])
'''Aim Matrix'''
wi=np.array([0.5, 1])

'''结束时间'''
tend=100
'''参数获得'''
num=np.array([5,6,3,9])#di
den=np.array([[3,1,3],[2,1,7],[1,2,5],[2,3,6]])#ai bi ci
k_wn_zata=np.zeros((4,3))

#request kp wn and epsn
for loop_c in range(0,4):
        ai=den[loop_c,0]
        bi=den[loop_c,1]
        ci=den[loop_c,2]
        di=num[loop_c]
        print("ai= "+str(ai)+" bi= "+str(bi)+" ci= "+str(ci)+" di= "+str(di))
        '''kp=di/ci'''
        k_wn_zata[loop_c, 0] =di/ci
        '''zeta=(bi)*(sqrt(ci/4*ai*ci))'''
        k_wn_zata[loop_c, 2] = bi*np.sqrt(1/(4*ai*ci))
        '''wn=sqrt(ai/ci)'''
        k_wn_zata[loop_c, 1]=np.sqrt(ai/ci)
print(k_wn_zata)

'''计算预测时间序列'''
Yt_result=np.zeros((4,int(N*(1/delta))))
for loop_node in range(4):
    Kpi = k_wn_zata[loop_node, 0]
    Wni = k_wn_zata[loop_node, 1]
    Zatei = k_wn_zata[loop_node, 2]
    Wdi = np.sqrt(1 - np.power(Zatei, 2)) * Wni
    for loop_t in range(0,int(N*(1/delta)),1):
        temp_e=(np.exp(-1 * Wni * Zatei * ((loop_t+1) *delta-Tao[loop_node]))) / np.sqrt(1-np.power(Zatei,2))
        temp_sin=np.sin(Wdi * ((loop_t+1) *delta-Tao[loop_node]) + np.arctan(np.sqrt(1-np.power(Zatei,2)) / Zatei))
        Yt_result[loop_node,loop_t]=Kpi*(1-temp_e*temp_sin)

'''滞后时间段内的响应切除'''
for loop_node in range(4):
    for loop_set0 in range(int(Tao[loop_node]/delta)):
        Yt_result[loop_node,loop_set0]=0

fig, ax = plt.subplots()
X=np.arange(0,N,delta)
ax.plot(X,Yt_result[0,:],'b-')
ax.plot(X,Yt_result[1,:],'g-')
ax.plot(X,Yt_result[2,:],'r-')
ax.plot(X,Yt_result[3,:],'y-')
plt.show()


'''如果响应矩阵，'''
A_time_series=np.zeros((p, m, Yt_result.shape[1]))#这个shape[1]起始时响应的模型N,[p,m,N]

'''重构阶跃响应'''
for loop_outi in range(p):
    for loop_ini in range(m):
        A_time_series[loop_outi, loop_ini, :]=Yt_result[p * loop_outi + loop_ini]#0 1 2 3









'''H Matrix'''
H=np.zeros((p*Yt_result.shape[1],p))#[输出引脚*阶跃时序长度，输出引脚]
'''位移矩阵'''
S=np.zeros((p*Yt_result.shape[1],p*Yt_result.shape[1]))#[输出引脚*阶跃时序长度，输出引脚*阶跃时序长度]

'''build矫正 H matrix'''
for loop_outi in range(p):
    for loop_timei in range(Yt_result.shape[1]):
        H[loop_timei+Yt_result.shape[1]*loop_outi,loop_outi]=1

'''构造计算位移矩阵S'''
for loop_outi in range(p):
    for loop_stepi in range(0,Yt_result.shape[1]-1):
        S[loop_outi*Yt_result.shape[1]+loop_stepi,loop_outi*Yt_result.shape[1]+loop_stepi+1]=1
    S[loop_outi*Yt_result.shape[1]+Yt_result.shape[1]-1,loop_outi*Yt_result.shape[1]+Yt_result.shape[1]-1]=1

'''得到R矩阵 优化时间区域'''
R_t=np.eye((M*m))
for loop_ini in range(m):
    R_t[M*loop_ini:M*(loop_ini+1),:]= ri[loop_ini] * R_t[M * loop_ini:M * (loop_ini + 1), :]


'''控制优化区域Q矩阵'''
Q=np.eye(p*P)
for loop_ini in range(p):
    Q[P*loop_ini:P*(loop_ini+1),:]= qi[loop_ini] * Q[P * loop_ini:P * (loop_ini + 1), :]

'''构建目标矩阵矩阵'''
W_i=np.zeros((p * P, 1))
for loop_ri in range(P*p):
    W_i[loop_ri, 0]=wi[int(loop_ri / P)]


'''预测值'''
y_N=np.zeros((p*Yt_result.shape[1],tend))

'''实时值'''
y_0N=np.zeros((p * Yt_result.shape[1], tend))

'''抽取实时序列'''
y_0P=np.zeros((p * P, tend))

'''输出矫正'''
y_Ncor=np.zeros((p*Yt_result.shape[1],tend))

'''限制输入  Umin<U<Umax'''
limitU=np.array([[0,100],[0,100]])
'''分解为Umin和Umax'''
Umin=np.zeros((m*M,1))
Umax=np.zeros((m*M,1))
for indexIn in range(m):
    for nodein in range(M):
        Umin[indexIn*M+nodein,0]=limitU[indexIn,0]
        Umax[indexIn*M+nodein,0]=limitU[indexIn,1]

'''限制输出Ymin<Y<Ymax'''
limitY=np.array([[0,100],[0,100]])
'''分解为Ymin和Ymax'''
Ymin=np.zeros((p*P,1))
Ymax=np.zeros((p*P,1))
for indexIn in range(p):
    for nodein in range(P):
        Ymin[indexIn*P+nodein,0]=limitY[indexIn,0]
        Ymax[indexIn*P+nodein,0]=limitY[indexIn,1]

tools=Help.Tools()





for loo_outi in range(p):
    for loop_timei in range(Yt_result.shape[1]):
        y_N[loo_outi*Yt_result.shape[1]+loop_timei,0]=0
        y_0N[loo_outi * Yt_result.shape[1] + loop_timei, 0]=0

    for loop_Pi in range(P):
        y_0P[loo_outi * P + loop_Pi, 0]=0


deltaD=np.zeros((p * P, tend))
deltaU=np.zeros((m*M,tend))#np.zeros((m,tend))


'''实际输出'''
y_Real=np.zeros((p,tend))

'''输出'''
U=np.zeros((m,tend))

'''L矩阵 只取即时控制增量'''
L=np.zeros((m,M*m))
for loopouti in range(m):
    L[loopouti,loopouti*M]=1

'''K矩阵 只取本次预测增量'''
K=np.zeros((p,p*Yt_result.shape[1]))
for loopouti in range(p):
    K[loopouti,loopouti*Yt_result.shape[1]]=1

A_N=np.zeros((p * A_time_series.shape[2], m))
for loop_outi in range(p):
    for loop_ini in range(m):
        A_N[A_time_series.shape[2] * loop_outi:A_time_series.shape[2] * (loop_outi + 1), loop_ini]= A_time_series[loop_outi, loop_ini, :]

'''前馈响应矩阵赋值'''
B_time_series=A_N*-0.01

print(A_time_series.shape[2])
dmc=DynamicMatrixControl.DMC(A_time_series,R_t, Q, M, P, m, p)
results=dmc.compute()

minJ=QP.MinJ(0,0,0,results['A'],Q,R_t,M,P,m,p,Umin,Umax,Ymin,Ymax)

for time_devi in range(tend-1):
    '''这里先开始输出原先的输出值U,deltaU=0 U(k)=U(k-1)+deltaU'''
    print("time_devi")
    print(time_devi)

    '''加上前馈'''
    y_0N[: ,time_devi]+=np.dot(B_time_series,delta_v[:,time_devi+1].transpose())
    '''输出以后，先计算下数据的在这个deltaU的作用下，预测下1-N个时刻的数据'''
    for pull_away_M in range(p):
        y_0P[pull_away_M * P:(pull_away_M + 1) * P, time_devi]= y_0N[pull_away_M * Yt_result.shape[1]:(pull_away_M) * Yt_result.shape[1] + P, time_devi]
    '''计算deltaD'''
    deltaD[:, time_devi] = W_i.transpose() - y_0P[:, time_devi]

    '''计算得到m个输入的M个连续的输出的deltaU'''
    deltaU[:, time_devi] = np.dot(results['deltau'], deltaD[:, time_devi])
    '''校验输入值是否超过限制'''
    willUM=tools.buildU(U[:, time_devi], m, M)+deltaU[:, time_devi].reshape(m*M,1)




    '''检查增量下界上界'''
    if((Umin<=willUM).all() and (Umax>=willUM).all() and   False and np.std(willUM)<0.01):
        print("good U limit")
        willYP = np.dot(results['A'], deltaU[:, time_devi].reshape(m * M, 1))+y_0P[:,time_devi]
        if ((Ymin <= willYP).all() and (willYP <= Ymax).all()):
            print("good Y limit")
            pass
        else:
            '''这里需要进行约束'''
            print("这里需要进行约束，因为Y不满足")
            minJ.setu0(tools.buildU(U[:, time_devi], m, M))
            minJ.setwp(W_i.transpose())
            minJ.sety0(y_0P[:, time_devi])
            aaaa = minJ.comput()
            deltaU[:, time_devi] = aaaa
            pass
    else:
        print("这里进行约束,因为U不满足")
        minJ.setu0(tools.buildU(U[:,time_devi],m,M))
        minJ.setwp(W_i.transpose())
        minJ.sety0(y_0P[:,time_devi])

        res=minJ.comput()
        deltaU[:, time_devi]=res.x
        #print(aaaa)

    '''得到m个输入的本次作用增量'''
    thisTimedelU=np.dot(L, deltaU[:,time_devi])
    '''加上本次增量的系统输入'''
    U[:,time_devi+1]=U[:,time_devi]+thisTimedelU.transpose()#这个里需要校验是否满足约束
    '''作用完成后，做预测数据计算'''
    y_predictionN= y_0N[:, time_devi] + np.dot(A_N, thisTimedelU.transpose())
    '''等待到下一次将要输出时候，获取实际值，并与预测值的差距'''
    firstNodePredict=np.dot(K, y_predictionN)#提取上一个作用deltau后，第一个预测值


    y_Real[:,time_devi+1]=firstNodePredict.transpose()#这里为了模拟，先把他赋值给

    e=y_Real[:,time_devi+1]-firstNodePredict

    y_Ncor[:, time_devi+1] = y_predictionN + np.dot(H, e.transpose())
    y_0N[:, time_devi+1] = np.dot(S, y_Ncor[:, time_devi+1])








    # e[:,time_devi]=y_Real[:,time_devi]-y[:,time_devi]
    # y_Ncor[:,time_devi]=y_N[:,time_devi]+np.dot(H,e[:,time_devi])
    # y_0N[:, time_devi]=np.dot(S, y_Ncor[:, time_devi])
    # for pull_away_M in range(p):
    #     y_0P[pull_away_M * P:(pull_away_M + 1) * P, time_devi]= y_0N[pull_away_M * N:(pull_away_M) * N + P, time_devi]
    # deltaD[:, time_devi] = W_i.transpose() - y_0P[:, time_devi]
    # deltaU[:, time_devi]=np.dot(results['D'], deltaD[:, time_devi])
    # U[:,time_devi+1]=U[:,time_devi]+deltaU[:,time_devi]
    # y_N[:,time_devi+1]= y_0N[:, time_devi] + np.dot(A_N, deltaU[:, time_devi])
    # for loopouti in range(p):
    #     y[loopouti,time_devi+1]=y_N[loopouti*N,time_devi+1]
    #
    # y_Real=y


fig, ax1 = plt.subplots()
PPP=np.arange(0,tend)
ax1.plot(PPP,y_Real[0,:],'-b')
ax1.plot(PPP,y_Real[1,:],'-r')
ax1.plot(PPP,U[0,:],'-k')
ax1.plot(PPP,U[1,:],'-g')
plt.show()




