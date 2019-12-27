import numpy as np
import matplotlib.pyplot as plt
import DynamicMatrixControl

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

'''结束时间'''
tend=500

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
        temp_e=(np.exp(-1 * Wni * Zatei * (loop_t *delta-Tao[loop_node]))) / np.sqrt(1-np.power(Zatei,2))
        temp_sin=np.sin(Wdi * (loop_t *delta-Tao[loop_node]) + np.arctan(np.sqrt(1-np.power(Zatei,2)) / Zatei))
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

A=np.zeros((p,m,int(N/delta)))

'''重构阶跃响应'''
for loop_outi in range(p):
    for loop_ini in range(m):
        A[loop_outi,loop_ini,:]=Yt_result[p*loop_outi+loop_ini]#0 1 2 3

print(A[:,:,39])

'''Q Matrix'''
ywt=np.array([1,3])
'''R Matrix'''
uwt=np.array([400,300])
'''H Matrix'''
alpha=np.array([1,1])
'''Aim Matrix'''
r=np.array([1,2])
'''H Matrix'''
H=np.zeros((2*N,2))
'''位移矩阵'''
S=np.zeros((2*N,2*N))

'''build H matrix'''
for loop_outi in range(p):
    for loop_timei in range(N):
        H[loop_timei+N*loop_outi,loop_outi]=1

'''构造计算位移矩阵'''
for loop_outi in range(p):
    for loop_stepi in range(0,N-1):
        S[loop_outi*N+loop_stepi,loop_outi*N+loop_stepi+1]=1
    S[loop_outi*N+N-1,loop_outi*N+N-1]=1

'''构建R矩阵'''
R=np.zeros((p*P,1))
for loop_ri in range(P*p):
    R[loop_ri,0]=r[int(loop_ri/P)]

'''实际输出'''
y_Real=np.zeros((p,tend))
'''误差'''
e=np.zeros((p,tend))
'''期望'''
y=np.zeros((p,tend))
'''输入'''
U=np.zeros((p,tend))

y_N=np.zeros((p*N,1))
y_N0=np.zeros((p*N,1))
y_P0=np.zeros((p*P,1))

for loo_outi in range(p):
    for loop_timei in range(N):
        y_N[loo_outi*N+loop_timei,0]=0
        y_N0[loo_outi*N+loop_timei,0]=0

    for loop_Pi in range(P):
        y_P0[loo_outi*P+loop_Pi,0]=0


deltaY=np.zeros((1,tend))
deltaU=np.zeros((1,tend))
y_Ncor=np.zeros((1,tend))


dmc=DynamicMatrixControl.DMC(A,uwt,ywt,M,P,m,p,N)
dmc.compute()


print()



