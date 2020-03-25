import numpy as np
import matplotlib.pyplot as plt
import DynamicMatrixControl
import Help
import QP
import sys
import requests
import json
import time

if __name__ == '__main__':
    #stringjson = '{"p":1,"P":12,"1":[{"3":[0.0,0.4151304360067797,1.176822478429792,1.590779306025315,1.4264776926660865,0.9715664675338392,0.6636080291971198,0.7046978340239161,0.96691428652255,1.1833583291582954,1.1962406597511275,1.0511694712410666,0.9055599035571009,0.8741424378067684,0.9503684965178477,1.0447759901910296,1.0780947838763422,1.0408348570641193,0.981644839331238,0.9531081048328375,0.9692863400976179,1.0052033436203236,1.0271898224847225,1.0217590754024484,1.0006928920927902,0.9848504554785261,0.9852679605682885,0.9971640670749762,1.0080340643148051,1.0096051384111164,1.0031874978182627,0.9960190291852486,0.9939463573954769,0.9972050203498327,1.0017707203955273,1.0036926839547362,1.0021824697550235,0.999368400179349,0.9978219848878931,0.9984129800783667,1.000090988409581,1.0012378717318327,1.0010965742962357,1.0001310550954547,0.9993268144975218,0.9992725589200009,0.9998055991591668,1.0003456436656775,1.0004658026258857,1.0001865065942093,0.9998369303088337,0.9997114086035236,0.9998473594328934,1.000066195931829,1.0001730129950057,1.0001144306425287,0.9999818434768685,0.9998998531347962,0.9999191234732244,0.9999967599662898,1.0000556852952884,1.0000546539582504,1.0000108968029757,0.9999705476656342,0.9999644247984766,0.9999879735182591,1.000014537281888,1.000022386862253,1.0000104701510673,0.9999935776849916,0.9999863650809752,0.9999918572434927,1.0000022530706456,1.0000080290066224,1.0000059050072585,0.9999997159043338,0.9999954452523759,0.9999959283433218,0.999999483207258,1.0000024712435802,1.000002696333106,1.000000737982725,0.9999987351965977,0.9999982761739459,0.999999299368072,1.0000005937009586,1.0000010663847057,1.000000570463732,0.9999997614425409,0.9999993616980226,0.9999995737163868,1.0000000631244375,1.0000003688338526,1.0000003005815201,1.000000014486805,0.9999997953393841,0.9999997972571651,0.9999999581921387,1.0000001079543899,1.000000131752676]}],"m":1,"M":6}'
    url='http://192.168.165.187:8080/AILab/python/modlebuild/1.do'#sys.argv[0]
    modleId=1#sys.argv[1]
    resp=requests.get(url)#sys.argv[0]'http://localhost:8080/python/modleparam/1.do'
    modle=json.loads(resp.text)
    #mvtime=np.array(modle["mv"])
    # x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    # ap=x0[:-1]
    # a=np.eye(4)
    # print(a[0:2,:])

    '''预测时域长度'''
    P=200#modle["P"]

    '''控制时域长度'''
    M=6#modle["M"]

    '''输入个数'''
    m=modle["m"]

    '''输出个数'''
    p=modle["p"]

    '''建模时域'''
    N=modle["N"]

    # '''采样时间'''
    # delta=1
    #
    # '''滞后时间'''
    # Tao=[0,0,0,0]#[5,2,12,20]

    '''前馈数量'''
    feedforwardNum=modle["f"]
    unhandleff_time_series=0;
    if feedforwardNum!=0:
        unhandleff_time_series=np.array(modle["ff"])
    '''前馈的响应'''
    B_time_series=0
    if feedforwardNum!=0:
        B_time_series=np.zeros((p*N,feedforwardNum))
        for outi in range(p):
            for ini in range(feedforwardNum):
                B_time_series[outi*N:(outi+1)*N,ini]=unhandleff_time_series[outi,ini]



    '''前馈变动赋值'''
    # delta_v=np.zeros((feedforwardNum,100))
    # delta_v[0,:]=np.arange(0,100)*0.001#仅用于测试
    # delta_v[1,:]=np.arange(0,100)*0.001#仅用于测试

    #
    #
    '''时序域 Matrix'''
    qi=np.array([990, 1])
    '''控制域 Matrix'''
    ri=np.array([1000000, 300])
    '''H Matrix'''
    hi=np.array([1, 1])
    # '''Aim Matrix'''
    # wi=np.array([0.5, 1])
    #

    '''如果响应矩阵，'''
    A_time_series=np.array(modle["mv"])#np.zeros((p, m, N.shape[1]))#这个shape[1]起始时响应的模型N,[p,m,N]


    '''H Matrix'''
    H=np.zeros((p*N,p))#[输出引脚*阶跃时序长度，输出引脚]
    '''位移矩阵'''
    S=np.zeros((p*N,p*N))#[输出引脚*阶跃时序长度，输出引脚*阶跃时序长度]

    '''build矫正 H matrix'''
    for loop_outi in range(p):
        for loop_timei in range(N):
            H[loop_timei+N*loop_outi,loop_outi]=hi[loop_outi]

    '''构造计算位移矩阵S'''
    for loop_outi in range(p):
        for loop_stepi in range(0,N-1):
            S[loop_outi*N+loop_stepi,loop_outi*N+loop_stepi+1]=1
        S[loop_outi*N+N-1,loop_outi*N+N-1]=1

    '''得到R矩阵 优化时间区域'''
    R_t=np.eye((M*m))
    for loop_ini in range(m):
        R_t[M*loop_ini:M*(loop_ini+1),:]= ri[loop_ini] * R_t[M * loop_ini:M * (loop_ini + 1), :]


    '''控制优化区域Q矩阵'''
    Q=np.eye(p*P)
    for loop_ini in range(p):
        Q[P*loop_ini:P*(loop_ini+1),:]= qi[loop_ini] * Q[P * loop_ini:P * (loop_ini + 1), :]

    '''构建目标矩阵矩阵'''
    W_i=0


    '''预测值'''
    y_N=np.zeros((p*N,1))

    '''实时值'''
    y_0N=np.zeros((p*N, 1))

    '''抽取实时序列'''
    y_0P=np.zeros((p * P, 1))

    '''输出矫正'''
    y_Ncor=np.zeros((p*N,1))



    tools=Help.Tools()





    # for loo_outi in range(p):
    #     for loop_timei in range(N):
    #         y_N[loo_outi*N+loop_timei,0]=0
    #         y_0N[loo_outi * N + loop_timei, 0]=0
    #
    #     for loop_Pi in range(P):
    #         y_0P[loo_outi * P + loop_Pi, 0]=0

    deltaD=np.zeros((p * P, 1))
    deltaU=np.zeros((m*M,1))#np.zeros((m,tend))

    '''实际输出'''
    y_Real=np.zeros((p,1))

    '''输出'''
    U=np.zeros((m,1))

    '''L矩阵 只取即时控制增量'''
    L=np.zeros((m,M*m))
    for loopouti in range(m):
        L[loopouti,loopouti*M]=1

    '''K矩阵 只取本次预测值'''
    K=np.zeros((p,p*N))
    for loopouti in range(p):
        K[loopouti,loopouti*N]=1

    A_N=np.zeros((p * N, m))
    for loop_outi in range(p):
        for loop_ini in range(m):
            A_N[N * loop_outi:N * (loop_outi + 1), loop_ini]= A_time_series[loop_outi, loop_ini, :]

    #'''前馈响应矩阵赋值'''
    #B_time_series=A_N*-0.1

    #print(A_time_series.shape[2])
    dmc=DynamicMatrixControl.DMC(A_time_series,R_t, Q, M, P, m, p)
    results=dmc.compute()

    minJ=QP.MinJ(0,0,0,results['A'],Q,R_t,M,P,m,p,0,0,0,0)
    isfirst=True
    while(True):
        resp_opc=requests.get("http://192.168.165.187:8080/AILab/python/opcread/%d.do" % modleId)
        opcModleData=json.loads(resp_opc.text)
        testwi=np.array(opcModleData['wi'])
        W_i=tools.biuldWi(p,P,np.array(opcModleData['wi']))#np.array([800])
        '''限制输入  Umin<U<Umax'''
        limitU =np.array(opcModleData["limitU"]) #np.array(opcModleData["limitU"])#np.array([[0, 100], [0, 100]])

        U[:,0]=np.array(opcModleData["U"])
        '''分解为Umin和Umax'''
        Umin = np.zeros((m * M, 1))
        Umax = np.zeros((m * M, 1))
        for indexIn in range(m):
            for nodein in range(M):
                Umin[indexIn * M + nodein, 0] = limitU[indexIn, 0]
                Umax[indexIn * M + nodein, 0] = limitU[indexIn, 1]

        '''限制输出Ymin<Y<Ymax'''
        limitY = np.array(opcModleData["limitY"])#np.array([[0, 100], [0, 100]])
        '''分解为Ymin和Ymax'''
        Ymin = np.zeros((p * P, 1))
        Ymax = np.zeros((p * P, 1))
        for indexIn in range(p):
            for nodein in range(P):
                Ymin[indexIn * P + nodein, 0] = limitY[indexIn, 0]
                Ymax[indexIn * P + nodein, 0] = limitY[indexIn, 1]

        '''这里先开始输出原先的输出值U,deltaU=0 U(k)=U(k-1)+deltaU'''
        '''加上前馈'''
        if feedforwardNum!=0:
            if isfirst:
                y_0N=tools.buildY0(p, N, opcModleData['y0'])+np.dot(B_time_series, np.array(opcModleData["deltff"]).transpose()).reshape(1,-1).T
                isfirst=False
            else:
                y_0N = tools.buildY0(p, N, y_0N) + np.dot(B_time_series,np.array(opcModleData["deltff"]).transpose()).reshape(1,-1).T
        else:
            if isfirst:
                y_0N=tools.buildY0(p,N,np.array(opcModleData['y0']))
                isfirst=False
            else:
                pass
        '''输出以后，先计算下数据的在这个deltaU的作用下，预测下1~N个时刻的数据'''
        for pull_away_M in range(p):
            y_0P[pull_away_M * P:(pull_away_M + 1) * P, 0]= y_0N[pull_away_M * N:(pull_away_M) * N+ P, 0]
        '''计算deltaD'''
        deltaD[:, 0] = W_i.transpose() - y_0P[:, 0]

        '''计算得到m个输入的M个连续的输出的deltaU'''
        deltaU[:, 0] = np.dot(results['deltau'], deltaD[:, 0])
        '''校验输入值是否超过限制'''
        willUM=tools.buildU(np.array(opcModleData["U"]), m, M)+deltaU[:, 0].reshape(m*M,1)

        '''检查增量下界上界'''
        if((Umin<=willUM).all() and (Umax>=willUM).all()):
            print("good U limit")
            willYP = np.dot(results['A'], deltaU[:, 0].reshape(m * M, 1))+y_0P[:,0]
            if ((Ymin <= willYP).all() and (willYP <= Ymax).all()):
                print("good Y limit")
                pass
            else:
                '''这里需要进行约束'''
                print("这里需要进行约束，因为Y不满足")
                minJ.setu0(tools.buildU(U[:, 0], m, M))
                minJ.setwp(W_i.transpose())
                minJ.sety0(y_0P[:, 0])

                minJ.setUmin(Umin)
                minJ.setUmax(Umax)
                minJ.setYmin(Ymin)
                minJ.setYmax(Ymax)

                aaaa = minJ.comput()
                deltaU[:, 0] = aaaa
                pass
        else:
            print("这里进行约束,因为U不满足")
            # for ii in range(m):
            #     if(Umin[ii*M+0,0] >= willUM[ii*M+0,0]):
            #         print("overminU", deltaU)
            #         print("willUM", willUM)
            #         print("willUM", Umin)
            #         print("U",tools.buildU(np.array(opcModleData["U"]), m, M))
            #         deltaU[ii*M+0, 0]=Umin[ii * M + 0, 0]-tools.buildU(np.array(opcModleData["U"]), m, M)[ii*M+0,0]
            #         print("overminU",deltaU)
            #
            #     elif Umax[ii*M+0,0] <= willUM[ii*M+0,0]:
            #         deltaU[ii*M+0:, 0]=Umax[ii*M+0,0]-tools.buildU(np.array(opcModleData["U"]), m, M)[ii*M+0,0]
            #         print("overmaxU", deltaU)

            minJ.setu0(tools.buildU(U[:,0],m,M))
            minJ.setwp(W_i.transpose())
            minJ.sety0(y_0P[:,0])
            minJ.setUmin(Umin)
            minJ.setUmax(Umax)
            minJ.setYmin(Ymin)
            minJ.setYmax(Ymax)

            aaaa=minJ.comput()
            deltaU[:, 0]=aaaa
            print(aaaa)

        '''得到m个输入的本次作用增量'''
        thisTimedelU=np.dot(L, deltaU[:,0])
        '''加上本次增量的系统输入'''
        print("U0",U[:,0])
        U[:,0]=U[:,0]+thisTimedelU.transpose()#这个里需要校验是否满足约束
        payload = {'id': modleId, 'U': U[:,0]}
        write_resp=requests.get("http://192.168.165.187:8080/AILab/python/opcwrite.do",params=payload)
        print(write_resp.text)
        print("u",U[:,0])
        print("", thisTimedelU.transpose())
        '''作用完成后，做预测数据计算'''
        y_predictionN= y_0N[:, 0] + np.dot(A_N, thisTimedelU.transpose())
        '''等待到下一次将要输出时候，获取实际值，并与预测值的差距'''
        firstNodePredict=np.dot(K, y_predictionN)#提取上一个作用deltau后，第一个预测值

        time.sleep(1)

        resp_opc = requests.get("http://192.168.165.187:8080/AILab/python/opcread/%d.do" % modleId)
        opcModleData = json.loads(resp_opc.text)

        y_Real[:,0]=np.array(opcModleData['y0'])#firstNodePredict.transpose()#这里为了模拟，先把他赋值给

        e=y_Real[:,0]-firstNodePredict

        y_Ncor[:, 0] = y_predictionN + np.dot(H, e.transpose())
        y_0N[:, 0] = np.dot(S, y_Ncor[:, 0])



    fig, ax1 = plt.subplots()
    PPP=np.arange(0,tend)
    ax1.plot(PPP,y_Real[0,:],'-b')
    ax1.plot(PPP,y_Real[1,:],'-r')
    ax1.plot(PPP,U[0,:],'-k')
    ax1.plot(PPP,U[1,:],'-g')
    plt.show()




