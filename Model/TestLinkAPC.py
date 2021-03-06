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
    DEBUG=True
    for i in range(1, len(sys.argv)):
        strs = sys.argv[i]
        print(i)
        print(strs)
        #stringjson = '{"p":1,"P":12,"1":[{"3":[0.0,0.4151304360067797,1.176822478429792,1.590779306025315,1.4264776926660865,0.9715664675338392,0.6636080291971198,0.7046978340239161,0.96691428652255,1.1833583291582954,1.1962406597511275,1.0511694712410666,0.9055599035571009,0.8741424378067684,0.9503684965178477,1.0447759901910296,1.0780947838763422,1.0408348570641193,0.981644839331238,0.9531081048328375,0.9692863400976179,1.0052033436203236,1.0271898224847225,1.0217590754024484,1.0006928920927902,0.9848504554785261,0.9852679605682885,0.9971640670749762,1.0080340643148051,1.0096051384111164,1.0031874978182627,0.9960190291852486,0.9939463573954769,0.9972050203498327,1.0017707203955273,1.0036926839547362,1.0021824697550235,0.999368400179349,0.9978219848878931,0.9984129800783667,1.000090988409581,1.0012378717318327,1.0010965742962357,1.0001310550954547,0.9993268144975218,0.9992725589200009,0.9998055991591668,1.0003456436656775,1.0004658026258857,1.0001865065942093,0.9998369303088337,0.9997114086035236,0.9998473594328934,1.000066195931829,1.0001730129950057,1.0001144306425287,0.9999818434768685,0.9998998531347962,0.9999191234732244,0.9999967599662898,1.0000556852952884,1.0000546539582504,1.0000108968029757,0.9999705476656342,0.9999644247984766,0.9999879735182591,1.000014537281888,1.000022386862253,1.0000104701510673,0.9999935776849916,0.9999863650809752,0.9999918572434927,1.0000022530706456,1.0000080290066224,1.0000059050072585,0.9999997159043338,0.9999954452523759,0.9999959283433218,0.999999483207258,1.0000024712435802,1.000002696333106,1.000000737982725,0.9999987351965977,0.9999982761739459,0.999999299368072,1.0000005937009586,1.0000010663847057,1.000000570463732,0.9999997614425409,0.9999993616980226,0.9999995737163868,1.0000000631244375,1.0000003688338526,1.0000003005815201,1.000000014486805,0.9999997953393841,0.9999997972571651,0.9999999581921387,1.0000001079543899,1.000000131752676]}],"m":1,"M":6}'
    url='http://192.168.165.187:8080/AILab/python/modlebuild/4.do'#sys.argv[0]
    modleId=4#int(sys.argv[2])
    resp=requests.get(url)#sys.argv[0]'http://192.168.165.187:8080/AILab/python/modleparam/1.do'
    modle=json.loads(resp.text)

    '''预测时域长度'''
    P=modle["P"]#100#200 date 3/25

    '''控制时域长度'''
    M=modle["M"]#modle["M"]6 date 3/25

    '''输入个数'''
    m=modle["m"]

    '''输出个数'''
    p=modle["p"]

    '''建模时域'''
    N=modle["N"]

    '''输出间隔'''
    outStep =modle["APCOutCycle"]

    '''前馈数量'''
    feedforwardNum=modle["f"]



    unhandleff_time_series=0
    if feedforwardNum!=0:
        unhandleff_time_series=np.array(modle["B"])
    '''前馈的响应'''
    B_time_series=0
    if feedforwardNum!=0:
        B_time_series=np.zeros((p*N,feedforwardNum))
        for outi in range(p):
            for ini in range(feedforwardNum):
                B_time_series[outi*N:(outi+1)*N,ini]=unhandleff_time_series[outi,ini]


    '''时序域 Matrix'''
    qi=np.array(modle["Q"])
    '''控制域 Matrix'''
    ri=np.array(modle["R"])
    '''H Matrix误差矫正'''
    hi=np.array([1, 1])
    # '''Aim Matrix'''
    # wi=np.array([0.5, 1])
    #

    '''如果响应矩阵，'''
    A_time_series=np.array(modle["A"])#np.zeros((p, m, N.shape[1]))#这个shape[1]起始时响应的模型N,[p,m,N]


    '''H Matrix 矫正系数矩阵'''
    H=np.zeros((p*N,p))#[输出引脚*阶跃时序长度，输出引脚]
    '''位移矩阵'''
    S=np.zeros((p*N,p*N))#[输出引脚*阶跃时序长度，输出引脚*阶跃时序长度]

    '''build矫正 H matrix'''
    for loop_outi in range(p):
        for loop_timei in range(N):
            H[loop_timei+N*loop_outi,loop_outi]=1#hi[loop_outi]

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
    W_i=np.zeros((m,1))


    '''预测值'''
    y_N=np.zeros((p*N,1))

    '''实时值初始值'''
    y_0N=np.zeros((p*N, 1))

    '''抽取实时初始值序列'''
    y_0P=np.zeros((p * P, 1))

    '''输出矫正'''
    y_Ncor=np.zeros((p*N,1))



    tools=Help.Tools()


    deltaY=np.zeros((p * P, 1))
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
    '''DMC矩阵计算'''
    dmc=DynamicMatrixControl.DMC(A_time_series,R_t, Q, M, P, m, p)
    results=dmc.compute()

    minJ=QP.MinJ(0,0,0,results['A'],Q,R_t,M,P,m,p,0,0,0,0)
    isfirst=True

    '''上一次MV的反馈FBU'''
    lastTimeMVFB=np.zeros(m)
    '''本次MV更新'''
    thisTimeMVFB=np.zeros(m)

    '''本次前馈'''
    thisTimeFF=np.zeros(feedforwardNum)
    '''上次前馈'''
    lastTimeFF = np.zeros(feedforwardNum)

    Costtime=0
    isEnable=modle["enable"]
    while(isEnable==0):
        resp_opc=requests.get("http://192.168.165.187:8080/AILab/python/opcread/%d.do" % modleId)
        opcModleData=json.loads(resp_opc.text)
        # testwi=np.array(opcModleData['wi'])

        W_i=tools.biuldWi(p,P,np.array(opcModleData['wi']))#np.array([800])




        '''限制输入  Umin<U<Umax'''
        limitU =np.array(opcModleData["limitU"]) #np.array(opcModleData["limitU"])#np.array([[0, 100], [0, 100]])
        U[:,0]=np.array(opcModleData["U"])# 上一次MV给定量

        thisTimeMVFB=np.array(opcModleData["UFB"])# 作用mv之前的MV反馈

        '''分解为Umin和Umax'''
        Umin = np.zeros((m * M, 1))
        Umax = np.zeros((m * M, 1))
        for indexIn in range(m):
            for nodein in range(M):
                Umin[indexIn * M + nodein, 0] = limitU[indexIn, 0]
                Umax[indexIn * M + nodein, 0] = limitU[indexIn, 1]

        # '''限制输出Ymin<Y<Ymax'''
        # limitY = np.array(opcModleData["limitY"])#np.array([[0, 100], [0, 100]])
        # '''分解为Ymin和Ymax'''
        # Ymin = np.zeros((p * P, 1))
        # Ymax = np.zeros((p * P, 1))
        # for indexIn in range(p):
        #     for nodein in range(P):
        #         Ymin[indexIn * P + nodein, 0] = limitY[indexIn, 0]
        #         Ymax[indexIn * P + nodein, 0] = limitY[indexIn, 1]

        '''这里先开始输出原先的输出值U,deltaU=0 U(k)=U(k-1)+deltaU'''
        '''加上前馈'''
        if feedforwardNum!=0:
            if isfirst:
                thisTimeFF = np.array(opcModleData["FF"])
                y_0N=tools.buildY0(p, N, opcModleData['y0'])#+np.dot(B_time_series, np.array(opcModleData["deltff"]).transpose()).reshape(1,-1).T
                isfirst=False
            else:
                thisTimeFF = np.array(opcModleData["FF"])
                y_0N = y_0N + np.dot(B_time_series,((thisTimeFF-lastTimeFF)*(np.array(opcModleData["FFLmt"]))).transpose()).reshape(1,-1).T
                if DEBUG:
                    print("B")
                    #print(np.dot(B_time_series, (thisTimeFF-lastTimeFF).transpose()).reshape(1, -1).T)
                    print("y0N")
                    #print( y_0N)
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
        deltaY[:, 0] = W_i.transpose() - y_0P[:, 0]

        '''计算得到m个输入的M个连续的输出的deltaU'''
        deltaU[:, 0] = np.dot(results['deltau'], deltaY[:, 0])
        '''校验输入值是否超过限制'''
        B_Matrix = np.zeros((m * M, m * M))
        for indexIn in range(m):
            for noderow in range(M):
                for nodecol in range(M):
                    if (nodecol <= noderow):
                        B_Matrix[indexIn * M + noderow, indexIn * M + nodecol] = 1
        willUM=tools.buildU(np.array(opcModleData["U"]), m, M)+np.dot(B_Matrix,deltaU[:, 0].reshape(m*M,1))

        '''检查增量下界上界'''
        if((Umin<=willUM).all() and (Umax>=willUM).all()):
            if DEBUG:
                print("good U limit")
            Costtime=0
            pass
        else:
            if DEBUG:
                print("这里进行约束,因为U不满足")

            minJ.setu0(tools.buildU(U[:,0],m,M))
            minJ.setwp(W_i.transpose())
            minJ.sety0(y_0P[:,0])
            minJ.setUmin(Umin)
            minJ.setUmax(Umax)
            # minJ.setYmin(Ymin)
            # minJ.setYmax(Ymax)
            res=minJ.comput()
            deltaU[:, 0]=res.x
            Costtime=res.execution_time
            if DEBUG:
                print( deltaU[:, 0])

        if DEBUG:
            print("Costtime")
            print(Costtime)

        # '''得到m个输入的本次作用增量'''
        # thisTimedelU=np.dot(L, deltaU[:,0])
        # '''加上本次增量的系统输入'''
        # U[:,0]=U[:,0]+thisTimedelU.transpose()
        thisTimedelU=np.zeros((m,1))
        '''新增加死区时间和漏斗初始值'''
        linesUpAndDown=tools.buildFunel(np.array(opcModleData['wi']), np.array(opcModleData['deadZones']), np.array(opcModleData['funelInitValues']), N, p)
        if DEBUG:
            # fig, ax1 = plt.subplots()
            # PPP=np.arange(0,N)
            # ax1.plot(PPP,linesUpAndDown[0,:],'-r')
            # ax1.plot(PPP,linesUpAndDown[1,:],'-k')
            # ax1.plot(PPP,y_0N,'-g')
            pass

        if((linesUpAndDown[0,:]>=y_0N).all() and (linesUpAndDown[1,:]<=y_0N).all()):
            if DEBUG:
                print("不进行更新")
            pass
        else:
            '''得到m个输入的本次作用增量'''
            thisTimedelU = np.dot(L, deltaU[:, 0])
            for index,needcheckdmv in np.ndenumerate(thisTimedelU):
                if(np.abs(needcheckdmv)>0.2):
                    thisTimedelU[index]=0.2 if (thisTimedelU[index]>0) else -0.2

            # if thisTimedelU
            '''加上本次增量的系统输入'''
            U[:, 0] = U[:, 0] + thisTimedelU.transpose()
            if DEBUG:
                print("u", U[:, 0])
                print("unhandle",np.dot(L, deltaU[:, 0]))
                print("deltau", thisTimedelU.transpose())


        payload = {'id': modleId, 'U': U[:,0]}
        write_resp=requests.get("http://192.168.165.187:8080/AILab/python/opcwrite.do",params=payload)
        if DEBUG:
            print(write_resp.text)

        #这里要改成输出周期结束以后再取读取反馈值
        time.sleep((outStep-Costtime) if ((outStep-Costtime)>= 0) else 0)#third=outStep-Costtime if outStep-Costtime>= 0 else 0  判断是否经过QP

        resp_opc = requests.get("http://192.168.165.187:8080/AILab/python/opcread/%d.do" % modleId)
        opcModleData = json.loads(resp_opc.text)
        isEnable = opcModleData["enable"]

        lastTimeMVFB=thisTimeMVFB
        thisTimeMVFB=opcModleData['UFB']
        if feedforwardNum!=0:
            lastTimeFF=thisTimeFF
            thisTimeFF=opcModleData['FF']

        #也就是挪到这里，读取反馈值
        '''作用完成后，做预测数据计算'''
        tempY=np.dot(A_N, (thisTimeMVFB - lastTimeMVFB).transpose())
        y_predictionN = y_0N[:, 0] +tempY   # 反馈
        '''等待到下一次将要输出时候，获取实际值，并与预测值的差距'''
        firstNodePredict = np.dot(K, y_predictionN)  # 提取上一个作用deltau后，第一个预测值

        y_Real[:,0]=np.array(opcModleData['y0'])# firstNodePredict.transpose()
        e=y_Real[:,0]-firstNodePredict

        if DEBUG:
            print("e")
            print(e)

        y_Ncor[:, 0] = y_predictionN + np.dot(H, e.transpose())
        y_0N[:, 0] = np.dot(S, y_Ncor[:, 0])
        payload = {'id': modleId
                    , 'data': json.dumps(
                                            {'mv': U[:, 0].tolist()
                                                , 'dmv': np.dot(L, deltaU[:, 0]).reshape(-1).tolist()
                                                , 'e': e.tolist()
                                                , 'predict': y_predictionN.tolist()
                                                ,'funelupAnddown': linesUpAndDown.tolist()
                                             }
                                        )
                   }
        write_resp = requests.post("http://192.168.165.187:8080/AILab/python/updateModleData.do", data=payload)
        if DEBUG:
            print(write_resp.text)
            print(write_resp.elapsed.total_seconds())
    # fig, ax1 = plt.subplots()
    # PPP=np.arange(0,tend)
    # ax1.plot(PPP,y_Real[0,:],'-b')
    # ax1.plot(PPP,y_Real[1,:],'-r')
    # ax1.plot(PPP,U[0,:],'-k')
    # ax1.plot(PPP,U[1,:],'-g')
    # plt.show()




