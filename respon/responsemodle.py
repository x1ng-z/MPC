import numpy as np
import xlrd
import xlwt
import matplotlib.pyplot as plt


def modle(k,T,tao,time):
    if time<=tao:
        return 0
    if time>tao:
        return k*(1-np.exp(-(time-tao)/T))

def respons(k,T,tao,N):
    result=np.zeros((N,1))
    for time in range(N):
        result[time,0]=modle(k, T, tao, time)
    return result

def buildtotalrespon(k,T,tao,timeseris):
    dmvs=(np.random.rand(timeseris)-0.5)*2
    delay=np.zeros(dmvs.size)
    resp=np.zeros(timeseris)
    for indexi, dmv in enumerate(dmvs):
        delay[indexi]=tao+indexi
    for t in range(timeseris):

        for m in range(dmvs.size):
            resp[t]+=modle(k,T,delay[m],t)*dmvs[m]
    return resp,dmvs
    pass



def manuldata(k,T,tao,N):
    resp=respons(k,T,tao,N)

    y0=resp
    yreal=np.zeros(2000)
    mv=0.1*np.ones((1,N))
    '''位移矩阵'''
    # S = np.zeros((1 * 2000, 1 * 2000))  # [输出引脚*阶跃时序长度，输出引脚*阶跃时序长度]

    # '''构造计算位移矩阵S'''
    # for loop_outi in range(1):
    #     for loop_stepi in range(0, N - 1):
    #         S[loop_outi * N + loop_stepi, loop_outi * N + loop_stepi + 1] = 1
    #     S[loop_outi * N + N - 1, loop_outi * N + N - 1] = 1
    for index in range(2000-N-1):
        temp=np.dot(mv,resp.reshape(-1,1))
        yreal[N+1+index] =  temp
        # y0=y0.reshape(-1, 1) + resp.reshape(-1, 1) * (0.001)#
        # yreal[index]=(y0)[0]
        # if index==0:
        #     mv[index]=(0.001)
        # else:
        #     mv[index] =mv[index-1]+ (0.001)
        # y0=np.dot(S,y0)

    return  yreal,mv
    pass
#excel数据读取
def readexceldata(workBook, row):
    sheet1 = workBook.sheet_by_name('Sheet1')
    rows = sheet1.row_values(row+1)
    return rows
    pass

def readmanuldata(yreal,mv,row):
    rows=np.zeros(2,float)
    rows[0] = yreal[row]
    rows[1]=mv[row]
    return rows
    pass

#数据构建 构建是fi
def builmanulddata( yreal,mv,start,N):
    '''
    :param N:数据响应个数
    :return:
    '''
    fi=np.zeros((N,1))
    deltayi= yreal[start]#readmanuldata( yreal,mv, start)[0] - readmanuldata(yreal,mv, start - 1)[0]
    for indexi in range(N):
        fi[indexi,0]= mv[start-1-indexi]#readmanuldata(yreal,mv, start - 1 - indexi)[1] - readmanuldata(yreal,mv, start - 1 - indexi - 1)[1]
    return deltayi,fi
    pass


def genermanuldata(start,sample,timeserse,k,T,tao,N):
    # yreal,mv = manuldata(k,T,tao,N)
    resul, dmv = buildtotalrespon(k, T, tao, timeserse)

    deltaYs=np.zeros((sample,1))
    deltaFs=np.zeros((sample,N))
    for indexi in range(sample):
        deltayi, fi = builmanulddata( resul,dmv,start+indexi, N)
        deltaYs[indexi]=deltayi
        deltaFs[indexi,:]=fi.reshape(1,-1)
        print(indexi)
    return deltaYs,deltaFs
    pass


#数据构建 构建是fi
def builddata(workBook,start,N):
    '''
    :param N:数据响应个数
    :return:
    '''
    fi=np.zeros((N,1))
    deltayi= readexceldata(workBook, start)[0] - readexceldata(workBook, start - 1)[0]
    for indexi in range(N):
        fi[indexi,0]= readexceldata(workBook, start - 1 - indexi)[1] - readexceldata(workBook, start - 1 - indexi - 1)[1]
    return deltayi,fi
    pass


##数据产生  deltaY  deltaU
def generdata(start,timeserse,N):
    workBook = xlrd.open_workbook('C:\\Users\\zaixz\\Desktop\\analys.xlsx')
    deltaYs=np.zeros((timeserse,1))
    deltaFs=np.zeros((timeserse,N))
    for indexi in range(timeserse):
        deltayi, fi = builddata(workBook,start-indexi, N)
        deltaYs[indexi]=deltayi
        deltaFs[indexi,:]=fi.reshape(1,-1)
        print(indexi)
    return deltaYs,deltaFs
    pass



#响应计算，做小二乘法
def leastsquares(deltaYs,deltaFis):
     '''
     :param deltaYs: y的变化值  shape=(N,1)
     :param deltaFis: big fi shape=(N,N)
     :return:
     '''
     Pn=np.linalg.pinv(np.dot(deltaFis.transpose(),deltaFis))
     Kn=np.dot(deltaFis.transpose(), deltaYs)
     result=np.dot(Pn,Kn)
     return result
     pass




#响应计算，递推最小二乘法

def  recursiveleastsquares(deltaY,deltaFi):
    pass


if __name__ == '__main__':

    deltaYs, deltaFs=genermanuldata(101,200,1000,23,10,7,100)
    fi=deltaFs[0, :]
    aaa=np.dot(deltaFs[0,:],respons(23, 10, 7, 100).reshape(-1,1))
    print(aaa)
    print(deltaYs[0])
    result=leastsquares( deltaYs, deltaFs)


    # treal,mv=manuldata(23,180,35,1000)
    # result=respons(23, 10, 35, 100)

    # resul,dmv=buildtotalrespon(23,10,7,500)
    fig, ax1 = plt.subplots()
    PPP = np.arange(0, result.size)
    ax1.plot(PPP,result.reshape(-1,1), '-r')
    # ax1.plot(PPP, dmv.reshape(-1, 1), '-r')
    plt.show()
