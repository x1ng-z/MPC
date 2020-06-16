import numpy as np
import matplotlib.pyplot as plt



def productData(time):
    return 10*np.sin(2*np.pi*time/100)+np.random.random()*10

def moveAveragefilter(unfilterdata):
    return np.sum(unfilterdata)/unfilterdata.size

def FirstOrderLagfilter(Yn_1,Xn,a):
    Yn=(1-a)*Yn_1 + a*Xn#Yn_1+a*(Xn-Yn_1)
    return Yn



if __name__ == '__main__':
    win=20
    time=300
    unfilterdata=np.zeros(time)
    moveAvefilterdata=np.zeros(time)
    onestep=np.zeros(time)
    alph=0.6
    Yn=0
    for indext in range(time):
        unfilterdata[indext]=productData(indext)
        if indext>=(win-1):
            moveAvefilterdata[indext]=moveAveragefilter(unfilterdata[indext - win + 1:indext])
        if indext==0:
            Yn=unfilterdata[0]
        Yn=FirstOrderLagfilter(Yn,unfilterdata[indext],alph)
        onestep[indext]=Yn


    fig, ax1 = plt.subplots()
    PPP=np.arange(0,time)

    ax1.plot(PPP,unfilterdata,'-b')
    # plt.show()
    # fig, ax1 = plt.subplots()
    ax1.plot(PPP, moveAvefilterdata, '-r')
    # plt.show()
    # fig, ax1 = plt.subplots()
    ax1.plot(PPP,onestep,'-g')
    # ax1.plot(PPP,U[1,:],'-k')
    plt.show()
    pass