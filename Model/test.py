import numpy as np
import sys
import Help
import matplotlib.pyplot as plt
import json


str = '{"key": "wwww", "word": "qqqq"}'
j = json.loads(str)
a=json.dumps(j)



aaaaa=True
aaaaa=(aaaaa==False)
aaa=np.array([1,2]).reshape(-1,1)

print(len(aaa))
if aaa!=[]:
    print(aaa.shape)
matrixPvMvMapping=np.array([[1,1],[1,0]])
aaa=matrixPvMvMapping.copy()
a1=np.sum(matrixPvMvMapping,axis=0)
b1=[1,2]
print(a1/b1)

nua=np.ones((2,3,4))
aa=nua.shape

print(aa[0])
print(aa[1])
print(aa[2])

ll=np.array([1,2,3])
print(ll)

print(ll.reshape(-1))
for index, needcheckdmv in np.ndenumerate(ll):
    print(index[0])
    print(needcheckdmv)

print(matrixPvMvMapping)
print(aaa)

matrixPvMvMapping[matrixPvMvMapping==1]=[3,4,5]


p=90
aaa=np.array([1 - 0.95** i for i in range(1, p + 1)]).reshape(-1, 1)
aaa2=np.array([1 - 0.8 ** i for i in range(1, p + 1)]).reshape(-1, 1)
aaa3=np.array([1 - 0.5 ** i for i in range(1, p + 1)]).reshape(-1, 1)
aaa4=np.array([1 - 0.1 ** i for i in range(1, p + 1)]).reshape(-1, 1)

# aaa=np.flipud(aaa)

plt.figure()
X = np.arange(0, p, 1)
plt.plot(X,aaa, 'k-',label="0.95")
plt.plot(X,aaa2, 'g-',label="0.8")
plt.plot(X,aaa3, 'r-',label="0.5")
plt.plot(X,aaa4, 'b-',label="0.1")
plt.legend(loc='upper right')#绘制曲线图例，信息来自类型label
plt.show()


a=np.zeros(2)

tools = Help.Tools()

funels=tools.buildFunel(np.array([2,3]),np.array([0.2,0.2]),np.array([1,1]),10,2)

wi=tools.biuldWiByFunel(2,8,10,np.array([1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]).reshape(-1,1),funels)

N=20
coeff = np.array([1 - 0.8 ** i for i in range(1,N+1)])
m=3
p=2
A=np.random.rand(p,m)
Q=np.random.rand(p,p)
delu=np.random.rand(m,1)
C=np.random.rand(p,1)


p=2
P=20
alphe=[0.7,0.8]
alphecoeff = np.zeros((p * P, 1))
for indexp in range(p):
    alphecoeff[P * indexp:P * (indexp + 1)] = np.array([1 - alphe[indexp] ** i for i in range(1, P + 1)]).reshape(-1,1)
alphediag = np.diagflat(alphecoeff)
alphediag_2=np.power(np.diagflat(alphecoeff),2)



t1=np.dot(np.dot(A.transpose(),Q),np.dot(A,delu)*C)

Cdiag=np.diagflat(C)

t2=np.dot(np.dot(np.dot(A.transpose(),Q),Cdiag),np.dot(A,delu))

print('t1=',t1,'t2',t2)




def biuldWi( p, P, wi,yreal,alph):
    W_i = np.zeros((p * P, 1))

    for indexp in range(p):
        for indexP in range(P):
            tempwp=(1-alph**(indexP+1))*wi[indexp]+alph**(indexP+1)*yreal[indexp]
            W_i[indexp * P + indexP, 0]=tempwp
    return W_i


if __name__=='__main__':

    aaaa=0.6**2
    p=1
    P=30
    W_i=biuldWi(p,P,[10],[0],0.5)




    firstonedmv=np.array([[1]])
    limitdmv=np.array([[0,3]])
    limitmv=np.array([[17,20]])
    mv=np.array([6])
    for index, needcheckdmv in np.ndenumerate(firstonedmv):
        '''检查下dmv是否在限制之内'''
        if (np.abs(needcheckdmv) > limitdmv[index[0], 1]):
            firstonedmv[index[0], 0] = limitdmv[index[0], 1] if (firstonedmv[index[0]] > 0) else (
                    -1 * limitdmv[index[0], 1])
        '''dmv是否小于最小调节量，如果小于，则不进行调节'''
        if (np.abs(needcheckdmv) <= limitdmv[index[0], 0]):
            firstonedmv[index[0], 0] = 0
        '''nv叠加dmv完成以后是否大于mvmax'''
        if ((mv[index[0]] + firstonedmv[index[0]]) >= limitmv[index[0], 1]):
            firstonedmv[index[0], 0] = limitmv[index[0], 1] - mv[index[0]]
        '''nv叠加dmv完成以后是否大于mvmax'''
        if ((mv[index[0]] + firstonedmv[index[0]]) <= limitmv[index[0], 0]):
            firstonedmv[index[0], 0] = limitmv[index[0], 0] - mv[index[0]]


    d = {'1': 'one', '3': 'three', '2': 'two', '5': 'five', '4': 'four'}
    print("100" in d)
    dmvmin = np.zeros((2* 3, 1))
    print(dmvmin.reshape(-1))
    x = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    x0 = np.asarray(x)
    y_0P = np.zeros((1 * 135, 1))
    will = y_0P + np.dot(self.A, deltu)
    gradient = 2 * np.dot(np.dot(-1 * self.A.transpose(), self.Q), will) + 2 * np.dot(self.R, deltu)

    print(H)


    mmm=np.array([[1],[2]])
    nnn=np.array([1,2])
    for index,needcheckdmv in np.ndenumerate(nnn):
        print(index,needcheckdmv)
        print(nnn[index])
    tmmm=nnn.reshape(-1)


    k=2*4000 // 8000
    aa=np.zeros((3,1))
    bb=np.ones((1,3))
    cc=aa.all()<=bb
    for i in range(0, len(sys.argv)):
        m=2
        M=3
        B = np.zeros((m * M, m * M))
        for indexIn in range(m):
            for noderow in range(M):
                for nodecol in range(M):
                    if (nodecol <= noderow):
                        B[indexIn * M + noderow, indexIn * M + nodecol] = 1
        print(B)
        strs = sys.argv[i]
        print(i)
        if i==2:
            print(int(strs))
