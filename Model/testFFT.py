import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


def readexceldata(workBook, row):
    sheet1 = workBook.sheet_by_name('Sheet1')
    rows = sheet1.row_values(row)
    return rows
    pass

def generdata(start,N):
    workBook = xlrd.open_workbook('C:\\Users\\zaixz\\Desktop\\tf.xlsx')
    deltaYs=np.zeros(N)

    for indexi in range(N):
        deltayi= readexceldata(workBook,start+indexi)
        print(str(deltayi[0])+',')
        deltaYs[indexi]=deltayi[0]
    return deltaYs
    pass



# y=generdata(1,20)
# N=100

# 采样点数
start=2977
N = 3577-start#4000

# 采样频率 (根据采样定理，采样频率必须大于信号最高频率的2倍，信号才不会失真)
Fs = 1/1
# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x = np.linspace(0, N/Fs, N)

# 设置需要采样的信号，频率分量有200，400和600
y = generdata(start,N)#7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)
y_mean=np.mean(y)
y_norm=y-y_mean
fft_y = fft(y_norm)  # 快速傅里叶变换

# N = 1400
x = np.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = np.angle(fft_y)  # 取复数的角度
mo_abs_y=abs_y.copy()
mo_abs_y[0]=mo_abs_y[0]/N
mo_abs_y[1:]=mo_abs_y[1:]/(N/2)
normalization_y = mo_abs_y  # 归一化处理（双边频谱）
normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

third=int(len(normalization_half_y)/3)

sum=0
for indexlowhz in range(third):
    sum+=normalization_half_y[indexlowhz]**2

print(np.sqrt(sum/(third)))
print(third)

plt.subplot(231)
plt.plot(x, y)
plt.title('原始波形'+str(np.sqrt(sum/third)))

plt.subplot(232)
plt.plot(x, fft_y, 'black')
plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')

plt.subplot(233)
plt.plot(x, abs_y, 'r')
plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')

plt.subplot(234)
plt.plot(x, angle_y, 'violet')
plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')

plt.subplot(235)
plt.plot(x, normalization_y, 'g')
plt.title('双边振幅谱(归一化)', fontsize=9, color='green')

plt.subplot(236)
plt.plot(half_x, normalization_half_y, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')

plt.show()















# 采样点数
# N = 52#4000
#
# # 采样频率 (根据采样定理，采样频率必须大于信号最高频率的2倍，信号才不会失真)
# Fs = 1
# x = np.linspace(0.0, N/Fs, N)
#
# # 时域信号，包含：直流分量振幅1.0，正弦波分量频率100hz/振幅2.0, 正弦波分量频率150Hz/振幅0.5/相位np.pi
# y = 1.0 + 2.0 * np.sin(100.0 * 2.0*np.pi*x) + 0.5*np.sin(150.0 * 2.0*np.pi*x + np.pi)
# y=np.array([
# 865,
# 865,
# 865,
# 865,
# 865,
# 865,
# 865,
# 865,
# 865,
# 865,
# 865,
# 865,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 864,
# 863,
# 863,
# 863,
# 863,
# 863,
# 863,
# 863,
# 862,
# 862,
# 862,
# 862,
# 862,
# 862,
# 862,
# 862,
# 861,
# 861,
# 861,
# 861,
# 861,
# 861,
# 860,
# 860,
# 860,
# 860,
# 860
# ])
# y=[1,0,0,1]
# ,进行fft变换
# yf = fft(y)
#
# # 获取振幅，取复数的绝对值，即复数的模
# abs_yf = np.abs(yf)
#
# # 获取相位，取复数的角度
# angle_y=np.angle(yf)
#
#
# fft_y = fft(y)  # 快速傅里叶变换
#
# # N = 1400
# x = np.arange(N)  # 频率个数
# half_x = x[range(int(N / 2))]  # 取一半区间
#
# abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
# angle_y = np.angle(fft_y)  # 取复数的角度
# normalization_y = abs_y*2 / N  # 归一化处理（双边频谱）
# normalization_y[0]=normalization_y[0]/2
# normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
#
# plt.subplot(231)
# plt.plot(x, y)
# plt.title('原始波形')
#
# plt.subplot(232)
# plt.plot(x, fft_y, 'black')
# plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')
#
# plt.subplot(233)
# plt.plot(x, abs_y, 'r')
# plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')
#
# plt.subplot(234)
# plt.plot(x, angle_y, 'violet')
# plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')
#
# plt.subplot(235)
# plt.plot(x, normalization_y, 'g')
# plt.title('双边振幅谱(归一化)', fontsize=9, color='green')
#
# plt.subplot(236)
# plt.plot(half_x, normalization_half_y, 'blue')
# plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
#
# plt.show()
#
# # 直流信号
# print('\n直流信号')
# print('振幅:', abs_yf[0]/N) # 直流分量的振幅放大了N倍
# print('其他振幅:', abs_yf[1:]*2.0/N)
#
# fig, ax1 = plt.subplots()
# PPP=np.arange(0,N)
# ax1.plot(PPP,y[0:],'-b')
# ax1.plot(PPP[1:],abs_yf[1:],'-r')
# plt.show()
#
#
# # 100hz信号
# index_100hz = 100 * N // Fs # 波形的频率 = i * Fs / N，倒推计算索引：i = 波形频率 * N / Fs
#
#
#
# print('\n100hz波形')
# print('振幅:', abs_yf[index_100hz] * 2.0/N) # 弦波分量的振幅放大了N/2倍
# print('相位:', angle_y[index_100hz])
#
# # 150hz信号
# index_150hz = 150 * N // Fs # 波形的频率 = i * Fs / N，倒推计算索引：i = 波形频率 * N / Fs
# print('\n150hz波形')
# print('振幅:', abs_yf[index_150hz] * 2.0/N) # 弦波分量的振幅放大了N/2倍
# print('相位:', angle_y[index_150hz])
# print('100hz与150hz相位差:',  angle_y[index_150hz] - angle_y[index_100hz])
# print('\n')