"""
Raw signal  原始信号
Filter signal 滤波
Derivative 差分处理
Squared Derivative 平方处理
Moving Window Integral 积分窗口滑动获得积分波形
"""

from cProfile import label
from sklearn import preprocessing
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import pywt

figure(num=None, figsize=(15, 5), dpi=80)

# 读取数据
data = pd.read_csv('1.csv')
data = data.values

T = 256
N = 24

rE = T//3
E = T//7

x = data[:]
x = x.astype("float")

# 原始信号
plt.subplot(7, 1, 1)
plt.title('Raw Signal')
plt.plot(x)

# 标准化
# x = (x - np.mean(x)) / np.std(x)
# plt.subplot(6, 1, 2)
# plt.title('Standardization Signal')
# plt.plot(x)

# 低通滤波器
x1 = sig.lfilter([1,0,0,0,0,0,-2,0,0,0,0,0,1],[1,-2,1],x)
plt.subplot(7,1,2)
plt.title('lowpass filter')
plt.plot(x1)

# 高通滤波器
x2 = sig.lfilter([-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[1,1],x1)
# x2 = x2 / max(abs(x2))
plt.subplot(7,1,3)
plt.title('highpass filter')
plt.plot(x2)

# Derivative 差分处理
x3 = np.zeros(x.shape)
for i in range(2,len(x2)-2):
    x3[i] = (-1*x2[i-2] -2*x2[i-1] + 2*x2[i+1] + x2[i+2])/(8*T)
plt.subplot(7,1,4)
plt.title('Derivative')
plt.plot(x3)

# 将微分处理得到数据逐点平方
x4 = x3*x3
plt.subplot(7, 1, 5)
plt.title('Squared Derivative')
plt.plot(x4)

# Moving Window Integral 积分窗口滑动获得积分波形
x5 = np.zeros(x.shape) #生成全0矩阵
for i in range(N, len(x4)-N):
    for j in range(N):
        x5[i] += x4[i-j]
x5 = x5/N
plt.subplot(7, 1, 6)
plt.title('Moving Window Integral')
plt.plot(x5)


peaki = x5[0]
spki = 0
npki = 0
c = 0
peak = [0]
threshold1 = spki
pk = []
for i in range(1, len(x5)):
    if x5[i] > peaki:
        peaki = x5[i]

    npki = ((npki*(i-1))+x5[i])/i
    spki = ((spki*(i-1))+x5[i])/i
    spki = 0.875*spki + 0.125*peaki
    npki = 0.875*npki + 0.125*peaki

    threshold1 = npki + 0.25*(spki-npki)
    threshold2 = 0.5 * threshold1

    if(x5[i] >= threshold2):

        if(peak[-1]+N < i):
            peak.append(i)
            pk.append(x5[i])

p = np.zeros(len(x5))
rPeak = []
Q = np.zeros(2)
S = np.zeros(2)
THR = 50
for i in peak:
    if(i == 0 or i<2*rE):
        continue
    p[i] = 1

    # ind = np.argmax(x2[i-rE:i+rE])
    ind = np.argmax(x[i-rE:i+rE])
    maxIndexR = (ind+i-rE)
    rPeak.append(maxIndexR)
    # plt.plot(maxIndexR, x2[maxIndexR], 'ro', markersize=12)
    plt.subplot(7,1,7)
    plt.title('R peak find')
    plt.plot(x)
    plt.plot(maxIndexR, x[maxIndexR], 'ro', markersize=5)
    prevDiffQ = 0
    prevDiffS = 0

# #    FIND THE Q POINT
#     for i in range(1, THR):

#         Q[0] = x2[maxIndexR-i]
#         Q[1] = x2[maxIndexR-(i+1)]

#         diffQ = Q[0]-Q[1]

#         if(diffQ < prevDiffQ):
#             minIndexQ = maxIndexR-i
#             break
#         prevDiffQ = diffQ / 5

#     plt.plot(minIndexQ, x2[minIndexQ], 'bo', markersize=6)   

# #    FIND THE S POINT
#     for i in range(1, THR):

#         S[0] = x2[maxIndexR+i]
#         S[1] = x2[maxIndexR+(i+1)]

#         diffS = S[0]-S[1]

#         if(diffS < prevDiffS):
#             minIndexS = maxIndexR+i
#             break
#         prevDiffS = diffS / 5

#     plt.plot(minIndexS, x2[minIndexS], 'go', markersize=6)

rPeak = np.unique(rPeak)

plt.xlabel('time')
plt.show()
