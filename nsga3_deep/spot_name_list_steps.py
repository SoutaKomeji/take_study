from math import factorial
import numpy as np
import random
# import xlrd
import time
import pandas as pd

# import matplotlib.pyplot as plt
import numpy
# import pymop.factory

from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

start_time = time.time()

# Problem definition
# PROBLEM = "dtlz2"
NOBJ = 8
# K = 10
# NDIM = NOBJ + K - 1
P = 2
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))

# 制限時間
timeLimit = 60 * 5

# 各評価値の最大，最小値を測定
BOUND_LOW, BOUND_UP = [],[]

# 
maxList = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
minList = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

# 観光スポット数
SPOT_NUM = 58

# Excelデータの読み込みを始める行番号を指定
row = 26

baisuu = 7
# Algorithm parameters
MU = 500#int(H + (4 - H % 4)) * baisuu
NGEN = 500
CXPB = 100 # ％表記でお願いします
MUTPB = 20 # ％表記でお願いします
##

## 観光スポットのデータ作成

# 観光スポットの情報を格納する箱を用意
# dtype = [("id","u1"),("nature", "u1"), ("landscape","u1"),("culture","u1"),("food","u1"),("shopping","u1"),("admission","u2")]
# spotData = np.zeros(61, dtype = dtype)
spotName = []
spotData = []
tTimeData = []
stepData = []

# pandasでread_excel
df = pd.read_excel('preExpData.xlsx')

# pandasで特定のシートを読み込む
featureSheet = pd.read_excel('preExpData.xlsx', sheet_name='特徴表')
tTimeSheet = pd.read_excel('preExpData.xlsx', sheet_name='スポット間移動時間')
stepSheet = pd.read_excel('preExpData.xlsx', sheet_name='スポット間移動歩数')

for i in range(SPOT_NUM + 1):
    spotName.append(featureSheet.iloc[i+25,0])
    spotData.append(featureSheet.iloc[i+25,1:7].values)
    tTimeData.append(tTimeSheet.iloc[i,2:SPOT_NUM+3].values)
    stepData.append(stepSheet.iloc[i,2:SPOT_NUM+3].values)
## 観光スポットのデータ作成終了
# 500世代１回目（自然と移動時間）　61, 52, 37, 39, 26, 50, 56, 61
#2000世代の時　61, 58, 55, 49, 54, 57, 50, 61
# 100世代の時 58, 53, 51, 54, 47, 55, 46, 52, 45, 23, 34, 58
print("観光スポットのリストを入力してください")
# spot_list = [61, 58, 55, 49, 54, 57, 50, 61]#[61, 50, 24, 33, 52, 57, 25, 41, 58, 48, 46, 49, 51, 54, 61]
spot_list = list(map(int, input().split(", ")))
sum = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
avg = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
# print(spotData)
# for i in spot_list:
#     # print(i)
#     # print(spotData[i])
#     print(spotData[i][0])

spot_name = []

print("spot_list : ",spot_list) # sample 10世代[58, 52, 38, 55, 23, 47, 34, 36, 3, 46, 53, 58]
for k in spot_list:
     spot_name.append(spotName[k])
print("spot_name : ",spot_name)
for i in spot_list:
    print("spotData[i] : ",spotData[i])
    if(i != 58):
        for j in range(len(spotData[i])):
            # print("j : ",j)
            if(j == 5):
                sum[j] += float(spotData[i][j])
                
            else:
                # print("spotData[i][j+1] : ",spotData[i][j+1])
                if spotData[i][j] >= 100:
                    sum[j] += float(max(spotData[i][j], 0.0))
                else:   
                    sum[j] += float(max(spotData[i][j+1] - 1.8, 0.0))
                    # sum[j] += float(max(spotData[i][j], 0.0))
                # print("sum[j] : ",sum[j])

time = 0
for j in range(len(spot_list) - 1):
    sum[6] += tTimeData[spot_list[j]][spot_list[j+1]] + 20
    print("tTimeData[spot_list[j]][spot_list[j+1]] : ",tTimeData[spot_list[j]][spot_list[j+1]])
    print("spot_list[j] : ",spot_list[j])
    sum[7] += stepData[spot_list[j]][spot_list[j+1]]
    print("stepData[spot_list[j]][spot_list[j+1]] : ",stepData[spot_list[j]][spot_list[j+1]])
sum[6] -= 20
avg[6] = sum[6] / (len(spot_list) - 1)


# for i in range(len(spotData[0]) - 1):
#     # print("sum",sum[i])
#     # print("len",len(spot_list) - 2)
#     avg[i] = sum[i] / (len(spot_list) - 2)

print("sum:",sum)
# print(avg)