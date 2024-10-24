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
df = pd.read_excel('preExpData_fatigue.xlsx')

# pandasで特定のシートを読み込む
featureSheet = pd.read_excel('preExpData_fatigue.xlsx', sheet_name='特徴表')
tTimeSheet = pd.read_excel('preExpData_fatigue.xlsx', sheet_name='スポット間移動時間')
stepSheet = pd.read_excel('preExpData_fatigue.xlsx', sheet_name='スポット間移動歩数')

for i in range(SPOT_NUM + 1):
    spotName.append(featureSheet.iloc[i+25,0])
    spotData.append(featureSheet.iloc[i+25,1:10].values)
    tTimeData.append(tTimeSheet.iloc[i,2:SPOT_NUM+3].values)
    stepData.append(stepSheet.iloc[i,2:SPOT_NUM+3].values)
## 観光スポットのデータ作成終了

# 自然とmental_fatigue
# 50世代の時　58, 55, 52, 46, 23, 49, 54, 34, 47, 58
# 自然： 7.6000000000000005 ，風景： 12.400000000000002 ，文化： 5.000000000000001 ，食： 2.4 ，買い物： 2.4 ，料金： 2700.0 ，phy： 87.0 ，men: 90.0 ，spottime: 261.0 ，観光移動時間: 0.0 ，観光歩数: 8050.0 ，スポットの数: 0.0
# 200世代の時 58, 53, 56, 51, 49, 34, 47, 23, 55, 54, 52, 45, 58
# 自然： 11.2 ，風景： 15.999999999999996 ，文化： 5.2 ，食： 2.4 ，買い物： 2.4 ，料金： 2900.0 ，phy： 77.0 ，men: 81.0 ，spottime: 317.0 ，観光移動時間: 0.0 ，観光歩数: 7650.0 ，スポットの数: 0.0
# 500世代の時 [[58, 10, 52, 23, 55, 34, 49, 56, 54, 51, 58], 277, 7400.0, 1]
# 自然： 7.6000000000000005 ，風景： 10.399999999999999 ，文化： 4.200000000000001 ，食： 3.4000000000000004 ，買い物： 1.2 ，料金： 2600.0 ，phy： 82.0 ，men: 87.0 ，spottime: 277.0 ，観光移動時間: 0.0 ，観光歩数: 7400.0 ，スポットの数: 0.0

# 自然とphysical_fatigue


# 自然と文化


# 100世代の時 58, 53, 51, 54, 47, 55, 46, 52, 45, 23, 34, 58
print("観光スポットのリストを入力してください")
# spot_list = [61, 58, 55, 49, 54, 57, 50, 61]#[61, 50, 24, 33, 52, 57, 25, 41, 58, 48, 46, 49, 51, 54, 61]
spot_list = list(map(int, input().split(", ")))
# [自然，風景:1，文化，食，買い物，料金:5，phy:6, men:7, spottime, 観光移動時間:9，観光歩数，スポットの数:11]
sum = [0.0,0.0,0.0,0.0,0.0,0.0,100.0,100.0,0.0,0.0,0.0,0.0]
avg = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
# print(spotData)
# for i in spot_list:
#     # print(i)
#     # print(spotData[i])
#     print(spotData[i][0])

spot_name = []


# [自然，風景:1，文化，食，買い物，料金:5，phy:6, men:7, spottime, 観光移動時間:9，観光歩数，スポットの数:11]
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
                sum[j] += spotData[i][j]
                
            elif j < 5:
                sum[j] += max(spotData[i][j] - 1.8, 0.0)
                print("spotData[i][j+1] : ",spotData[i][j])
                if spotData[i][j] >= 100:
                    sum[j] += float(max(spotData[i][j], 0.0))
                # print("sum[j] : ",sum[j])
            elif j > 5 and j < 8:
                sum[j] -= spotData[i][j]

time = 0

# [自然，風景:1，文化，食，買い物，料金:5，phy:6, men:7, spottime:8, 観光移動時間:9，観光歩数:10，スポットの数:11]
for j in range(len(spot_list) - 1):
    sum[8] += tTimeData[spot_list[j]][spot_list[j+1]] + 20
    print("tTimeData[spot_list[j]][spot_list[j+1]] : ",tTimeData[spot_list[j]][spot_list[j+1]])
    print("spot_list[j] : ",spot_list[j])
    sum[10] += stepData[spot_list[j]][spot_list[j+1]]
    print("stepData[spot_list[j]][spot_list[j+1]] : ",stepData[spot_list[j]][spot_list[j+1]])
sum[8] -= 20

for i in range(len(spotData[0]) - 1):
    # print("sum",sum[i])
    # print("len",len(spot_list) - 2)
    avg[i] = sum[i] / (len(spot_list) - 2)
# avg[6] = sum[6] / (len(spot_list) - 1)


# for i in range(len(spotData[0]) - 1):
#     # print("sum",sum[i])
#     # print("len",len(spot_list) - 2)
#     avg[i] = sum[i] / (len(spot_list) - 2)

print("自然：",sum[0], "，風景：",sum[1], "，文化：",sum[2], "，食：",sum[3], "，買い物：",sum[4], "，料金：",sum[5], "，phy：",
      sum[6], "，men:",sum[7],"，spottime:",sum[8],"，観光移動時間:",sum[9],"，観光歩数:",sum[10],"，スポットの数:",sum[11])
# print(avg)