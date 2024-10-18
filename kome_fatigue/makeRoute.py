from math import factorial
import numpy as np
import random
# import xlrd
import time

# import matplotlib.pyplot as plt
import numpy
# import pymop.factory
import copy
import pandas as pd

from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

start_time = time.time()
# Problem definition
# PROBLEM = "dtlz2"

#目的関数の数の設定
# NOBJ = 9 #歩数あり
NOBJ = 12 #歩数，身体的疲労，精神的疲労あり

K = 10
NDIM = NOBJ + K - 1
P = 2
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))

# 制限時間
timeLimit = 60 * 5

# 各評価値の最大，最小値を測定
BOUND_LOW, BOUND_UP = [],[]


# 疲労，観光時間追加
maxList = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
minList = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

# 観光スポット数
# SPOT_NUM = 61
SPOT_NUM = 58

# Excelデータの読み込みを始める行番号を指定
row = 26

# Algorithm parameters
MU = 500 #int(H + (4 - H % 4))
# NGEN = 10
print("何世代で実行するか入力してください")
NGEN = int(input())
# NGEN = 30
CXPB = 100 # ％表記でお願いします
MUTPB = 20 # ％表記でお願いします
##

# 各リファレンスポイントに関連する解の個体数を把握するための変数
num_associate_rp = []

# １つのリファレンスポイントの評価値を保存する用の変数
record_evaluate_value = []

# 各世代の親個体，交叉した個体，変異した個体の個数を格納する
# 生存率の計算が機能しているか確認する用
ind_ratio = [0,0,0]

# 生存率を格納する容器
# survive_mate_pro = []
# survive_mutate_pro = []
survive_new_ind_pro = [] # 交叉した個体と突然変異した個体を合計した生存率

# 各個体が何世代目に生まれたかを格納する
ind_gen = [0] * (NGEN + 1)

## 観光スポットのデータ作成

# 観光スポットの情報を格納する箱を用意
# dtype = [("id","u1"),("nature", "u1"), ("landscape","u1"),("culture","u1"),("food","u1"),("shopping","u1"),("admission","u2")]
# spotData = np.zeros(61, dtype = dtype)
spotData = []
tTimeData = []
stepData = []
# Excelファイル全体の読み込み
# wb = xlrd.open_workbook('preExpData.xlsx')
# pandasでread_excel
df = pd.read_excel('preExpData_fatigue.xlsx')

# Excelファイル内の特定のシートの読み込み
# featureSheet = wb.sheet_by_name('特徴表')
# tTimeSheet = wb.sheet_by_name('スポット間移動時間')
# pandasで特定のシートを読み込む
featureSheet = pd.read_excel('preExpData_fatigue.xlsx', sheet_name='特徴表')
tTimeSheet = pd.read_excel('preExpData_fatigue.xlsx', sheet_name='スポット間移動時間')
stepSheet = pd.read_excel('preExpData_fatigue.xlsx', sheet_name='スポット間移動歩数')

for i in range(SPOT_NUM + 1):
    spotData.append(featureSheet.iloc[i+25,1:10].values)
    tTimeData.append(tTimeSheet.iloc[i,2:SPOT_NUM + 3].values)
    stepData.append(stepSheet.iloc[i,2:SPOT_NUM + 3].values)

# print("spotData:", spotData)
## 観光スポットのデータ作成終了

# 観光コース内のコースを回る順番を作成(歩数あり)
def singleCourseData(spotData, tTimeData, stepData, minSpotNum, maxSpotNum):
    nobj = len(spotData[0])
    spotNum = random.randint(minSpotNum, maxSpotNum)

    route = []
    # 出発地点の追加(函館駅を指定)
    route.append(58)

    # 全スポットから重複なしでランダムにスポットを選択する
    # range(x) は 0 から x-1 までの値を指す
    route.extend(random.sample(range(SPOT_NUM), k=spotNum))
    
    # 終着地点の追加(函館駅を指定)
    route.append(58)

    time = 0
    steps = 0
    for j in range(len(route) - 1):
        time += tTimeData[route[j]][route[j+1]] + 20
        steps += stepData[route[j]][route[j+1]]
    
    routeData = []
    routeData.append(route)
    routeData.append(time)
    routeData.append(steps)

    routeData.append(0)

    return routeData


# 観光コース内のコースを回る順番を作成(歩数あり)
def singleCourseData(spotData, tTimeData, stepData, minSpotNum, maxSpotNum):
    nobj = len(spotData[0])
    spotNum = random.randint(minSpotNum, maxSpotNum)

    route = []
    # 出発地点の追加(函館駅を指定)
    route.append(58)

    # 全スポットから重複なしでランダムにスポットを選択する
    # range(x) は 0 から x-1 までの値を指す
    route.extend(random.sample(range(SPOT_NUM), k=spotNum))
    
    # 終着地点の追加(函館駅を指定)
    route.append(58)

    time = 0
    steps = 0
    # for j in range(len(route) - 1):
    #     time += tTimeData[route[j]][route[j+1]] + 20
    #     steps += stepData[route[j]][route[j+1]]

    for j in range(len(route) - 1):
        time += tTimeData[route[j]][route[j+1]] + 20
        steps += stepData[route[j]][route[j+1]]
    
    routeData = []
    routeData.append(route)
    routeData.append(time)
    routeData.append(steps)

    routeData.append(0)

    return routeData

# コース評価用の関数
def evaluate(spotData, tTimeData, stepData, inds):#参照しているのは単独の個体

    # print("inds:", inds)
    nature = 0
    landscape = 0
    culture = 0
    food = 0
    shopping = 0
    admission = 0
    phy_fatigue = 100
    men_fatigue = 100
    tour_time = 0    
    steps = 0
    time = 0
    num = 0
    # コース内の観光スポット数分

    ind = toolbox.clone(inds)

    for j in ind[0][1:-1]:
        nature += max(spotData[j][0] - 1.8, 0) #自然
        landscape += max(spotData[j][1] - 1.8, 0) #風景
        culture += max(spotData[j][2] - 1.8, 0) #文化
        food += max(spotData[j][3] - 1.8, 0) #食
        shopping += max(spotData[j][4] - 1.8, 0) #買い物
        admission += spotData[j][5] #入場料
        phy_fatigue -= spotData[j][6] #身体的疲労
        men_fatigue -= spotData[j][7] #精神的疲労
        tour_time += spotData[j][8] # 1スポットあたりの観光時間


        
    # 出発地点と終着地点を省略
    time = ind[1]
    # print("time (ind[1]): ",time)
    # スポットの数？
    num = len(ind[0]) - 2

    return nature, landscape, culture, food, shopping, admission, phy_fatigue, men_fatigue, tour_time, time, steps, num



toolbox = base.Toolbox()