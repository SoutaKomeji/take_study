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

# # 歩数なし
# maxList = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# minList = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

# # 歩数あり
# maxList = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# minList = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

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

# Create uniform reference point
# ここでリファレンスポイントを作成している（NOBJ:目的関数の数，P:リファレンスポイントの次元）
ref_points = tools.uniform_reference_points(NOBJ, P)

# reference points の表示
# for i in range(len(ref_points)):
#     print("ref_points[",i,"]:",ref_points[i])


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

# pandasでread_excel
df = pd.read_excel('preExpData_fatigue.xlsx')

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

def mate(inds1, inds2, gen):
    # コースを分割する場所を決定
    ind1 = copy.deepcopy(inds1)
    ind2 = copy.deepcopy(inds2)

    child1= []
    child2= []

    breakP1 = random.randint(1,len(ind1[0])-1)
    breakP2 = random.randint(1,len(ind2[0])-1)

    ind1Front = ind1[0][0:breakP1]
    ind1Behind = ind1[0][breakP1:len(ind1[0])]
    ind2Front = ind2[0][0:breakP2]
    ind2Behind = ind2[0][breakP2:len(ind2[0])]

    # 重複があった場合に，ランダムに削除する(course1用)
    t = [x for x in set(ind1Front + ind2Behind) if (ind1Front + ind2Behind).count(x) > 1]
    t.remove(58)
    for i in t:
        loot = random.randint(0,1)

        if(loot == 0):
            ind1Front.remove(i)
        else:
            ind2Behind.remove(i)
    
    # 重複があった場合に，ランダムに削除する(course2用)
    t = [x for x in set(ind2Front + ind1Behind) if (ind2Front + ind1Behind).count(x) > 1]
    t.remove(58)
    # if(t):
    for i in t:
        loot = random.randint(0,1)
        if(loot == 0):
            ind2Front.remove(i)
        else:
            ind1Behind.remove(i)
    
    course1 = ind1Front + ind2Behind
    course2 = ind2Front + ind1Behind

    child1.append(course1)
    child2.append(course2)

    #　観光ルートの時間と歩数を再計算している
    time = 0
    steps = 0
    for j in range(len(course1) - 1):
        time += tTimeData[course1[j]][course1[j+1]] + 20
        steps += stepData[course1[j]][course1[j+1]]
    time -= 20 # 最後の観光地(函館駅)の観光時間を引く
    child1.append(time)
    child1.append(steps)
    child1.append(gen + 1)

    time = 0
    steps = 0
    for j in range(len(course2) - 1):
        time += tTimeData[course2[j]][course2[j+1]] + 20
        steps += stepData[course2[j]][course2[j+1]]
    time -= 20 # 最後の観光地(函館駅)の観光時間を引く
    child2.append(time)
    child2.append(steps)
    child2.append(gen + 1)

    childA = copy.deepcopy(child1)
    childB = copy.deepcopy(child2)

    return type(ind1)(childA), type(ind1)(childB)

def mutate(ind,gen): # [[61, 43, 42, 20] 560]

    loot = 0
    if(len(ind[0]) > 3):
        loot = random.randint(0,1)  
    
    # 増加変異
    if(loot == 0):
        # 既にコース内に含まれている観光スポットを削除し，
        # 削除した中から追加するスポットを選択
        t = set(ind[0])^set(range(SPOT_NUM))
        t.remove(58)
        addSpot = random.choice(list(t))
        # 観光スポットを追加する場所を選択
        addPoint = random.randint(1,len(ind[0]) - 1)
        ind[0].insert(addPoint,addSpot)
        # print("増加変異")

    # 減少変異
    elif(loot == 1):
        t = set(ind[0])^set([SPOT_NUM]) #スタート地点を削除する
        removeSpot = random.choice(list(t))
        ind[0].remove(removeSpot)
        # print("減少変異")

    time = 0
    steps = 0
    for j in range(len(ind[0]) - 1):
        time += tTimeData[ind[0][j]][ind[0][j+1]] + 20
        steps += stepData[ind[0][j]][ind[0][j+1]]
    time -= 20 # 最後の観光地(函館駅)の観光時間を引く
        
    # print("steps : ",steps)
    
    ind[1] = time
    ind[2] = steps
    ind[3] = gen + 1
    
    return ind