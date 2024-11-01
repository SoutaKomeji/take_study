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

import makeRoute
import debug
import GA

start_time = time.time()

# Problem definition
# PROBLEM = "dtlz2"

#目的関数の数の設定
# NOBJ = 9 #歩数あり
NOBJ = 12 #歩数，身体的疲労，精神的疲労あり

# 設計変数の数
K = 10

# 次元数
NDIM = NOBJ + K - 1

#　参照点の数
P = 2

# ハイパーグリッドのサイズ
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))

# 制限時間
timeLimit = 60 * 5

# 観光スポット数
SPOT_NUM = 58

# Excelデータの読み込みを始める行番号を指定
row = 26

# Algorithm parameters
MU = 500 #int(H + (4 - H % 4))
# NGEN = 10
print("何世代で実行するか入力してください")
NGEN = int(input())
# NGEN = 30
# 交叉率
CXPB = 100 # ％表記でお願いします

# 突然変異率
MUTPB = 20 # ％表記でお願いします
##

# Create uniform reference point
# ここでリファレンスポイントを作成している（NOBJ:目的関数の数，P:リファレンスポイントの次元）
ref_points = tools.uniform_reference_points(NOBJ, P)

# for i in range(len(ref_points)):
#     print("ref_points[",i,"]:",ref_points[i])

# # 特定の条件のリファレンスポイントの追加の場合はこっちから
# ref_points = np.array([
#     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
#     [0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.7,0.0,0.0,0.0,0.0],
#     [0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5,0.0,0.0,0.0,0.0],
#     [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3,0.0,0.0,0.0,0.0],
#     [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
#     [0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0],
#     [0.5,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0,0.0]
# ])



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

# ここはクラスを作っているだけ
# Create classes
creator.create("FitnessMin", base.Fitness, weights=(1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# # 疲労，観光時間追加
maxList = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
minList = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

# maxList = [-10.0] * NOBJ
# minList = [10.0] * NOBJ

def makeMinMax(pop):
    # 最大値，最小値の測定
    for i in range(len(pop)): # 最初の個体を最大値と最小値に設定するため「-1」する
        for j in range(len(maxList)):
            if(maxList[j] < pop[i].fitness.values[j]):
                maxList[j] = pop[i].fitness.values[j]
            if(minList[j] > pop[i].fitness.values[j]):
                minList[j] = pop[i].fitness.values[j]

def normalizedInd(ind):
    # 最大値と最小値から正規化

    evaList = list(ind.fitness.values)
    for j in range(len(evaList)):
        if(minList[j] == maxList[j]): 
            # print("最大値と最小値が同じである")
            evaList[j] = 0.5

        # 最小化したいものはこっち(5は利用費，9はtime，10は観光歩数)
        elif(j==5 or j == 6 or j ==7):
            # print("pre_evaList[j]:",evaList[j])
            evaList[j] = 1 - ((evaList[j] - minList[j]) / (maxList[j] - minList[j]))
            # print("evaList[j]:",evaList[j])

        # 最大化したいものはこっち
        else:
            evaList[j] = (evaList[j] - minList[j]) / (maxList[j] - minList[j])
    # print("ind[1]:",ind[1])
    # 制限時間を超過した場合に，評価値にペナルティを与えている．
    if(timeLimit < ind[1]):
        for j in range(len(evaList)):
            if 0 < evaList[j] - 1.0 * ((ind[1] - timeLimit) / 200):
                evaList[j] -= 1.0 * ((ind[1] - timeLimit) / 200)
                # print("minusdayo")
            else:
                evaList[j] = 0
            #　もとは1.0だった

    return tuple(evaList)

def perpendicular_distance(direction, point):
    # 垂直距離を求める際に，原点との直線上にある垂直距離になる場所まで，点を移動させているが，使い方が間違っているっぽいぞ！？
    k = np.dot(direction, point) / np.sum(np.power(direction, 2))
    # 単純に2点間の距離を求めている．
    d = np.sum(np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point) , 2))
    return np.sqrt(d)

class ReferencePoint(list):
    '''A reference point exists in objective space an has a set of individuals
    associated to it.'''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []

def associate(individuals, reference_points_data):
    '''Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014).'''
    pareto_fronts = tools.sortLogNondominated(individuals, len(individuals))
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        rp_dists = [(rp, perpendicular_distance(rp, ind.fitness.values))
                    for rp in reference_points_data]
        # print("rp:", rp)

        # # 各リファレンスポイントに関連する解の個体数を把握するための変数
        # num_associate_rp = [0,[]] * len(ref_points)
        # num_associate_rp[0][1].append(0.9)

        # num_associate_rp = []
        list = []

        # ある個体の各リファレンスポイントとの距離を格納している
        t = rp_dists
        num_ind = 0

        for i in range(len(t)):
            list.append(t[i][1])
        
        #  num_associate_rp に各個体とリファレンスポイントごとの距離を格納
        # num_associate_rp.append(list)

        best_rp, best_dist = sorted(rp_dists, key=lambda rpd:rpd[1])[0]
        # ここの型にエラーが出るかも
        ind.reference_point = best_rp
        ind.ref_point_distance = best_dist
        best_rp.associations_count +=1 # update de niche number
        best_rp.associations += [ind]

# reference_points　は引数から削除する
def niching_select(individuals, k,reference_points_data):
    '''Secondary niched selection based on reference points. Corresponds to
    steps 13-17 of Algorithm 1 and to Algorithm 4.'''
    if len(individuals) == k:
        return individuals

    # 個体とリファレンスポイントの関連づけ
    associate(individuals, reference_points_data)

    # ある競争が必要なフロント内で，生存する個体を格納する
    res = []
    # 生存する個体が一定数を超えない間繰り返す
    while len(res) < k:
        # rp.associations_count は reference_points.associations_count と同義
        # 最も関連が少ないリファレンスポイントを抽出
        # https://qiita.com/komorin0521/items/2fc2335b3008059c19ab
        min_assoc_rp = min(reference_points_data, key=lambda rp: rp.associations_count)
        # 最も関連が少ないリファレンスポイントが複数ある可能性があるため，それら全てを格納する
        min_assoc_rps = [rp for rp in reference_points_data if rp.associations_count == min_assoc_rp.associations_count]
        # 最小のリファレンスポイント群からランダムに1つを選択する
        chosen_rp = min_assoc_rps[random.randint(0, len(min_assoc_rps)-1)]

        #print('Rps',min_assoc_rp.associations_count, chosen_rp.associations_count, len(min_assoc_rps))

        # 選択したリファレンスポイントに関連する個体を代入する
        associated_inds = chosen_rp.associations

        # この時点で提供されている個体（次世代に生存が決まっていない個体内のパレートフロントの解）だけで
        # リファレンスポイントとの関係を作成しておく必要がある．

        if chosen_rp.associations:
            # 選択したリファレンスポイントが最も近傍である個体が存在しない時
            if chosen_rp.associations_count == 0:
                # 最も近傍にある個体を選択する（ここが解が増殖する原因の1つだと考えられる）
                sel = min(chosen_rp.associations, key=lambda ind: ind.ref_point_distance)
            # 普通に選択したリファレンスポイントが最も近傍である個体が存在する時
            else:
                # 選択したリファレンスポイント内の最近傍の個体から，ランダムに(？)選択
                sel = chosen_rp.associations[random.randint(0, len(chosen_rp.associations)-1)]
            # 次世代の解に追加
            res += [sel]
            # 追加した個体を選択した個体から削除する（ここも解が増殖する原因の1つだと考えられる）
            chosen_rp.associations.remove(sel)
            # リファレンスポイントに関連づけられた個体数のカウントを調整
            chosen_rp.associations_count += 1
            # 個体群から選択した個体を削除（ただし，他のリファレンスポイントとの関連づけの値が残っている可能性がある．）
            individuals.remove(sel)
        else:
            # リファレンスポイントから削除する
            reference_points_data.remove(chosen_rp)
    return res

def sel_nsga_iii(individuals, k, reference_points_data):
    '''Implements NSGA-III selection as described in
    Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
    Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
    Part I: Solving Problems With Box Constraints. IEEE Transactions on
    Evolutionary Computation, 18(4), 577–601. doi:10.1109/TEVC.2013.2281535.
    '''
    # 個体数が生存個体数を超過した時にアラート
    assert len(individuals) >= k

    # 個体数が生存個体数より少ない時，そのまま個体を返す
    if len(individuals)==k:
        return individuals

    # パレートフロント毎にソートする
    # Algorithm 1 steps 4--8
    fronts = tools.sortLogNondominated(individuals, len(individuals))

    # 次世代に生存する個体数を調整する
    limit = 0
    # 次世代の個体を追加していく変数?
    res =[]

    # どのパレートで次世代の個体数を超えるかを確認
    for f, front in enumerate(fronts):
        res += front
        if len(res) > k:
            limit = f
            break
    
    # Algorithm 1 steps
    # 次世代の生存個体を保存する変数
    selection = []

    # パレートフロントごと追加する場合の処理
    if limit > 0:
        for f in range(limit):
            selection += fronts[f]

    # 選択が必要なパレートフロントに達した場合の処理
    # complete selected inividuals using the referece point based approach
    selection += niching_select(fronts[limit], k - len(selection), reference_points_data)
    return selection

# 必要な評価値のみを格納する箱
one_nature = []
one_phy = []
two_nature = []
two_culture = []
three_nature = []
three_men = []



# 現時点では，評価値が既に正規化されている想定で書かれている
def best_individuals_show_for_each_reference_point(individuals):
    pareto_fronts = tools.sortLogNondominated(individuals, len(individuals))[0]
    print("len(pareto_fronts)",len(pareto_fronts))

    # リファレンスポイントと個体との関連づけ用に新規のリファレンスポイントを準備する
    ref_data_to_show = []
    # 関数の関係上，numpy.ndarray を list に変換
    ref_points_tolist = ref_points.tolist()
    # 変換したリファレンスポイントを一つずつ取り出して格納する
    for i in range(len(ref_points_tolist)):
        ref_data_to_show.append(ReferencePoint(ref_points_tolist[i]))
    # リファレンスポイントとパレートフロントの関連づけを行う
    associate(pareto_fronts, ref_data_to_show)

    # 特定のリファレンスポイント(memoに対応するもの記載)に対して，出力を行いたいときはこちら
    # 71(自然とphysical_fatigue) 75(自然と文化) 70(自然とmental_fatigue)
    for i in [19,70,71]:
        # 最短距離に対応する個体の番号を取得する
        associations_number = -1
        lowest_associations_value = 10.0
        print("--------------------------------------------------------------------------")
        if i == 71:
            print("自然とphysical_fatigue")
        elif i == 19:
            print("men and phy")
        elif i == 70:
            print("自然とmental_fatigue")
        print("reference point", ref_data_to_show[i])
        print("len(ref_data_to_show[i].associations)",len(ref_data_to_show[i].associations))

        for j in range(len(ref_data_to_show[i].associations)):
            # print("ref_data_to_show[i].associations[j].ref_point_distance:",ref_data_to_show[i].associations[j].ref_point_distance)
            if(lowest_associations_value > ref_data_to_show[i].associations[j].ref_point_distance):
                lowest_associations_value = ref_data_to_show[i].associations[j].ref_point_distance
                associations_number = j
        if(associations_number > -1):
            # print("associations_number",associations_number)
            print("最良個体",ref_data_to_show[i].associations[associations_number])
            inds_value = toolbox.evaluate(ref_data_to_show[i].associations[associations_number])
            print("評価値（正規化済）",ref_data_to_show[i].associations[associations_number].fitness.values)
            print("評価値（絶対値）", inds_value)
            #　自然:0，風景:1，文化:2，食:3，買い物:4，料金:5，phy:6, men:7, spottime:8, 観光移動時間:9，観光歩数:10，スポットの数:11
            # if(i == 71):
            #     one_nature.append(inds_value[0])
            #     print("inds_value[0]",inds_value[0])
            #     one_phy.append(inds_value[6])
            #     print("inds_value[6]",inds_value[6])
            # elif(i == 75):
            #     two_nature.append(inds_value[0])
            #     two_culture.append(inds_value[2])
            # elif(i == 70):
            #     three_nature.append(inds_value[0])
            #     three_men.append(inds_value[7])
                
        else:
            print("最良個体なし")


# # 特定のリファレンスポイント(memoに対応するもの記載)に対して，出力を行いたいときはこちら
#     # 71(自然とphysical_fatigue) 75(自然と文化) 70(自然とmental_fatigue) ##条件NOBJ = 12 , P = 2
#     # 54 55##条件NOBJ = 12 , P = 3
#     for i in [34,49,54,55]:
#         # 最短距離に対応する個体の番号を取得する
#         associations_number = -1
#         lowest_associations_value = 10.0
#         print("--------------------------------------------------------------------------")
#         if i == 34:
#             print("自然0.0,身体的疲労0.0,精神的疲労1.0")
#         elif i == 49:
#             print("自然0.0,身体的疲労0.3,精神的疲労0.7")
#         elif i == 2:
#             print("自然0.0,身体的疲労0.5,精神的疲労0.5")
#         elif i ==54:
#             print("自然0.0,身体的疲労0.7,精神的疲労0.3")
#         elif i ==55:
#             print("自然0.0,身体的疲労1.0,精神的疲労0.0")
#         elif i == 5:
#             print("自然0.5,身体的疲労0.0,精神的疲労0.5")
#         elif i == 6:
#             print("自然0.5,身体的疲労0.5,精神的疲労0.0")

#         print("reference point", ref_data_to_show[i])
#         print("len(ref_data_to_show[i].associations)",len(ref_data_to_show[i].associations))

#         for j in range(len(ref_data_to_show[i].associations)):
#             # print("ref_data_to_show[i].associations[j].ref_point_distance:",ref_data_to_show[i].associations[j].ref_point_distance)
#             if(lowest_associations_value > ref_data_to_show[i].associations[j].ref_point_distance):
#                 lowest_associations_value = ref_data_to_show[i].associations[j].ref_point_distance
#                 associations_number = j
#         if(associations_number > -1):
#             # print("associations_number",associations_number)
#             print("最良個体",ref_data_to_show[i].associations[associations_number])
#             inds_value = toolbox.evaluate(ref_data_to_show[i].associations[associations_number])
#             print("評価値（正規化済）",ref_data_to_show[i].associations[associations_number].fitness.values)
#             print("評価値（絶対値）", inds_value)
#             #　自然:0，風景:1，文化:2，食:3，買い物:4，料金:5，phy:6, men:7, spottime:8, 観光移動時間:9，観光歩数:10，スポットの数:11
#             if(i == 71):
#                 one_nature.append(inds_value[0])
#                 one_phy.append(inds_value[6])
#             elif(i == 75):
#                 two_nature.append(inds_value[0])
#                 two_culture.append(inds_value[2])
#             elif(i == 70):
#                 three_nature.append(inds_value[0])
#                 three_men.append(inds_value[7])
                
#         else:
#             print("最良個体なし")


toolbox = base.Toolbox()
toolbox.register("attr_float", makeRoute.singleCourseData, spotData, tTimeData, stepData, 4, 8) #歩数あり
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", makeRoute.evaluate, spotData, tTimeData, stepData) #歩数あり
toolbox.register("mate", GA.mate)
toolbox.register("mutate", GA.mutate)
toolbox.register("normalized", normalizedInd)
toolbox.register("select",tools.selNSGA3, ref_points=ref_points)
# toolbox.register("select", tools.selNSGA3WithMemory(ref_points))

# 個体数
parents = toolbox.population(n=MU)


for gen in range(1, NGEN):
    # 生成されたコースの評価を行う
    # print("evaluate : Start")

    for i in range(len(parents)):
        # print("parents[i][0]",parents[i][0])
        parents[i].fitness.values = toolbox.evaluate(parents[i])
        # print("evaluate : ",parents[i].fitness.values)
        # print("parents[i].fitness.values",parents[i].fitness.values)

    # リファレンスポイントと個体の関連性を保存する容器
    ref_data = []

    ref_points_tolist = ref_points.tolist()
    for i in range(len(ref_points_tolist)):
        ref_data.append(ReferencePoint(ref_points_tolist[i]))
    
    # for i in range(len(ref_data)):
    #     print(ref_data[i])


    if(debug.individual_check):
        # 初期個体の実際のデータを確認
        parents.sort()
        for k in range(len(parents)):
            print("初期　parents[",k,"]",parents[k])
            print("評価値",parents[k].fitness.values)  

    offsprings = []
    # 交叉
    if(debug.do_mate):
        # print("mate : Start")
        for i in range(MU * CXPB // 200):
            # 交叉する親個体を選ぶセクション
            num = random.sample(range(len(parents)), 2)
            # 選択した親を交叉させるセクション
            child1, child2 = toolbox.mate(toolbox.clone(parents[num[0]]),toolbox.clone(parents[num[1]]),gen)

            if(debug.mate_check):
                print("交叉する親個体A：",num[0],parents[num[0]],"交叉により生まれたchild1：", child1)
                print("交叉する親個体B：",num[1],parents[num[1]],"交叉により生まれたchild2：", child2)
            offsprings.append(child1)
            offsprings.append(child2)
        # print("mate : End")


    # 突然変異
    if(debug.do_mutate):
        # print("mutate : Start")
        for i in random.sample(range(len(parents)),MU * MUTPB // 100):
            # 複製した個体を変異させる
            c = copy.deepcopy(parents[i])
            mutant = toolbox.mutate(c,gen)
            if(debug.mutate_check):
                print("突然変異する親個体A：",i,parents[i])
                print("突然変異した個体：", mutant)
            offsprings.append(mutant)
        # print("mutate : End")


    # 子集団の評価値算出
    # print("offsprings evaluate : Start")
    for i in range(len(offsprings)):
        offsprings[i].fitness.values = toolbox.evaluate(offsprings[i])
    # print("offsprings evaluate : End")

    if(debug.offsprings_duplicate_delete):
        if offsprings:
            offsprings.sort()
            last = offsprings[-1]
            for i in range(len(offsprings)-2, -1, -1):
                if last == offsprings[i]:
                    del offsprings[i]
                else:
                    last = offsprings[i]

    offsprings_num = len(offsprings)

    if(debug.do_normalize):
        # print("最大最小 : 測定　Start")
        makeMinMax(parents + offsprings)
        # print("最大最小 : 測定　End")

        # print("normalized Start -----------------------------------------------------------")
        for i in range(len(parents)):
            parents[i].fitness.values = toolbox.normalized(toolbox.clone(parents[i]))  

        for i in range(len(offsprings)):
            offsprings[i].fitness.values = toolbox.normalized(offsprings[i]) 
        # print("normalized End --------------｀---------------------------------------------")

    t = parents + offsprings
    
    if(debug.duplicate_delete_before_select):
        if t:
            t.sort()
            last = t[-1]
            for i in range(len(t)-2, -1, -1):
                if last == t[i]:
                    del t[i]
                else:
                    last = t[i]
        print("len(t)",len(t))

    if(debug.select_check):
        # 選択アルゴリズム前の個体確認用
        t.sort()
        for k in range(len(t)):   
            print(gen+1,"世代の選択前個体",k,t[k])
            print("評価値",t[k].fitness.values)
        print("--------------------------")

    if(debug.do_select):
        # print("選択アルゴリズム : 開始")
        # parents = toolbox.select(individuals=t, k=MU)
        parents = sel_nsga_iii(t,MU,ref_data)
        # print("選択アルゴリズム : 終了")

    best_individuals_show_for_each_reference_point(parents)

    if(debug.select_check):
        # 選択アルゴリズム後の個体確認用
        parents.sort()
        for k in range(len(parents)):
            print(gen+1,"世代の生存個体",k,parents[k])
            print("評価値",parents[k].fitness.values)


    # print("前len(parents)",len(parents))
    # print(len(parents))
    if(debug.duplicate_delete_after_select):
        if parents:
            parents.sort()
            last = parents[-1]
            for i in range(len(parents)-2, -1, -1):
                if last == parents[i]:
                    del parents[i]
                else:
                    last = parents[i]
        parents_num = len(parents)
        print("後len(parents)",len(parents))

    if(debug.generation_show):
        print(gen+1,"世代")
        print("============================================================================")

    if(debug.offsprings_survive_rate_show):
        # 交叉と突然変異した個体の生存個体数を計算（合算版）
        survive_new_ind = 0
        for i in parents:
            if(i[2] == (gen+1)):
                survive_new_ind += 1
        # print(survive_new_ind,":",((MU * CXPB // 200) + (MU * MUTPB // 100)))
        survive_new_ind_pro.append(survive_new_ind / offsprings_num)
        
        # 生存した個体が子集団よりも大きくなった際に出力する（増殖時のバグチェックに用いた）
        if(survive_new_ind / offsprings_num > 1):
            parents.sort()
            for k in range(len(parents)):
                print(gen+1,"世代の生存個体",k,parents[k])
                print("評価値",parents[k].fitness.values)

    if(debug.hundred_times_survived_individual_show):
        # １００回ごとに出力するやつ
        if((gen+1)%100 == 0):
            print("世代",gen + 1)
            for k in range(len(parents)):
                print("parents[",k,"]", toolbox.evaluate(parents[k]), parents[k])

if(debug.do_associate_individual_reference_point):
    associate(parents, ref_data)
    min_value = [1.0] * len(ref_data)
    min_value_parents_id = [0] * len(ref_data)
    for i in range(len(parents)):
        # num_associate_rp[i] は個体番号[i]のリファレンスポイントとの近さを表す
        # print(num_associate_rp[i])
        for j in range(len(num_associate_rp[i])):
            if(min_value[j] > num_associate_rp[i][j]):
                # print(parents[i])
                # print("maxvalue[", j,"] の値の更新:",num_associate_rp[i][j])
                
                min_value[j] = num_associate_rp[i][j]
                min_value_parents_id[j] = i

    if(debug.associate_individual_by_reference_point_show):
        # 各リファレンスポイントに対応する解を表示するプログラム
        for k in range(len(min_value_parents_id)):
            print("-----------------------------------------------------------------------------------")
            print("右のリファレンスポイントに最も近い親個体を表示する", ref_points[k])
            print("親個体[",k,"]", toolbox.evaluate(parents[min_value_parents_id[k]]), parents[min_value_parents_id[k]])
            # print("正規化した値", kkk[k].fitness.values)
            print("リファレンスラインとの差",min_value[k])
            print("正規化した値", parents[min_value_parents_id[k]].fitness.values)
            print("最も近傍に位置するリファレンスポイント", parents[min_value_parents_id[k]].reference_point)
            print(" ")

# print("ref_points num:", len(ref_points))
print("min",minList)
print("max",maxList)

if(debug.elapsed_time_show):
    elapsed_time = time.time() - start_time
    print ("実行時間:{0}".format(elapsed_time) + "[sec]")

if(debug.individual_generation_show):
    for i in parents:
        # print("ind_gen[i]:",ind_gen[i])
        ind_gen[int(i[3])] += 1

    for i in range(len(ind_gen)):
        if i!= 0:
            print(i,"世代:", ind_gen[i], "個体が生存")

if(debug.offsprings_survive_rate_show):
    print("新たに生成された個体の生存率の推移")
    for i in survive_new_ind_pro:
        print(i)

# print("交叉した個体の生存率の変遷")
# for i in survive_mate_pro:
#     print(i)
# print("突然変異した個体の生存率の変遷")
# for i in survive_mutate_pro:
#     print(i)
    
