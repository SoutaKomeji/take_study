from math import factorial
import numpy as np
import random
import xlrd
import time

# import matplotlib.pyplot as plt
import numpy
# import pymop.factory
import copy

from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

start_time = time.time()

# デバッグを簡単にする用の変数
do_mate = True
do_mutate = True
do_normalize = True
do_select = True

individual_check = False
mate_check = False
mutate_check = False
select_check = False

offsprings_duplicate_delete = False
duplicate_delete_before_select = True
duplicate_delete_after_select= False

# 世代数の表示
generation_show = True
# 100世代毎に生存個体を表示
hundred_times_survived_individual_show = False

# リファレンスポイントと個体の関連づけを行う
do_associate_individual_reference_point = False
# 最終的にそれぞれのリファレンスポイントに最も近い個体を表示する
associate_individual_by_reference_point_show = True

# 実行時間を表示する
elapsed_time_show = True
# 新個体の生存率を表示する
offsprings_survive_rate_show = False
# 各世代の生存した個体を表示する
individual_generation_show = True

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
SPOT_NUM = 61

# Excelデータの読み込みを始める行番号を指定
row = 26

# Algorithm parameters
MU = 500 #int(H + (4 - H % 4))
NGEN = 100
CXPB = 100 # ％表記でお願いします
MUTPB = 20 # ％表記でお願いします
##

# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)

# 任意のリファレンスポイントの追加
# ref_points = np.append(ref_points,[[0.7,0., 0.,  0.,  0.,  0.,  0.3,  0. ]],axis=0)
# ref_points = np.append(ref_points,[[0.7,0., 0.,  0.,  0.,  0.,  0.,  0.3 ]],axis=0)
# ref_points = np.append(ref_points,[[0.2, 0., 0.,  0.,  0.,  0.,  0.4,  0.4 ]],axis=0)


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
# Excelファイル全体の読み込み
wb = xlrd.open_workbook('preExpData.xlsx')
# Excelファイル内の特定のシートの読み込み
featureSheet = wb.sheet_by_name('特徴表')
tTimeSheet = wb.sheet_by_name('スポット間移動時間')

for i in range(SPOT_NUM + 1):
    spotData.append(featureSheet.row_values(i+26,1,7))
    tTimeData.append(tTimeSheet.row_values(i+1,2,SPOT_NUM + 3))

## 観光スポットのデータ作成終了

# ここはクラスを作っているだけ
# Create classes
creator.create("FitnessMin", base.Fitness, weights=(1.0,)* NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)

# 観光コース内のコースを回る順番を作成
def singleCourseData(spotData, tTimeData, minSpotNum, maxSpotNum):
    nobj = len(spotData[0])
    spotNum = random.randint(minSpotNum, maxSpotNum)

    route = []
    # 出発地点の追加(函館駅を指定)
    route.append(61)

    # 全スポットから重複なしでランダムにスポットを選択する
    # range(x) は 0 から x-1 までの値を指す
    route.extend(random.sample(range(SPOT_NUM), k=spotNum))
    
    # 終着地点の追加(函館駅を指定)
    route.append(61)

    time = 0
    for j in range(len(route) - 1):
        time += tTimeData[route[j]][route[j+1]] + 20
    
    routeData = []
    routeData.append(route)
    routeData.append(time)

    routeData.append(0)

    return routeData

# コース評価用の関数
def evaluate(spotData, tTimeData ,inds):#参照しているのは単独の個体
    nature = 0
    landscape = 0
    culture = 0
    food = 0
    shopping = 0
    admission = 0
    time = 0
    num = 0
    # コース内の観光スポット数分

    ind = toolbox.clone(inds)

    for j in ind[0][1:-1]:
        nature += max(spotData[j][0] - 1.8, 0)
        landscape += max(spotData[j][1] - 1.8, 0)
        culture += max(spotData[j][2] - 1.8, 0)
        food += max(spotData[j][3] - 1.8, 0)
        shopping += max(spotData[j][4] - 1.8, 0)
        admission += spotData[j][5]
        
    # 出発地点と終着地点を省略
    time = ind[1]
    num = len(ind[0]) - 2

    return nature, landscape, culture, food, shopping, admission, time, num

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
    t.remove(61)
    # if(t):
    for i in t:
        loot = random.randint(0,1)

        if(loot == 0):
            ind1Front.remove(i)
        else:
            ind2Behind.remove(i)
    
    # 重複があった場合に，ランダムに削除する(course2用)
    t = [x for x in set(ind2Front + ind1Behind) if (ind2Front + ind1Behind).count(x) > 1]
    t.remove(61)
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

    time = 0
    for j in range(len(course1) - 1):
        time += tTimeData[course1[j]][course1[j+1]] + 20
    child1.append(time)
    child1.append(gen + 1)

    time = 0
    for j in range(len(course2) - 1):
        time += tTimeData[course2[j]][course2[j+1]] + 20
    child2.append(time)
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
        t.remove(61)
        addSpot = random.choice(list(t))
        # 観光スポットを追加する場所を選択
        addPoint = random.randint(1,len(ind[0]) - 1)
        ind[0].insert(addPoint,addSpot)

    # 減少変異
    elif(loot == 1):
        t = set(ind[0])^set([SPOT_NUM]) #スタート地点を削除する
        removeSpot = random.choice(list(t))
        ind[0].remove(removeSpot)

    time = 0
    for j in range(len(ind[0]) - 1):
        time += tTimeData[ind[0][j]][ind[0][j+1]] + 20
    ind[1] = time
    ind[2] = gen + 1
    
    return ind

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
            print("最大値と最小値が同じである")
            evaList[j] = 0.5
        elif(j==5 or j==6):
            evaList[j] = 1 - ((evaList[j] - minList[j]) / (maxList[j] - minList[j]))
        else:
            evaList[j] = (evaList[j] - minList[j]) / (maxList[j] - minList[j])

    # 制限時間を超過した場合に，評価値にペナルティを与えている．
    if(timeLimit < ind[1]):
        for j in range(len(evaList)):
            evaList[j] -= 1.0 * ((ind[1] - timeLimit) / 200)

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

# def generate_reference_points(num_objs, num_divisions_per_obj=4):
#     '''Generates reference points for NSGA-III selection. This code is based on
#     `jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>`_.
#     '''
#     # work_point : [0] * 目的関数の数
#     # num_objs   : 目的関数の数
#     # left       : 
#     def gen_refs_recursive(work_point, num_objs, left, total, depth):
#         if depth == num_objs - 1:
#             work_point[depth] = left/total
#             ref = ReferencePoint(copy.deepcopy(work_point))
#             return [ref]
#         else:
#             res = []
#             for i in range(left):
#                 work_point[depth] = i/total
#                 res = res + gen_refs_recursive(work_point, num_objs, left-i, total, depth+1)
#             return res
#     return gen_refs_recursive([0]*num_objs, num_objs, num_objs*num_divisions_per_obj,
#                               num_objs*num_divisions_per_obj, 0)

def associate(individuals, reference_points_data):
    '''Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014).'''
    pareto_fronts = tools.sortLogNondominated(individuals, len(individuals))
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        rp_dists = [(rp, perpendicular_distance(rp, ind.fitness.values))
                    for rp in reference_points_data]

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

    #individuals = copy.deepcopy(individuals)

    # ideal_point = find_ideal_point(individuals)
    # extremes = find_extreme_points(individuals)
    # intercepts = construct_hyperplane(individuals, extremes)
    # normalize_objectives(individuals, intercepts, ideal_point)

    # reference_points = generate_reference_points(len(individuals[0].fitness.values))

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
one_move = []
two_nature = []
two_calture = []


# 現時点では，評価値が既に正規化されている想定で書かれている．
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

    # 全てのリファレンスポイントに対して，出力を行いたいときはこちら
    # for i in range(len(ref_data_to_show)):
    #     # 最短距離に対応する個体の番号を取得する
    #     associations_number = -1
    #     lowest_associations_value = 10.0
    #     print("reference point", ref_data_to_show[i])
    #     # for j in range(len(ref_data_to_show[i].associations)):
    #         # print(ref_data_to_show[i].associations[j])
    #         # print(ref_data_to_show[i].associations[j].fitness.values)
    #     for j in range(len(ref_data_to_show[i].associations)):
    #         if(lowest_associations_value > ref_data_to_show[i].associations[j].ref_point_distance):
    #             lowest_associations_value = ref_data_to_show[i].associations[j].ref_point_distance
    #             associations_number = j
    #     if(associations_number > -1):
    #         # print("associations_number",associations_number)
    #         print("最良個体",ref_data_to_show[i].associations[associations_number])
    #         print("評価値（正規化済）",ref_data_to_show[i].associations[associations_number].fitness.values)
    #         print("評価値（絶対値）", toolbox.evaluate(ref_data_to_show[i].associations[associations_number]))
    #     else:
    #         print("最良個体なし")

    for i in [29,34]:
        # 最短距離に対応する個体の番号を取得する
        associations_number = -1
        lowest_associations_value = 10.0
        print("reference point", ref_data_to_show[i])
        # for j in range(len(ref_data_to_show[i].associations)):
            # print(ref_data_to_show[i].associations[j])
            # print(ref_data_to_show[i].associations[j].fitness.values)
        for j in range(len(ref_data_to_show[i].associations)):
            if(lowest_associations_value > ref_data_to_show[i].associations[j].ref_point_distance):
                lowest_associations_value = ref_data_to_show[i].associations[j].ref_point_distance
                associations_number = j
        if(associations_number > -1):
            # print("associations_number",associations_number)
            print("最良個体",ref_data_to_show[i].associations[associations_number])
            inds_value = toolbox.evaluate(ref_data_to_show[i].associations[associations_number])
            print("評価値（正規化済）",ref_data_to_show[i].associations[associations_number].fitness.values)
            print("評価値（絶対値）", inds_value)
            if(i == 29):
                one_nature.append(inds_value[0])
                one_move.append(inds_value[6])
            elif(i == 34):
                two_nature.append(inds_value[0])
                two_calture.append(inds_value[1])
                
        else:
            print("最良個体なし")

# def aaaaaaaaaaaaaaa(inds, reference_points_data):
#     # 必要なもの：
#     # 全個体（もしくはパレートフロントの個体）
#     # 全リファレンスポイント

#     # パレートフロントの抽出
#     pareto_fronts = tools.sortLogNondominated(inds, len(inds))
#     # 個体とリファレンスポイントの関連づけ
#     associate(pareto_fronts, reference_points_data)
#     # 



toolbox = base.Toolbox()
toolbox.register("attr_float", singleCourseData, spotData, tTimeData, 4, 8)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate, spotData, tTimeData)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("normalized", normalizedInd)
toolbox.register("select",tools.selNSGA3, ref_points=ref_points)
# toolbox.register("select", tools.selNSGA3WithMemory(ref_points))

parents = toolbox.population(n=MU)


# if parents:
#     parents.sort()
#     last = parents[-1]
#     for i in range(len(parents)-2, -1, -1):
#         if last == parents[i]:
#             del parents[i]
#         else:
#             last = parents[i]
# print("first len",len(parents))

for gen in range(NGEN):
    # 生成されたコースの評価を行う
    # print("evaluate : Start")

    for i in range(len(parents)):
        # print("parents[i][0]",parents[i][0])
        parents[i].fitness.values = toolbox.evaluate(parents[i])
    # print("evaluate : End")
    
    # pareto_fronts = tools.sortLogNondominated(parents, len(parents))
    # print("type",type(pareto_fronts))
    # print("len",len(pareto_fronts))
    # for i in range(len(pareto_fronts)):
    #     print(pareto_fronts[i])

    # print(type(ref_points))
    # print(ref_points.tolist())

    # リファレンスポイントと個体の関連性を保存する容器
    ref_data = []

    ref_points_tolist = ref_points.tolist()
    for i in range(len(ref_points_tolist)):
        ref_data.append(ReferencePoint(ref_points_tolist[i]))
    
    # for i in range(len(ref_data)):
    #     print(ref_data[i])


    if(individual_check):
        # 初期個体の実際のデータを確認
        parents.sort()
        for k in range(len(parents)):
            print("初期　parents[",k,"]",parents[k])
            print("評価値",parents[k].fitness.values)  

    # k = np.dot(ref_points[0], parents[0].fitness.values) / np.sum(np.power(ref_points[0], 2))   
    # print("k:",k)
    # print("ref_points[1]:",ref_points[1])
    # print("np.sum(np.power(ref_points[0], 2)",  np.sum(np.power(ref_points[1], 2)))

    offsprings = []
    # 交叉
    if(do_mate):
        # print("mate : Start")
        for i in range(MU * CXPB // 200):
            # 交叉する親個体を選ぶセクション
            num = random.sample(range(len(parents)), 2)
            # 選択した親を交叉させるセクション
            child1, child2 = toolbox.mate(toolbox.clone(parents[num[0]]),toolbox.clone(parents[num[1]]),0)
            if(mate_check):
                print("交叉する親個体A：",num[0],parents[num[0]],"交叉により生まれたchild1：", child1)
                print("交叉する親個体B：",num[1],parents[num[1]],"交叉により生まれたchild2：", child2)
            offsprings.append(child1)
            offsprings.append(child2)
        # print("mate : End")


    # 突然変異
    if(do_mutate):
        # print("mutate : Start")
        for i in random.sample(range(len(parents)),MU * MUTPB // 100):
            # 複製した個体を変異させる
            c = copy.deepcopy(parents[i])
            mutant = toolbox.mutate(c,0)
            if(mutate_check):
                print("突然変異する親個体A：",i,parents[i])
                print("突然変異した個体：", mutant)
            offsprings.append(mutant)
        # print("mutate : End")


    # 子集団の評価値算出
    # print("offsprings evaluate : Start")
    for i in range(len(offsprings)):
        offsprings[i].fitness.values = toolbox.evaluate(offsprings[i])
    # print("offsprings evaluate : End")

    if(offsprings_duplicate_delete):
        if offsprings:
            offsprings.sort()
            last = offsprings[-1]
            for i in range(len(offsprings)-2, -1, -1):
                if last == offsprings[i]:
                    del offsprings[i]
                else:
                    last = offsprings[i]

    offsprings_num = len(offsprings)

    if(do_normalize):
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
    
    if(duplicate_delete_before_select):
        if t:
            t.sort()
            last = t[-1]
            for i in range(len(t)-2, -1, -1):
                if last == t[i]:
                    del t[i]
                else:
                    last = t[i]
        print("len(t)",len(t))

    if(select_check):
        # 選択アルゴリズム前の個体確認用
        t.sort()
        for k in range(len(t)):   
            print(gen+1,"世代の選択前個体",k,t[k])
            print("評価値",t[k].fitness.values)
        print("--------------------------")

    if(do_select):
        # print("選択アルゴリズム : 開始")
        # parents = toolbox.select(individuals=t, k=MU)
        parents = sel_nsga_iii(t,MU,ref_data)
        # print("選択アルゴリズム : 終了")

    best_individuals_show_for_each_reference_point(parents)

    if(select_check):
        # 選択アルゴリズム後の個体確認用
        parents.sort()
        for k in range(len(parents)):
            print(gen+1,"世代の生存個体",k,parents[k])
            print("評価値",parents[k].fitness.values)


    # print("前len(parents)",len(parents))
    # print(len(parents))
    if(duplicate_delete_after_select):
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

    # for k in range(len(parents)):
    #     print("parents[",k,"]", parents[k].fitness.values)

    if(generation_show):
        print(gen+1,"世代")

    # print("親",parents_num," 子",offsprings_num)
    # 交叉と突然変異した個体の生存個体数を計算
    # survive_mate_ind = 0
    # survive_mutate_ind = 0
    # for i in parents:
    #     if(i[2] == 1):
    #         survive_mate_ind += 1
    #         i[2] = 0
    #     elif(i[2] == 2):
    #         survive_mutate_ind += 1
    #         i[2] = 0
    #
    # print("交叉の生存率:",survive_mate_ind / (MU * CXPB // 200))
    # print("突然変異の生存率",survive_mutate_ind / (MU * MUTPB // 100))
    # survive_mate_pro.append(survive_mate_ind / (MU * CXPB // 200))
    # survive_mutate_pro.append(survive_mutate_ind / (MU * MUTPB // 100))

    if(offsprings_survive_rate_show):
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

    if(hundred_times_survived_individual_show):
        # １００回ごとに出力するやつ
        if((gen+1)%100 == 0):
            print("世代",gen + 1)
            for k in range(len(parents)):
                print("parents[",k,"]", toolbox.evaluate(parents[k]), parents[k])

if(do_associate_individual_reference_point):
    associate(parents, ref_data)
    min_value = [1.0] * len(ref_data)
    min_value_parents_id = [0] * len(ref_data)
    for i in range(len(parents)):
        # print("個体番号", i)
        # num_associate_rp[i] は個体番号[i]のリファレンスポイントとの近さを表す
        # print(num_associate_rp[i])
        for j in range(len(num_associate_rp[i])):
            if(min_value[j] > num_associate_rp[i][j]):
                # print(parents[i])
                # print("maxvalue[", j,"] の値の更新:",num_associate_rp[i][j])
                
                min_value[j] = num_associate_rp[i][j]
                min_value_parents_id[j] = i

    if(associate_individual_by_reference_point_show):
        # 各リファレンスポイントに対応する解を表示するプログラム
        for k in range(len(min_value_parents_id)):
            print("-----------------------------------------------------------------------------------")
            print("右のリファレンスポイントに最も近い親個体を表示する", ref_points[k])
            # print("num_associate_rp[i][j]",num_associate_rp[min_value_parents_id[k]][k])
            print("親個体[",k,"]", toolbox.evaluate(parents[min_value_parents_id[k]]), parents[min_value_parents_id[k]])
            # print("正規化した値", kkk[k].fitness.values)
            print("リファレンスラインとの差",min_value[k])
            print("正規化した値", parents[min_value_parents_id[k]].fitness.values)
            print("最も近傍に位置するリファレンスポイント", parents[min_value_parents_id[k]].reference_point)
            print(" ")

# best_individuals_show_for_each_reference_point(parents)

# parents.sort()
# for i in range(len(parents)):
#     print(parents[i])

print("自然")
for i in one_nature:
    print(i)

print("移動時間")
for i in one_move:
    print(i)

print("自然2")
for i in two_nature:
    print(i)

print("歴史")
for i in two_calture:
    print(i)


# print("ref_points num:", len(ref_points))
print("min",minList)
print("max",maxList)

if(elapsed_time_show):
    elapsed_time = time.time() - start_time
    print ("実行時間:{0}".format(elapsed_time) + "[sec]")

if(individual_generation_show):
    for i in parents:
        ind_gen[i[2]] += 1
    for i in range(len(ind_gen)):
        print(i,"世代:", ind_gen[i], "個体が生存")

if(offsprings_survive_rate_show):
    print("新たに生成された個体の生存率の推移")
    for i in survive_new_ind_pro:
        print(i)

# print("交叉した個体の生存率の変遷")
# for i in survive_mate_pro:
#     print(i)
# print("突然変異した個体の生存率の変遷")
# for i in survive_mutate_pro:
#     print(i)
    



# print("-----------------------------------------------------------------------------------")

# kkk = toolbox.select(parents, MU // baisuu)
# for k in range(len(kkk)):
    
#     print("parents[",k,"]", toolbox.evaluate(kkk[k]), kkk[k])
#     # print("正規化した値", kkk[k].fitness.values)
#     print("parents", kkk[k].reference_point)



# print(ref_points)
