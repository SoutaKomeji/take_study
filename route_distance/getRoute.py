import osmnx as ox
import networkx as nx

# OSMの地図データを取得する場所を指定（例：東京）
place_name = "Hokkaido, Japan"
# 道路ネットワークを取得
G = ox.graph_from_place(place_name, network_type='drive')

# 緯度・経度のリスト（出発地と目的地）
route_points = [(35.681236, 139.767125),  # 東京駅
                (34.693737, 135.502165)]  # 大阪駅

# ルートの最寄りノードをネットワーク内で探す
start_node = ox.distance.nearest_nodes(G, route_points[0][1], route_points[0][0])
end_node = ox.distance.nearest_nodes(G, route_points[1][1], route_points[1][0])

# 最短経路（道のり）を計算
shortest_route = nx.shortest_path(G, start_node, end_node, weight='length')

# 経路の総距離を計算
route_length = nx.shortest_path_length(G, start_node, end_node, weight='length')

# キロメートルに変換
route_length_km = route_length / 1000
print(f"道のりに基づく合計距離: {route_length_km:.2f} km")