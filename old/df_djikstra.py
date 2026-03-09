import osmnx as ox
import networkx as nx # networkX usa distância euclidiana no A* como heurística
import matplotlib.pyplot as plt

#Aqui baixamos o mapa real do OpenStreetMap
local = "Brasilia, Distrito Federal, Brazil"
G = ox.graph_from_place(local, network_type="drive")

#Explicações : 
# graph_from_place é quem baixa os mapas do OpenStreetMap,
# network_type é o tipo de rua, ao colocar "drive", pedimos ruas que servem para carros
#--------------

#Aqui defiinimos ponto inicial e final utilizando latitude e longitude do teste
origem = (-15.7942, -47.8822)   # Rodoviária do Plano Piloto
destino = (-15.7998, -47.8645)  # Congresso Nacional

#Agora temos que encontrar os nós mais próximos
origem_node = ox.distance.nearest_nodes(G, origem[1], origem[0])
destino_node = ox.distance.nearest_nodes(G, destino[1], destino[0])
#Explicação: nearest_nodes é quem converte as coordenadas em nós para o grafo
#--------------

#Execução do Djikstra usando apenas distâncias como peso
rota = nx.dijkstra_path(
    G,
    origem_node,
    destino_node,
    weight="length" # é quem usa a distância real da rua
)

#Visualização
fig, ax = ox.plot_graph_route(G, rota, route_linewidth=4, node_size=0)