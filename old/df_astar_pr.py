import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq

# CONFIGURAÇÃO E DOWNLOAD DO MAPA
if 'surface' not in ox.settings.useful_tags_way:
    ox.settings.useful_tags_way.append('surface')

local = "Brasilia, Distrito Federal, Brazil"
print("Baixando mapa...")
G = ox.graph_from_place(local, network_type="drive")

#Aqui defiinimos ponto inicial e final utilizando latitude e longitude do teste
origem = (-15.8303, -47.9202)   # 114 Sul
destino = (-15.7375, -47.8918)  # 216 Norte

origem_node = ox.distance.nearest_nodes(G, origem[1], origem[0])
destino_node = ox.distance.nearest_nodes(G, destino[1], destino[0])

# IDENTIFICAR SEMÁFOROS E CRIAR PESOS
semaforos = set()
for node, data in G.nodes(data=True):
    if data.get('highway') == 'traffic_signals':
        semaforos.add(node)

for u, v, key, data in G.edges(keys=True, data=True):
    distancia = data.get('length', 0)
    tipo_via = data.get('highway', '')
    superficie = data.get('surface', 'unknown')
    
    multiplicador = 1.0 
    
    # Penalidades
    if tipo_via == 'residential': multiplicador += 0.3
    elif tipo_via == 'unclassified': multiplicador += 0.5
        
    if superficie in ['unpaved', 'dirt', 'gravel']: multiplicador += 2.0
    elif superficie == 'cobblestone': multiplicador += 0.5
        
    custo_final = distancia * multiplicador
    if v in semaforos: custo_final += 50 
        
    # Salva o novo peso
    G[u][v][key]['custo_personalizado'] = custo_final

# HEURÍSTICA E A* MANUAL (Para Animação)
def heuristica(n1, n2):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    return ((x1-x2)**2 + (y1-y2)**2)**0.5 * 111139

def a_star_animado(G, inicio, fim):
    fila = []
    heapq.heappush(fila, (0, inicio))

    g_custo = {inicio: 0}
    pais = {}
    visitados = [] # Aqui é onde a mágica da animação acontece!

    while fila:
        _, atual = heapq.heappop(fila)
        visitados.append(atual)

        if atual == fim:
            break

        for vizinho in G.neighbors(atual):
            # ATENÇÃO AQUI: Agora ele puxa o nosso custo_personalizado em vez do 'length'
            peso = G[atual][vizinho][0].get('custo_personalizado', G[atual][vizinho][0]['length'])
            novo_g = g_custo[atual] + peso

            if vizinho not in g_custo or novo_g < g_custo[vizinho]:
                g_custo[vizinho] = novo_g
                f = novo_g + heuristica(vizinho, fim)
                pais[vizinho] = atual
                heapq.heappush(fila, (f, vizinho))

    # Reconstruindo o caminho
    caminho = []
    atual = fim
    while atual in pais:
        caminho.append(atual)
        atual = pais[atual]
    caminho.append(inicio)
    caminho.reverse()

    return caminho, visitados

# EXECUÇÃO E ANIMAÇÃO
print("Calculando rota com A* inteligente...")
caminho_a, visitados_a = a_star_animado(G, origem_node, destino_node)

print("Gerando visualização...")
fig, ax = plt.subplots(figsize=(10, 10))
fig.canvas.manager.set_window_title('Exploração do A*')

# Desenha o mapa base uma vez só
ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0, edge_color='#555555', edge_alpha=0.5)

# Coordenadas para a animação
coords_a = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in visitados_a]
scat_a = ax.scatter([], [], c='cyan', s=15, zorder=3)

passo = max(1, len(coords_a) // 100)

# Loop da Animação
for i in range(1, len(coords_a) + passo, passo):
    idx = min(i, len(coords_a))
    scat_a.set_offsets(coords_a[:idx])
    ax.set_title(f"A* Explorando... ({idx} nós visitados)", color='cyan', fontsize=14)
    plt.pause(0.01)

# Desenha a rota final
route_x, route_y = zip(*[(G.nodes[n]['x'], G.nodes[n]['y']) for n in caminho_a])
ax.plot(route_x, route_y, c='red', linewidth=3, zorder=4)
ax.set_title(f"A* - Concluído! Rota Encontrada.", color='white', fontsize=14)

# Estilo Dark Mode
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')

plt.show()