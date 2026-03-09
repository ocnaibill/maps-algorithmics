import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import time

#  Baixar mapa real
local = "Brasilia, Distrito Federal, Brazil"
G = ox.graph_from_place(local, network_type="drive")

origem = (-15.8303, -47.9202)   # 114 Sul
destino = (-15.7375, -47.8918)  # 216 Norte

origem_node = ox.distance.nearest_nodes(G, origem[1], origem[0])
destino_node = ox.distance.nearest_nodes(G, destino[1], destino[0])

# Heurística (Corrigida para Metros)
def heuristica(n1, n2):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    
    # Distância em graus Euclidiana
    dist_graus = ((x1-x2)**2 + (y1-y2)**2)**0.5
    
    # Multiplicamos por ~111139 para converter graus em metros 
    # para ficar na mesma unidade do G[u][v][0]['length']
    return dist_graus * 111139

#  A* Manual
def a_star_manual(G, inicio, fim):
    fila = []
    heapq.heappush(fila, (0, inicio))

    g_custo = {inicio: 0}
    pais = {}
    visitados = []
    contador = 0

    inicio_tempo = time.time()

    while fila:
        _, atual = heapq.heappop(fila)
        visitados.append(atual)
        contador += 1

        if atual == fim:
            break

        for vizinho in G.neighbors(atual):
            peso = G[atual][vizinho][0]['length']
            novo_g = g_custo[atual] + peso

            if vizinho not in g_custo or novo_g < g_custo[vizinho]:
                g_custo[vizinho] = novo_g
                f = novo_g + heuristica(vizinho, fim)
                pais[vizinho] = atual
                heapq.heappush(fila, (f, vizinho))

    tempo_total = time.time() - inicio_tempo

    caminho = []
    atual = fim
    while atual in pais:
        caminho.append(atual)
        atual = pais[atual]
    caminho.append(inicio)
    caminho.reverse()

    return caminho, visitados, contador, tempo_total

# ==============================
# Dijkstra Manual
def dijkstra_manual(G, inicio, fim):
    fila = []
    heapq.heappush(fila, (0, inicio))

    dist = {inicio: 0}
    pais = {}
    visitados = []
    contador = 0

    inicio_tempo = time.time()

    while fila:
        custo_atual, atual = heapq.heappop(fila)
        visitados.append(atual)
        contador += 1

        if atual == fim:
            break

        for vizinho in G.neighbors(atual):
            peso = G[atual][vizinho][0]['length']
            novo_custo = custo_atual + peso

            if vizinho not in dist or novo_custo < dist[vizinho]:
                dist[vizinho] = novo_custo
                pais[vizinho] = atual
                heapq.heappush(fila, (novo_custo, vizinho))

    tempo_total = time.time() - inicio_tempo

    caminho = []
    atual = fim
    while atual in pais:
        caminho.append(atual)
        atual = pais[atual]
    caminho.append(inicio)
    caminho.reverse()

    return caminho, visitados, contador, tempo_total

# Executar ambos
caminho_a, visitados_a, nos_a, tempo_a = a_star_manual(G, origem_node, destino_node)
caminho_d, visitados_d, nos_d, tempo_d = dijkstra_manual(G, origem_node, destino_node)

print(f"A* → Tempo: {tempo_a:.4f}s | Nós visitados: {nos_a}")
print(f"Dijkstra → Tempo: {tempo_d:.4f}s | Nós visitados: {nos_d}")

# Animação (Lado a Lado)
# Cria figura com 2 eixos (1 linha, 2 colunas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.canvas.manager.set_window_title('Comparação A* vs Dijkstra')

# Desenha o grafo base em ambos APENAS UMA VEZ
ox.plot_graph(G, ax=ax1, show=False, close=False, node_size=0, edge_color='#555555', edge_alpha=0.5)
ox.plot_graph(G, ax=ax2, show=False, close=False, node_size=0, edge_color='#555555', edge_alpha=0.5)

# Pega coordenadas totais dos nós visitados
coords_a = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in visitados_a]
coords_d = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in visitados_d]

# Cria os objetos scatter vazios (vamos apenas atualizar os dados deles)
scat_a = ax1.scatter([], [], c='cyan', s=15, zorder=3)
scat_d = ax2.scatter([], [], c='magenta', s=15, zorder=3)

max_frames = max(len(coords_a), len(coords_d))
passo = max(1, max_frames // 100) # Atualiza a cada 1% para não demorar muito

# Loop da Animação
for i in range(1, max_frames + passo, passo):
    idx_a = min(i, len(coords_a))
    idx_d = min(i, len(coords_d))
    
    # Atualiza as coordenadas dos pontos
    scat_a.set_offsets(coords_a[:idx_a])
    scat_d.set_offsets(coords_d[:idx_d])
    
    # Atualiza os títulos dinamicamente
    ax1.set_title(f"A* - Explorando... ({idx_a} nós)", color='cyan', fontsize=14)
    ax2.set_title(f"Dijkstra - Explorando... ({idx_d} nós)", color='magenta', fontsize=14)
    
    # Pausa rápida para renderizar a tela
    plt.pause(0.01)

# Desenha a linha final do caminho por cima de tudo
route_a_x, route_a_y = zip(*[(G.nodes[n]['x'], G.nodes[n]['y']) for n in caminho_a])
ax1.plot(route_a_x, route_a_y, c='red', linewidth=3, zorder=4)
ax1.set_title(f"A* - Concluído! ({len(visitados_a)} nós visitados)", color='white', fontsize=14)

route_d_x, route_d_y = zip(*[(G.nodes[n]['x'], G.nodes[n]['y']) for n in caminho_d])
ax2.plot(route_d_x, route_d_y, c='red', linewidth=3, zorder=4)
ax2.set_title(f"Dijkstra - Concluído! ({len(visitados_d)} nós visitados)", color='white', fontsize=14)

# Estilização de fundo
fig.patch.set_facecolor('#111111')
ax1.set_facecolor('#111111')
ax2.set_facecolor('#111111')

plt.show()