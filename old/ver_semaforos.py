import osmnx as ox
import matplotlib.pyplot as plt


# Por padrão, o OSMnx ignora 'traffic_calming' nos nós. Temos que forçar ele a baixar:
if 'traffic_calming' not in ox.settings.useful_tags_node:
    ox.settings.useful_tags_node.append('traffic_calming')

local = "Brasilia, Distrito Federal, Brazil"
print(f"Baixando o mapa de {local}... (Isso pode levar alguns segundos)")
G = ox.graph_from_place(local, network_type="drive")

#  separação de elementos
semaforos_x, semaforos_y = [], []
lombadas_x, lombadas_y = [], []
radares_x, radares_y = [], []

for node, data in G.nodes(data=True):
    
    # Caçando Semáforos
    if data.get('highway') == 'traffic_signals':
        semaforos_x.append(data['x'])
        semaforos_y.append(data['y'])
        
    # Caçando Lombadas / Quebra-molas
    if data.get('traffic_calming') in ['bump', 'hump']:
        lombadas_x.append(data['x'])
        lombadas_y.append(data['y'])
        
    # Caçando Radares de Velocidade
    if data.get('highway') == 'speed_camera':
        radares_x.append(data['x'])
        radares_y.append(data['y'])

print("-" * 40)
print(f"🚦 Semáforos encontrados: {len(semaforos_x)}")
print(f"🏔️ Lombadas encontradas: {len(lombadas_x)}")
print(f"📸 Radares encontrados: {len(radares_x)}")
print("-" * 40)

# prova visual
print("Gerando visualização...")
fig, ax = plt.subplots(figsize=(12, 12))

# Desenha as ruas bem fraquinhas no fundo
ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0, edge_color='#333333', edge_linewidth=0.5)

# Plota os elementos com cores diferentes
if semaforos_x:
    ax.scatter(semaforos_x, semaforos_y, c='red', s=30, zorder=5, label=f'Semáforos ({len(semaforos_x)})')
if lombadas_x:
    ax.scatter(lombadas_x, lombadas_y, c='orange', s=30, zorder=5, label=f'Lombadas ({len(lombadas_x)})')
if radares_x:
    ax.scatter(radares_x, radares_y, c='cyan', s=30, zorder=5, label=f'Radares ({len(radares_x)})')

# Estilização
ax.set_title("Mapeamento Urbano no OpenStreetMap - Brasília", color='white', fontsize=16)
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')
plt.legend(facecolor='#222222', labelcolor='white', loc='upper left')

plt.show()