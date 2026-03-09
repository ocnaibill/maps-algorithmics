import osmnx as ox
import matplotlib.pyplot as plt

# Pegar um pedaço pequeno do mapa
ponto_central = (-15.7942, -47.8822) 
G = ox.graph_from_point(ponto_central, dist=1000, network_type="drive")

# Desenhar o Grafo Base
fig, ax = plt.subplots(figsize=(12, 12))

# Desenhamos o grafo, mas passamos show=False para não abrir a tela ainda
ox.plot_graph(
    G, ax=ax, show=False, close=False,
    node_size=50, node_color='yellow', node_edgecolor='black', node_zorder=3,
    edge_color='#00CCCC', edge_linewidth=2, edge_alpha=0.7, bgcolor='#222222'
)


# Adicionar Textos: IDs dos Vértices (Nós)
for node, data in G.nodes(data=True):
    x, y = data['x'], data['y']
    # Escreve o ID do nó deslocado alguns pixels para não ficar em cima da bolinha amarela
    ax.annotate(
        str(node), 
        (x, y), 
        xytext=(4, 4), textcoords="offset points", 
        color='white', fontsize=7, zorder=4
    )

#  Adicionar Textos: Nomes das Ruas (Arestas)
for u, v, key, data in G.edges(keys=True, data=True):
    # Verifica se a aresta tem o atributo 'name' (algumas vias de serviço não têm)
    if 'name' in data:
        nome_rua = data['name']
        
        # Às vezes o OSM retorna uma lista de nomes para a mesma rua
        if isinstance(nome_rua, list):
            nome_rua = nome_rua[0]
            
        # Pega as coordenadas dos dois vértices que formam essa rua
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        
        # Calcula o "meio" da rua para colocar o texto
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Escreve o nome da rua no centro da aresta com um fundinho escuro para dar leitura
        ax.annotate(
            nome_rua, 
            (mid_x, mid_y), 
            color='lightgreen', fontsize=8, ha='center', va='center', zorder=5,
            bbox=dict(facecolor='#222222', edgecolor='none', alpha=0.7, pad=0.5)
        )

ax.set_title("Estrutura do Grafo: Vértices e Arestas", color='white', fontsize=16)

# Agora sim mostramos o resultado final!
plt.show()