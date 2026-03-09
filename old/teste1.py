import osmnx as ox
import matplotlib.pyplot as plt

# ==========================================
# 1️⃣ RECORTAR O MAPA PELO NOME OFICIAL
# ==========================================
# Em vez de usar coordenadas, passamos o nome do local.
# O OSMnx vai buscar o polígono oficial dessa região no OpenStreetMap.
local = "Cruzeiro, Distrito Federal, Brazil"

print(f"Buscando as fronteiras oficiais de: {local}...")

# O OSMnx vai recortar o grafo EXATAMENTE no formato do Cruzeiro (Velho e Novo)
G_cruzeiro = ox.graph_from_place(local, network_type="drive")

print(f"Sucesso! Grafo criado com {len(G_cruzeiro.nodes)} cruzamentos e {len(G_cruzeiro.edges)} ruas.")

# ==========================================
# 2️⃣ VISUALIZAR O FORMATO DO RECORTE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))

# Desenhamos o grafo para você confirmar se é a área certa
ox.plot_graph(
    G_cruzeiro, 
    ax=ax, 
    show=False, 
    close=False, 
    node_size=15, 
    node_color='cyan', 
    edge_color='#555555', 
    bgcolor='#111111'
)

ax.set_title("Recorte Exato: Cruzeiro Velho e Cruzeiro Novo", color='white', fontsize=16)
fig.patch.set_facecolor('#111111')

plt.show()