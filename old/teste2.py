import osmnx as ox
import matplotlib.pyplot as plt

# ==========================================
# 1️⃣ RECORTAR O MAPA PELO NOME OFICIAL
# ==========================================
local = "Cruzeiro, Distrito Federal, Brazil"
print(f"Buscando as fronteiras oficiais de: {local}...")

# Baixando com algumas tags extras caso existam
if 'surface' not in ox.settings.useful_tags_way:
    ox.settings.useful_tags_way.append('surface')

G_cruzeiro = ox.graph_from_place(local, network_type="drive")
print(f"Sucesso! Grafo criado com {len(G_cruzeiro.nodes)} cruzamentos e {len(G_cruzeiro.edges)} ruas.\n")

# ==========================================
# 2️⃣ OUTPUT: RAIO-X DOS DADOS (Vértices e Arestas)
# ==========================================
print("=" * 60)
print("🕵️‍♂️ ANALISANDO OS DADOS DO GRAFO DO CRUZEIRO")
print("=" * 60)

# A) Amostra de Vértices (Nós)
print("\n📍 AMOSTRA DE VÉRTICES (Cruzamentos):")
# Pegamos apenas os 3 primeiros nós para olhar
for no, dados in list(G_cruzeiro.nodes(data=True))[:3]:
    print(f"ID: {no}")
    print(f"Atributos: {dados}\n")

# B) Amostra de Arestas (Ruas)
print("🛣️ AMOSTRA DE ARESTAS (Ruas):")
# Pegamos apenas as 3 primeiras ruas para olhar
for u, v, key, dados in list(G_cruzeiro.edges(keys=True, data=True))[:3]:
    # Filtrando só os atributos principais para facilitar a leitura
    atributos_uteis = {k: v for k, v in dados.items() if k in ['name', 'highway', 'length', 'maxspeed', 'lanes', 'oneway', 'surface']}
    print(f"Rua conectando {u} -> {v}")
    print(f"Atributos Principais: {atributos_uteis}\n")

# C) Resumo Geral do Bairro
tipos_vias = set()
velocidades = set()
superficies = set()

for u, v, k, dados in G_cruzeiro.edges(keys=True, data=True):
    # Pega tipos de via (highway)
    hw = dados.get('highway')
    if isinstance(hw, list): tipos_vias.update(hw)
    elif hw: tipos_vias.add(hw)
        
    # Pega velocidades (maxspeed)
    spd = dados.get('maxspeed')
    if isinstance(spd, list): velocidades.update(spd)
    elif spd: velocidades.add(spd)
        
    # Pega pavimentação (surface)
    sfc = dados.get('surface')
    if isinstance(sfc, list): superficies.update(sfc)
    elif sfc: superficies.add(sfc)

print("📊 RESUMO DOS DADOS DISPONÍVEIS NO BAIRRO:")
print(f"- Tipos de vias ('highway'): {tipos_vias}")
print(f"- Velocidades registradas ('maxspeed'): {velocidades}")
print(f"- Tipos de pavimento ('surface'): {superficies}")
print("=" * 60 + "\n")

# ==========================================
# 3️⃣ VISUALIZAR O FORMATO DO RECORTE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))

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