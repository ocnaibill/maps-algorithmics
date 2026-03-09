import osmnx as ox
import networkx as nx
import math

# ==========================================
# 1️⃣ CONFIGURAÇÕES E DOWNLOAD
# ==========================================
print("Carregando o sistema... Baixando mapa do Cruzeiro (DF)...")
if 'traffic_calming' not in ox.settings.useful_tags_node:
    ox.settings.useful_tags_node.append('traffic_calming')

G = ox.graph_from_place("Cruzeiro, Distrito Federal, Brazil", network_type="drive")

# ==========================================
# 2️⃣ MAPEAMENTO DOS OBSTÁCULOS
# ==========================================
semaforos = set()
lombadas = set()

for node, data in G.nodes(data=True):
    if data.get('highway') == 'traffic_signals':
        semaforos.add(node)
    if data.get('traffic_calming') in ['bump', 'hump']:
        lombadas.add(node)

# ==========================================
# 3️⃣ CÁLCULO DE PESOS (TEMPO EM SEGUNDOS)
# ==========================================
# Dicionário salva-vidas para ruas sem velocidade cadastrada no OSM
velocidade_padrao_kmh = {
    'trunk': 80, 'trunk_link': 60,
    'primary': 60, 'primary_link': 50,
    'secondary': 50, 'secondary_link': 40,
    'tertiary': 40, 'tertiary_link': 30,
    'residential': 30, 'unclassified': 30
}

VELOCIDADE_MAX_MAPA_MS = 80 / 3.6 # 80 km/h convertido para m/s (para a heurística)

for u, v, key, data in G.edges(keys=True, data=True):
    distancia_m = data.get('length', 0)
    tipo_via = data.get('highway', 'residential')
    if isinstance(tipo_via, list): tipo_via = tipo_via[0] # Pega o primeiro se for lista
    
    # 1. Descobrir a velocidade
    vel_kmh = data.get('maxspeed')
    if not vel_kmh or isinstance(vel_kmh, list):
        vel_kmh = velocidade_padrao_kmh.get(tipo_via, 30) # Fallback
    else:
        vel_kmh = float(vel_kmh)
        
    vel_ms = vel_kmh / 3.6 # Converter para Metros por Segundo
    
    # 2. Calcular tempo base da via (Física: Tempo = Distância / Velocidade)
    tempo_segundos = distancia_m / vel_ms
    
    # 3. Adicionar penalidades de obstáculos
    if v in semaforos:
        tempo_segundos += 15.0 # Perde 15 segundos no semáforo
    if v in lombadas:
        tempo_segundos += 10.0 # Perde 10 segundos freando na lombada
        
    # 4. Salvar no grafo
    G[u][v][key]['tempo_segundos'] = tempo_segundos

# ==========================================
# 4️⃣ ALGORITMO A* COM HEURÍSTICA DE TEMPO
# ==========================================
def heuristica_tempo(n1, n2):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    distancia_reta = ((x1-x2)**2 + (y1-y2)**2)**0.5 * 111139
    
    # Heurística precisa ser em TEMPO (O tempo otimista se ele fosse em linha reta voando na vel. máxima)
    return distancia_reta / VELOCIDADE_MAX_MAPA_MS

# ==========================================
# 5️⃣ INTERFACE CLI
# ==========================================
def menu_cli():
    print("\n" + "="*40)
    print("🚗 ROTEADOR INTELIGENTE - CRUZEIRO/DF 🚗")
    print("="*40)
    
    # Coordenadas pré-definidas para facilitar o teste no CLI
    locais = {
        "1": {"nome": "Feira do Cruzeiro", "coord": (-15.7906, -47.9407)},
        "2": {"nome": "Hospital das Forças Armadas (HFA)", "coord": (-15.8005, -47.9409)},
        "3": {"nome": "Terraço Shopping (Octogonal/Fronteira)", "coord": (-15.8000, -47.9333)},
        "4": {"nome": "Sair do Sistema"}
    }
    
    print("\nLocais disponíveis para teste:")
    for k, v in locais.items():
        print(f"[{k}] {v['nome']}")
        
    origem_id = input("\nEscolha o número do local de ORIGEM: ")
    if origem_id == '4': return
    
    destino_id = input("Escolha o número do local de DESTINO: ")
    if destino_id == '4' or origem_id not in locais or destino_id not in locais:
        print("Saindo...")
        return
        
    print("\nCalculando a rota mais rápida...")
    coord_o = locais[origem_id]['coord']
    coord_d = locais[destino_id]['coord']
    
    no_origem = ox.distance.nearest_nodes(G, coord_o[1], coord_o[0])
    no_destino = ox.distance.nearest_nodes(G, coord_d[1], coord_d[0])
    
    try:
        rota = nx.astar_path(G, no_origem, no_destino, weight="tempo_segundos", heuristic=heuristica_tempo)
        
        # Calcular estatísticas finais da rota
        tempo_total = 0
        distancia_total = 0
        for i in range(len(rota)-1):
            u = rota[i]
            v = rota[i+1]
            dados_aresta = G[u][v][0]
            tempo_total += dados_aresta['tempo_segundos']
            distancia_total += dados_aresta['length']
            
        print("\n✅ ROTA ENCONTRADA COM SUCESSO!")
        print(f"Distância: {distancia_total/1000:.2f} km")
        print(f"Tempo Estimado: {tempo_total/60:.1f} minutos")
        print(f"Passou por {len(rota)} cruzamentos.")
        
    except nx.NetworkXNoPath:
        print("❌ Erro: Não foi possível encontrar um caminho entre esses pontos.")

if __name__ == "__main__":
    menu_cli()