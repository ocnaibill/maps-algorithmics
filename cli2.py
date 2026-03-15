import sys
import matplotlib
from scipy import stats
if sys.platform == 'darwin': 
    matplotlib.use('TkAgg')
    matplotlib.rcParams['figure.dpi'] = 100  
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import heapq
import time
import gc
import json
import os

# Popup de input gráfico (multiplataforma)
try:
    import tkinter as tk
    from tkinter import simpledialog
    TK_DISPONIVEL = True
except ImportError:
    TK_DISPONIVEL = False

# ==========================================
# INICIALIZAÇÃO E DOWNLOAD DO MAPA
# ==========================================
print("Carregando o sistema e baixando o mapa do Cruzeiro (DF)...")
if 'traffic_calming' not in ox.settings.useful_tags_node:
    ox.settings.useful_tags_node.append('traffic_calming')
if 'surface' not in ox.settings.useful_tags_way:
    ox.settings.useful_tags_way.append('surface')

G = ox.graph_from_place("Cruzeiro, Distrito Federal, Brazil", network_type="drive")

# ── Substituição de nearest_nodes / nearest_edges sem scikit-learn ──────────
# osmnx exige scikit-learn para grafos não projetados (CRS geográfico).
# Estas funções fazem a busca por distância euclidiana em graus, o que é
# suficiente para a escala de um bairro como o Cruzeiro-DF.

def _nearest_node(G, lon, lat):
    """Retorna o nó mais próximo de (lon, lat) sem depender do scikit-learn."""
    melhor_no   = None
    melhor_dist = float('inf')
    for no, dados in G.nodes(data=True):
        dx = dados['x'] - lon
        dy = dados['y'] - lat
        d  = dx * dx + dy * dy          # distância² em graus (escala local ok)
        if d < melhor_dist:
            melhor_dist = d
            melhor_no   = no
    return melhor_no

def _nearest_edge(G, lon, lat):
    """
    Retorna (u, v, key) da aresta mais próxima de (lon, lat).
    Usa a distância do ponto à projeção sobre cada segmento de aresta.
    """
    melhor_uvk  = None
    melhor_dist = float('inf')

    for u, v, key, dados in G.edges(keys=True, data=True):
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        dx, dy  = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy

        if seg_len2 == 0:
            px, py = x1, y1
        else:
            t = max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / seg_len2))
            px, py = x1 + t * dx, y1 + t * dy

        d = (lon - px)**2 + (lat - py)**2
        if d < melhor_dist:
            melhor_dist = d
            melhor_uvk  = (u, v, key)

    return melhor_uvk

# Tenta usar a versão do osmnx (mais rápida com scikit-learn); cai no fallback.
def nearest_nodes(G, lon, lat):
    try:
        return ox.distance.nearest_nodes(G, lon, lat)
    except ImportError:
        return _nearest_node(G, lon, lat)

def nearest_edges(G, lon, lat):
    try:
        return ox.distance.nearest_edges(G, lon, lat)
    except ImportError:
        return _nearest_edge(G, lon, lat)

# ────────────────────────────────────────────────────────────────────────────

VELOCIDADE_MAX_MAPA_MS = 80 / 3.6
VELOCIDADE_PADRAO_KMH = {
    'trunk': 80, 'primary': 60, 'secondary': 60,
    'tertiary': 40, 'residential': 40, 'unclassified': 40
}

# ==========================================
# 1. MODELO FÍSICO DE PENALIDADES
# ==========================================
TABELA_PENALIDADES = {
    'semaforo30': 30.0,
    'semaforo60': 60.0,
    'lombada40': 15.0,
    'lombada60': 10.0,
    'lombada80': 5.0
}

def multiplicador_via(vel_kmh):
    if vel_kmh <= 40:
        return 1.5
    elif vel_kmh <= 60:
        return 1.2
    else:
        return 1.0

FRENAGEM_MS2    = 3.0
ACELERACAO_MS2  = 2.0
ESPERA_SEMAFORO_S = 30.0
VEL_LOMBADA_KMH = 20.0
ZONA_LOMBADA_M  = 30.0

# ==========================================
# 2. PERSISTÊNCIA DE EDIÇÕES (JSON)
# ==========================================
ARQUIVO_EDICOES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edicoes_mapa.json')

def salvar_edicoes():
    dados = {
        'semaforos60': list(edicoes_usuario['semaforos60']),
        'semaforos30': list(edicoes_usuario['semaforos30']),
        'lombadas40':  list(edicoes_usuario['lombadas40']),
        'lombadas60':  list(edicoes_usuario['lombadas60']),
        'lombadas80':  list(edicoes_usuario['lombadas80']),
        'velocidades': {f"{u},{v},{k}": vel for (u, v, k), vel in edicoes_usuario['velocidades'].items()},
        'removidos':   list(edicoes_usuario['removidos']),
    }
    with open(ARQUIVO_EDICOES, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)

def carregar_edicoes():
    if not os.path.exists(ARQUIVO_EDICOES):
        return {
            'semaforos60': set(), 'semaforos30': set(),
            'lombadas40': set(), 'lombadas60': set(), 'lombadas80': set(),
            'velocidades': {}, 'removidos': set()
        }
    try:
        with open(ARQUIVO_EDICOES, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        edicoes = {
            'semaforos60': set(dados.get('semaforos60', [])),
            'semaforos30': set(dados.get('semaforos30', [])),
            'lombadas40':  set(dados.get('lombadas40', [])),
            'lombadas60':  set(dados.get('lombadas60', [])),
            'lombadas80':  set(dados.get('lombadas80', [])),
            'velocidades': {},
            'removidos':   set(dados.get('removidos', [])),
        }
        for chave_str, vel in dados.get('velocidades', {}).items():
            partes = chave_str.split(',')
            edicoes['velocidades'][(int(partes[0]), int(partes[1]), int(partes[2]))] = vel
        return edicoes
    except Exception as e:
        print(f"⚠️ Erro ao carregar: {e}. Criando novo dicionário limpo.")
        return {
            'semaforos60': set(), 'semaforos30': set(),
            'lombadas40': set(), 'lombadas60': set(), 'lombadas80': set(),
            'velocidades': {}, 'removidos': set()
        }

edicoes_usuario = carregar_edicoes()
tem_edicoes_salvas = os.path.exists(ARQUIVO_EDICOES) and any([
    edicoes_usuario['semaforos60'], edicoes_usuario['semaforos30'],
    edicoes_usuario['lombadas40'], edicoes_usuario['lombadas60'], edicoes_usuario['lombadas80'],
    edicoes_usuario['velocidades'], edicoes_usuario['removidos']
])
if tem_edicoes_salvas:
    qtd = sum(len(v) for v in edicoes_usuario.values())
    print(f"📂 {qtd} edição(ões) carregada(s) do arquivo anterior.")

def resetar_todas_edicoes():
    for k in edicoes_usuario:
        edicoes_usuario[k].clear() if isinstance(edicoes_usuario[k], set) else edicoes_usuario[k].clear()
    if os.path.exists(ARQUIVO_EDICOES):
        os.remove(ARQUIVO_EDICOES)
        print("🗑️ Arquivo de edições removido do disco.")
    atualizar_pesos_do_grafo()
    print("🔄 Mapa resetado para os padrões originais!")

# ==========================================
# 3. MOTOR DE CÁLCULO DE PESOS
# ==========================================
def atualizar_pesos_do_grafo():
    sem30 = edicoes_usuario['semaforos30'].copy()
    sem60 = edicoes_usuario['semaforos60'].copy()
    lom40 = edicoes_usuario['lombadas40'].copy()
    lom60 = edicoes_usuario['lombadas60'].copy()
    lom80 = edicoes_usuario['lombadas80'].copy()

    for node, data in G.nodes(data=True):
        if node in edicoes_usuario['removidos']:
            continue
        if data.get('highway') == 'traffic_signals':
            if node not in sem60:
                sem30.add(node)
        if data.get('traffic_calming') in ['bump', 'hump']:
            if node not in lom60 and node not in lom80:
                lom40.add(node)

    for u, v, key, data in G.edges(keys=True, data=True):
        distancia_m = data.get('length', 0)
        vel_kmh = edicoes_usuario['velocidades'].get((u, v, key), 30)
        vel_ms  = max(0.1, vel_kmh / 3.6)
        tempo_base = distancia_m / vel_ms
        penalidade_total = 0

        if v in sem60:
            penalidade_total += TABELA_PENALIDADES['semaforo60']
        elif v in sem30:
            penalidade_total += TABELA_PENALIDADES['semaforo30']

        if v in lom80:
            penalidade_total += TABELA_PENALIDADES['lombada80']
        elif v in lom60:
            penalidade_total += TABELA_PENALIDADES['lombada60']
        elif v in lom40:
            penalidade_total += TABELA_PENALIDADES['lombada40']

        G[u][v][key]['tempo_segundos'] = tempo_base + penalidade_total

def fechar_janela_seguro(fig):
    plt.close(fig)
    for _ in range(5):
        plt.pause(0.05)
    plt.close('all')
    gc.collect()

def tem_edicoes_ativas():
    return bool(
        edicoes_usuario['semaforos30'] or edicoes_usuario['semaforos60'] or
        edicoes_usuario['lombadas40']  or edicoes_usuario['lombadas60']  or
        edicoes_usuario['lombadas80']  or edicoes_usuario['velocidades'] or
        edicoes_usuario['removidos']
    )

def calcular_rota_sem_edicoes(origem, destino):
    backup = {k: (v.copy() if isinstance(v, set) else dict(v)) for k, v in edicoes_usuario.items()}

    for k in edicoes_usuario:
        if isinstance(edicoes_usuario[k], set):
            edicoes_usuario[k] = set()
        else:
            edicoes_usuario[k] = {}
    atualizar_pesos_do_grafo()

    inicio = time.time()
    caminho_orig, visitados_orig = a_star_animado(origem, destino)
    tempo_orig = time.time() - inicio
    stats_orig = calcular_estatisticas_dict(caminho_orig)

    for k, v in backup.items():
        edicoes_usuario[k] = v
    atualizar_pesos_do_grafo()

    return caminho_orig, visitados_orig, tempo_orig, stats_orig

# ==========================================
# 4. ALGORITMOS DE BUSCA
# ==========================================
def heuristica_tempo(n1, n2):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    distancia_reta = ((x1 - x2)**2 + (y1 - y2)**2)**0.5 * 111139
    return distancia_reta / VELOCIDADE_MAX_MAPA_MS

def a_star_animado(inicio, fim):
    fila = []
    heapq.heappush(fila, (0, inicio))
    g_custo = {inicio: 0}
    pais    = {}
    visitados = []

    while fila:
        _, atual = heapq.heappop(fila)
        visitados.append(atual)
        if atual == fim:
            break
        for vizinho in G.neighbors(atual):
            peso   = G[atual][vizinho][0]['tempo_segundos']
            novo_g = g_custo[atual] + peso
            if vizinho not in g_custo or novo_g < g_custo[vizinho]:
                g_custo[vizinho] = novo_g
                f = novo_g + heuristica_tempo(vizinho, fim)
                pais[vizinho] = atual
                heapq.heappush(fila, (f, vizinho))

    caminho, atual = [], fim
    while atual in pais:
        caminho.append(atual)
        atual = pais[atual]
    caminho.append(inicio)
    caminho.reverse()
    return caminho, visitados

def dijkstra_animado(inicio, fim):
    fila = []
    heapq.heappush(fila, (0, inicio))
    g_custo   = {inicio: 0}
    pais      = {}
    visitados = []

    while fila:
        _, atual = heapq.heappop(fila)
        visitados.append(atual)
        if atual == fim:
            break
        for vizinho in G.neighbors(atual):
            peso   = G[atual][vizinho][0]['tempo_segundos']
            novo_g = g_custo[atual] + peso
            if vizinho not in g_custo or novo_g < g_custo[vizinho]:
                g_custo[vizinho] = novo_g
                pais[vizinho]    = atual
                heapq.heappush(fila, (novo_g, vizinho))

    caminho, atual = [], fim
    while atual in pais:
        caminho.append(atual)
        atual = pais[atual]
    caminho.append(inicio)
    caminho.reverse()
    return caminho, visitados

# ==========================================
# 5. ESTATÍSTICAS
# ==========================================
def calcular_estatisticas_dict(caminho):
    distancia_total = tempo_total = 0
    sem60 = sem30 = lom40 = lom60 = lom80 = 0

    sem30_ativos = edicoes_usuario['semaforos30'].copy()
    sem60_ativos = edicoes_usuario['semaforos60'].copy()
    lom40_ativas = edicoes_usuario['lombadas40'].copy()
    lom60_ativas = edicoes_usuario['lombadas60'].copy()
    lom80_ativas = edicoes_usuario['lombadas80'].copy()

    for n, d in G.nodes(data=True):
        if d.get('highway') == 'traffic_signals':
            sem30_ativos.add(n)
        if d.get('traffic_calming') in ['bump', 'hump']:
            lom40_ativas.add(n)

    for i in range(len(caminho) - 1):
        u, v = caminho[i], caminho[i + 1]
        dados = G[u][v][0]
        distancia_total += dados['length']
        tempo_total     += dados['tempo_segundos']
        if v in sem60_ativos: sem60 += 1
        if v in sem30_ativos: sem30 += 1
        if v in lom40_ativas: lom40 += 1
        if v in lom60_ativas: lom60 += 1
        if v in lom80_ativas: lom80 += 1

    vel_media = (distancia_total / tempo_total) * 3.6 if tempo_total > 0 else 0
    return {
        'distancia_km': distancia_total / 1000,
        'tempo_min':    tempo_total / 60,
        'vel_media':    vel_media,
        'semaforos60':  sem60,
        'semaforos30':  sem30,
        'lombadas40':   lom40,
        'lombadas60':   lom60,
        'lombadas80':   lom80,
    }

def formatar_estatisticas(stats, visitados, tempo_execucao, titulo="ESTATÍSTICAS DA ROTA"):
    total_sem = stats['semaforos30'] + stats['semaforos60']
    total_lom = stats['lombadas40'] + stats['lombadas60'] + stats['lombadas80']
    return (
        f"{titulo}\n\n"
        f"⏱️  Tempo de Busca:\n   {tempo_execucao:.4f} seg\n\n"
        f"🔍 Nós Explorados:\n   {len(visitados)} cruzamentos\n\n"
        f"🛣️  Distância Total:\n   {stats['distancia_km']:.2f} km\n\n"
        f"⏳ Tempo Estimado:\n   {stats['tempo_min']:.1f} minutos\n\n"
        f"🏎️  Velocidade Média:\n   {stats['vel_media']:.1f} km/h\n\n"
        f"🚦 Semáforos: {total_sem}\n   (30s: {stats['semaforos30']}, 60s: {stats['semaforos60']})\n\n"
        f"🏔️  Lombadas: {total_lom}"
    )

def formatar_comparacao(stats_mod, stats_orig):
    def diff(val_mod, val_orig, fmt=".1f", inverso=False):
        delta = val_mod - val_orig
        if abs(delta) < 0.01:
            return "="
        sinal = "+" if delta > 0 else ""
        cor = ("⬇️" if delta < 0 else "⬆️") if inverso else ("⬆️" if delta > 0 else "⬇️")
        return f"{cor}{sinal}{delta:{fmt}}"

    sem_mod  = stats_mod['semaforos30']  + stats_mod['semaforos60']
    sem_orig = stats_orig['semaforos30'] + stats_orig['semaforos60']
    lom_mod  = stats_mod['lombadas40']  + stats_mod['lombadas60']  + stats_mod['lombadas80']
    lom_orig = stats_orig['lombadas40'] + stats_orig['lombadas60'] + stats_orig['lombadas80']

    return (
        f"COMPARAÇÃO\nEditado vs Original\n\n"
        f"🛣️  Distância:\n"
        f"   {stats_mod['distancia_km']:.2f} vs {stats_orig['distancia_km']:.2f} km\n"
        f"   {diff(stats_mod['distancia_km'], stats_orig['distancia_km'], '.2f', inverso=True)}\n\n"
        f"⏳ Tempo:\n"
        f"   {stats_mod['tempo_min']:.1f} vs {stats_orig['tempo_min']:.1f} min\n"
        f"   {diff(stats_mod['tempo_min'], stats_orig['tempo_min'], inverso=True)}\n\n"
        f"🏎️  Vel. Média:\n"
        f"   {stats_mod['vel_media']:.1f} vs {stats_orig['vel_media']:.1f}\n"
        f"   {diff(stats_mod['vel_media'], stats_orig['vel_media'])}\n\n"
        f"🚦 Semáforos:\n   {sem_mod} vs {sem_orig}\n\n"
        f"🏔️  Lombadas:\n   {lom_mod} vs {lom_orig}"
    )

# ==========================================
# 6. VISUALIZAÇÃO PADRÃO (opções 1 e 2)
# ==========================================
def exibir_mapa_com_painel(caminho, visitados, tempo_execucao, animar=False,
                           caminho_original=None, visitados_orig=None,
                           tempo_orig=None, stats_orig=None):
    comparando     = caminho_original is not None
    rotas_identicas = comparando and caminho == caminho_original

    if comparando:
        fig = plt.figure(figsize=(18, 8))
        fig.canvas.manager.set_window_title('Resultado da Rota — Comparação com Mapa Original')
        gs      = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1])
        ax_mapa = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
        ax_comp = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(14, 8))
        fig.canvas.manager.set_window_title('Resultado da Rota')
        gs      = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_mapa = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])

    ax_info.set_facecolor('#111111')
    ax_info.axis('off')
    stats_mod   = calcular_estatisticas_dict(caminho)
    texto_stats = formatar_estatisticas(
        stats_mod, visitados, tempo_execucao,
        titulo="ROTA EDITADA ✏️" if comparando else "ESTATÍSTICAS DA ROTA"
    )
    ax_info.text(0.1, 0.5, texto_stats, color='lime', fontsize=11,
                 va='center', ha='left', family='monospace')

    if comparando and stats_orig is not None:
        ax_comp.set_facecolor('#111111')
        ax_comp.axis('off')
        texto_comp = formatar_comparacao(stats_mod, stats_orig)
        if rotas_identicas:
            texto_comp += "\n\n📌 MESMA ROTA\n   As edições não mudaram\n   o caminho."
        ax_comp.text(0.1, 0.5, texto_comp, color='#88CCFF', fontsize=11,
                     va='center', ha='left', family='monospace')

    ox.plot_graph(G, ax=ax_mapa, show=False, close=False,
                  node_size=0, edge_color='#555555', edge_alpha=0.5, bgcolor='#111111')
    fig.patch.set_facecolor('#111111')

    if animar:
        coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in visitados]
        scat   = ax_mapa.scatter([], [], c='cyan', s=15, zorder=3)
        passo  = max(1, len(coords) // 50)
        for i in range(1, len(coords) + passo, passo):
            scat.set_offsets(coords[:min(i, len(coords))])
            ax_mapa.set_title(f"A* Explorando... ({min(i, len(coords))} nós)",
                              color='cyan', fontsize=14)
            plt.pause(0.01)

    if comparando:
        titulo_mapa = "🟢 Rota Editada  vs  🟠 Rota Original"
        if rotas_identicas:
            titulo_mapa = "Mesmo caminho — só os tempos mudaram"
        ax_mapa.set_title(titulo_mapa, color='white', fontsize=13, fontweight='bold')
        legenda = [
            Line2D([0], [0], color='#00FF88', linewidth=3, label='Rota Editada'),
            Line2D([0], [0], color='#FF6600', linewidth=8, alpha=0.5, label='Rota Original'),
        ]
        ax_mapa.legend(handles=legenda, loc='best',
                       facecolor='#222222', labelcolor='white', fontsize=10)
        ox.plot_graph_routes(G, [caminho_original, caminho],
                             route_colors=['#FF6600', '#00FF88'],
                             route_linewidths=[8, 3], ax=ax_mapa,
                             node_size=0, route_alpha=0.7)
        orig_x, orig_y = G.nodes[caminho[0]]['x'],  G.nodes[caminho[0]]['y']
        dest_x, dest_y = G.nodes[caminho[-1]]['x'], G.nodes[caminho[-1]]['y']
        ax_mapa.scatter([orig_x], [orig_y], c='white', s=150, marker='o',
                        edgecolors='black', linewidths=2, zorder=8)
        ax_mapa.scatter([dest_x], [dest_y], c='white', s=150, marker='*',
                        edgecolors='black', linewidths=2, zorder=8)
    else:
        ox.plot_graph_route(G, caminho, ax=ax_mapa, route_color='red',
                            route_linewidth=4, node_size=0)

    plt.show()


# ==========================================
# 7. COMPARAÇÃO ANIMADA: A* vs DIJKSTRA  ★ NOVO ★
# ==========================================

def comparar_astar_dijkstra_animado(origem, destino):
    """
    Três fases de animação lado a lado:

      FASE 1 — EXPLORAÇÃO
        Nós visitados crescem frame a frame em ambos os mapas
        (pontos amarelos no A*, azuis no Dijkstra).

      FASE 2 — REVELAÇÃO DA ROTA (nó a nó)
        As arestas da rota são desenhadas segmento a segmento,
        simultaneamente nos dois mapas, com um ponto "cursor"
        que avança pela rota. Velocidade proporcional ao número
        de nós de cada rota para que terminem juntos.

      FASE 3 — ESTADO FINAL
        Rotas completas destacadas, exploração esmaecida,
        marcadores de origem/destino redesenhados por cima,
        painel de stats atualizado com resultado final.
    """

    # ── 7a. Executa os dois algoritmos e mede tempo ──────────────────────────
    print("  ⚡ Executando A*...")
    t0 = time.time()
    caminho_astar, visitados_astar = a_star_animado(origem, destino)
    tempo_astar = time.time() - t0

    print("  🔵 Executando Dijkstra...")
    t0 = time.time()
    caminho_dijkstra, visitados_dijkstra = dijkstra_animado(origem, destino)
    tempo_dijkstra = time.time() - t0

    stats_astar     = calcular_estatisticas_dict(caminho_astar)
    stats_dijkstra  = calcular_estatisticas_dict(caminho_dijkstra)
    rotas_identicas = caminho_astar == caminho_dijkstra

    # ── 7b. Monta a figura ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor('#0a0a0a')
    fig.canvas.manager.set_window_title('Comparação A* vs Dijkstra — Animação')

    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[5, 5, 3],
        left=0.02, right=0.98,
        top=0.93, bottom=0.04,
        wspace=0.06
    )
    ax_astar    = fig.add_subplot(gs[0])
    ax_dijkstra = fig.add_subplot(gs[1])
    ax_stats    = fig.add_subplot(gs[2])

    # ── 7c. Grafo base ───────────────────────────────────────────────────────
    for ax, titulo, cor in [
        (ax_astar,    "⚡ A*  (Heurístico)",     '#FFD700'),
        (ax_dijkstra, "🔵 Dijkstra (Exaustivo)", '#00BFFF'),
    ]:
        ox.plot_graph(G, ax=ax, show=False, close=False,
                      node_size=0, edge_color='#3a3a3a',
                      edge_alpha=0.6, bgcolor='#111111')
        ax.set_title(titulo, color=cor, fontsize=13, fontweight='bold', pad=6)

    ax_stats.set_facecolor('#111111')
    ax_stats.axis('off')
    fig.text(0.845, 0.965, "COMPARAÇÃO A* vs DIJKSTRA",
             color='white', fontsize=12, fontweight='bold',
             ha='center', va='top', family='monospace')

    # ── 7d. Paleta de cores ───────────────────────────────────────────────────
    CORES = {
        'astar_explore':    '#FFD700',
        'dijkstra_explore': '#00BFFF',
        'astar_rota':       '#FF4500',
        'dijkstra_rota':    '#00FF7F',
        'origem':           '#FFFFFF',
        'destino':          '#FF00FF',
        'cursor_a':         '#FFAA00',
        'cursor_d':         '#00DDFF',
    }

    # ── 7e. Objetos de animação ───────────────────────────────────────────────

    # Fase 1 — nuvens de exploração
    scat_a = ax_astar.scatter([], [], c=CORES['astar_explore'],
                               s=10, zorder=4, alpha=0.55)
    scat_d = ax_dijkstra.scatter([], [], c=CORES['dijkstra_explore'],
                                  s=10, zorder=4, alpha=0.55)

    # Fase 2 — segmentos da rota revelados progressivamente (Line2D acumulativo)
    linha_rota_a, = ax_astar.plot([], [], color=CORES['astar_rota'],
                                   linewidth=4, zorder=6, solid_capstyle='round')
    linha_rota_d, = ax_dijkstra.plot([], [], color=CORES['dijkstra_rota'],
                                      linewidth=4, zorder=6, solid_capstyle='round')

    # Fase 2 — cursor (ponto que avança na frente da linha)
    cursor_a = ax_astar.scatter([], [], c=CORES['cursor_a'],
                                 s=80, zorder=8, edgecolors='white', linewidths=1.2)
    cursor_d = ax_dijkstra.scatter([], [], c=CORES['cursor_d'],
                                    s=80, zorder=8, edgecolors='white', linewidths=1.2)

    # Marcadores de origem / destino (criados agora, redesenhados na fase 3)
    def _desenhar_pins(alpha=1.0):
        for ax in (ax_astar, ax_dijkstra):
            ax.scatter([G.nodes[origem]['x']],  [G.nodes[origem]['y']],
                       c=CORES['origem'],  s=180, marker='o',
                       edgecolors='black', linewidths=2, zorder=10, alpha=alpha)
            ax.scatter([G.nodes[destino]['x']], [G.nodes[destino]['y']],
                       c=CORES['destino'], s=220, marker='*',
                       edgecolors='black', linewidths=2, zorder=10, alpha=alpha)

    _desenhar_pins()

    # Texto de progresso em cada mapa
    txt_a = ax_astar.text(
        0.02, 0.04, "", transform=ax_astar.transAxes,
        color=CORES['astar_explore'], fontsize=10, family='monospace', zorder=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                  edgecolor=CORES['astar_explore'], alpha=0.85)
    )
    txt_d = ax_dijkstra.text(
        0.02, 0.04, "", transform=ax_dijkstra.transAxes,
        color=CORES['dijkstra_explore'], fontsize=10, family='monospace', zorder=11,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                  edgecolor=CORES['dijkstra_explore'], alpha=0.85)
    )

    txt_stats = ax_stats.text(
        0.05, 0.97, "", transform=ax_stats.transAxes,
        color='white', fontsize=10, va='top', ha='left',
        family='monospace', linespacing=1.5
    )

    # ── 7f. Painel de estatísticas ────────────────────────────────────────────
    def _atualizar_painel(fase='explorando', pct_a=0, pct_d=0, nos_rota_a=0, nos_rota_d=0):
        vencedor_nos   = "⚡ A*"      if len(visitados_astar) < len(visitados_dijkstra) else "🔵 Dijkstra"
        vencedor_tempo = "⚡ A*"      if tempo_astar < tempo_dijkstra                  else "🔵 Dijkstra"
        eficiencia     = len(visitados_astar) / max(len(visitados_dijkstra), 1) * 100

        if fase == 'explorando':
            status = (f"🔍 FASE 1 — EXPLORAÇÃO\n"
                      f"   A*:       {pct_a:3.0f}%\n"
                      f"   Dijkstra: {pct_d:3.0f}%\n")
        elif fase == 'rota':
            status = (f"🛤️  FASE 2 — TRAÇANDO ROTA\n"
                      f"   A*:       {nos_rota_a}/{len(caminho_astar)} nós\n"
                      f"   Dijkstra: {nos_rota_d}/{len(caminho_dijkstra)} nós\n")
        else:
            status = "✅ FASE 3 — CONCLUÍDO\n"

        linhas = [
            status,
            "─" * 28,
            "⏱️  TEMPO DE BUSCA",
            f"   A*:       {tempo_astar:.4f}s",
            f"   Dijkstra: {tempo_dijkstra:.4f}s",
            f"   Mais rápido: {vencedor_tempo}",
            "",
            "🔍 NÓS EXPLORADOS",
            f"   A*:       {len(visitados_astar):,}",
            f"   Dijkstra: {len(visitados_dijkstra):,}",
            f"   Eficiência A*: {eficiencia:.1f}%",
            f"   Menos nós: {vencedor_nos}",
            "",
            "─" * 28,
            "🛣️  DISTÂNCIA",
            f"   A*:       {stats_astar['distancia_km']:.3f} km",
            f"   Dijkstra: {stats_dijkstra['distancia_km']:.3f} km",
            "",
            "⏳ TEMPO ROTA",
            f"   A*:       {stats_astar['tempo_min']:.2f} min",
            f"   Dijkstra: {stats_dijkstra['tempo_min']:.2f} min",
            "",
            "🏎️  VEL. MÉDIA",
            f"   A*:       {stats_astar['vel_media']:.1f} km/h",
            f"   Dijkstra: {stats_dijkstra['vel_media']:.1f} km/h",
            "",
            "─" * 28,
        ]

        if fase == 'concluido':
            if rotas_identicas:
                linhas += ["📌 ROTAS IDÊNTICAS",
                           "   Ambos encontraram",
                           "   o mesmo caminho."]
            else:
                dd = stats_astar['distancia_km'] - stats_dijkstra['distancia_km']
                dt = stats_astar['tempo_min']    - stats_dijkstra['tempo_min']
                linhas += [
                    "📊 DIFERENÇA A* vs Dijk",
                    f"   Dist:  {'+' if dd>=0 else ''}{dd:.3f} km",
                    f"   Tempo: {'+' if dt>=0 else ''}{dt:.2f} min",
                ]

        txt_stats.set_text("\n".join(linhas))
        fig.canvas.draw_idle()

    # ── FASE 1: Exploração ───────────────────────────────────────────────────
    coords_a = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in visitados_astar]
    coords_d = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in visitados_dijkstra]

    FRAMES_EXP = 80
    passo_a = max(1, len(coords_a) // FRAMES_EXP)
    passo_d = max(1, len(coords_d) // FRAMES_EXP)

    print("  🎬 Fase 1 — Animando exploração...")
    for frame in range(FRAMES_EXP + 1):
        idx_a = min(frame * passo_a, len(coords_a))
        idx_d = min(frame * passo_d, len(coords_d))

        scat_a.set_offsets(coords_a[:idx_a] if idx_a else [(0, 0)])
        scat_d.set_offsets(coords_d[:idx_d] if idx_d else [(0, 0)])

        pct_a = idx_a / len(coords_a) * 100
        pct_d = idx_d / len(coords_d) * 100

        txt_a.set_text(f"Explorados: {idx_a:,} / {len(coords_a):,}")
        txt_d.set_text(f"Explorados: {idx_d:,} / {len(coords_d):,}")

        _atualizar_painel('explorando', pct_a, pct_d)
        plt.pause(0.04)

    # ── FASE 2: Revelação da rota nó a nó ────────────────────────────────────
    # Esmaece a nuvem de exploração para a rota ganhar destaque
    scat_a.set_alpha(0.18)
    scat_d.set_alpha(0.18)

    # Pré-calcula listas de (x, y) para cada rota
    rota_xy_a = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in caminho_astar]
    rota_xy_d = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in caminho_dijkstra]

    # Número de frames da fase 2 baseado na rota mais longa
    FRAMES_ROTA = max(len(rota_xy_a), len(rota_xy_d), 40)

    print("  🎬 Fase 2 — Revelando rotas nó a nó...")
    for frame in range(FRAMES_ROTA + 1):
        # Índice proporcional para cada rota (termina juntas)
        idx_ra = min(int(frame * len(rota_xy_a) / FRAMES_ROTA), len(rota_xy_a))
        idx_rd = min(int(frame * len(rota_xy_d) / FRAMES_ROTA), len(rota_xy_d))

        # Linha acumulada até o nó atual
        if idx_ra >= 2:
            xs_a, ys_a = zip(*rota_xy_a[:idx_ra])
            linha_rota_a.set_data(xs_a, ys_a)
        if idx_rd >= 2:
            xs_d, ys_d = zip(*rota_xy_d[:idx_rd])
            linha_rota_d.set_data(xs_d, ys_d)

        # Cursor no nó mais recente
        if idx_ra >= 1:
            cursor_a.set_offsets([rota_xy_a[idx_ra - 1]])
        if idx_rd >= 1:
            cursor_d.set_offsets([rota_xy_d[idx_rd - 1]])

        txt_a.set_text(f"Rota: {idx_ra}/{len(rota_xy_a)} nós")
        txt_d.set_text(f"Rota: {idx_rd}/{len(rota_xy_d)} nós")

        _atualizar_painel('rota', nos_rota_a=idx_ra, nos_rota_d=idx_rd)
        plt.pause(0.05)

    # ── FASE 3: Estado final ──────────────────────────────────────────────────
    print("  🗺️  Fase 3 — Estado final...")

    # Remove o cursor (já temos a linha completa)
    cursor_a.set_offsets([(0, 0)])
    cursor_a.set_alpha(0)
    cursor_d.set_offsets([(0, 0)])
    cursor_d.set_alpha(0)

    # Quando as rotas são diferentes, adiciona a rota do outro algoritmo
    # como referência semitransparente em cada mapa
    if not rotas_identicas:
        ax_astar.plot(
            [G.nodes[n]['x'] for n in caminho_dijkstra],
            [G.nodes[n]['y'] for n in caminho_dijkstra],
            color=CORES['dijkstra_rota'], linewidth=2.5,
            alpha=0.35, zorder=5, solid_capstyle='round'
        )
        ax_dijkstra.plot(
            [G.nodes[n]['x'] for n in caminho_astar],
            [G.nodes[n]['y'] for n in caminho_astar],
            color=CORES['astar_rota'], linewidth=2.5,
            alpha=0.35, zorder=5, solid_capstyle='round'
        )

        legend_a = [
            Line2D([0], [0], color=CORES['astar_rota'],    linewidth=4, label='A* (principal)'),
            Line2D([0], [0], color=CORES['dijkstra_rota'], linewidth=2,
                   alpha=0.45, label='Dijkstra (ref.)'),
        ]
        legend_d = [
            Line2D([0], [0], color=CORES['dijkstra_rota'], linewidth=4, label='Dijkstra (principal)'),
            Line2D([0], [0], color=CORES['astar_rota'],    linewidth=2,
                   alpha=0.45, label='A* (ref.)'),
        ]
        ax_astar.legend(handles=legend_a, loc='upper left',
                        facecolor='#1e1e1e', labelcolor='white', fontsize=9)
        ax_dijkstra.legend(handles=legend_d, loc='upper left',
                           facecolor='#1e1e1e', labelcolor='white', fontsize=9)

    # Títulos finais
    ax_astar.set_title(
        f"⚡ A* — {len(caminho_astar)} nós | {stats_astar['distancia_km']:.2f} km ✅",
        color=CORES['astar_rota'], fontsize=12, fontweight='bold'
    )
    ax_dijkstra.set_title(
        f"🔵 Dijkstra — {len(caminho_dijkstra)} nós | {stats_dijkstra['distancia_km']:.2f} km ✅",
        color=CORES['dijkstra_rota'], fontsize=12, fontweight='bold'
    )

    # Texto final em cada mapa
    txt_a.set_text(f"✅ {len(caminho_astar)} nós na rota")
    txt_d.set_text(f"✅ {len(caminho_dijkstra)} nós na rota")
    txt_a.set_color(CORES['astar_rota'])
    txt_d.set_color(CORES['dijkstra_rota'])

    # Redesenha pins por cima de tudo
    _desenhar_pins()

    _atualizar_painel('concluido')

    # Rodapé
    nota = ("📌 Rotas idênticas — A* explorou menos nós para chegar ao mesmo resultado"
            if rotas_identicas
            else "📌 Rotas diferentes — cada algoritmo escolheu um caminho distinto")
    fig.text(0.5, 0.005, nota, ha='center', color='#aaaaaa',
             fontsize=10, family='monospace')

    print("\n" + "=" * 60)
    print("  RESULTADO FINAL — A* vs DIJKSTRA")
    print("=" * 60)
    print(f"  Rotas idênticas?      {'SIM ✔️' if rotas_identicas else 'NÃO ✖️'}")
    print(f"  A*       — busca: {tempo_astar:.4f}s | explorados: {len(visitados_astar):,} | rota: {len(caminho_astar)} nós")
    print(f"  Dijkstra — busca: {tempo_dijkstra:.4f}s | explorados: {len(visitados_dijkstra):,} | rota: {len(caminho_dijkstra)} nós")
    print(f"  Eficiência A* (nós explorados): {len(visitados_astar)/max(len(visitados_dijkstra),1)*100:.1f}% do Dijkstra")
    print(f"  A*       — {stats_astar['distancia_km']:.3f} km | {stats_astar['tempo_min']:.2f} min")
    print(f"  Dijkstra — {stats_dijkstra['distancia_km']:.3f} km | {stats_dijkstra['tempo_min']:.2f} min")
    print("=" * 60)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()

    return (caminho_astar, visitados_astar, tempo_astar,
            caminho_dijkstra, visitados_dijkstra, tempo_dijkstra)


# ==========================================
# 8. MODO SELECIONAR ROTA (ATUALIZADO)
# ==========================================
def modo_selecionar_rota():
    print("\n🗺️  O mapa vai abrir. CLIQUE em 2 pontos: primeiro na ORIGEM, depois no DESTINO.")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title('Selecione Origem e Destino')
    ox.plot_graph(G, ax=ax, show=False, close=False,
                  node_size=5, edge_color='#555555', bgcolor='#111111')
    ax.set_title("CLIQUE NA ORIGEM E DEPOIS NO DESTINO", color='white')

    pontos = plt.ginput(2, timeout=0)
    fechar_janela_seguro(fig)

    if len(pontos) < 2:
        print("Você não clicou 2 vezes. Abortando...")
        return

    lon_o, lat_o = pontos[0]
    lon_d, lat_d = pontos[1]
    no_o = nearest_nodes(G, lon_o, lat_o)
    no_d = nearest_nodes(G, lon_d, lat_d)

    print("\nComo deseja visualizar a rota?")
    print("[1] Mapa Estático")
    print("[2] Animação do A*")
    print("[3] Comparar Rota com A* e Dijkstra")
    visualizacao = input("Escolha: ").strip()

    # ── Opção 3: Comparação A* vs Dijkstra ──────────────────────────────────
    if visualizacao == '3':
        print("\n🔬 Iniciando comparação A* vs Dijkstra...")
        comparar_astar_dijkstra_animado(no_o, no_d)
        return

    # ── Opções 1 e 2: fluxo original ────────────────────────────────────────
    print("\nCalculando rota, por favor aguarde...")
    inicio_tempo = time.time()
    caminho, visitados = a_star_animado(no_o, no_d)
    tempo_execucao = time.time() - inicio_tempo

    if tem_edicoes_ativas():
        print("Calculando rota original (sem edições) para comparação...")
        caminho_orig, visitados_orig, tempo_orig, stats_orig = calcular_rota_sem_edicoes(no_o, no_d)

        stats_mod = calcular_estatisticas_dict(caminho)
        nos_editado  = set(caminho)
        nos_original = set(caminho_orig)
        diff_dist  = stats_mod['distancia_km'] - stats_orig['distancia_km']
        diff_tempo = stats_mod['tempo_min']    - stats_orig['tempo_min']
        total_sem  = len(edicoes_usuario['semaforos30']) + len(edicoes_usuario['semaforos60'])
        total_lom  = (len(edicoes_usuario['lombadas40']) + len(edicoes_usuario['lombadas60'])
                      + len(edicoes_usuario['lombadas80']))

        print(f"\n{'=' * 60}")
        print(f"🔍 DEBUG — COMPARAÇÃO DE ROTAS")
        print(f"{'=' * 60}")
        print(f"  Rotas idênticas? {'SIM ✔️' if caminho == caminho_orig else 'NÃO ✖️'}")
        print(f"  Nós rota editada:  {len(caminho)}")
        print(f"  Nós rota original: {len(caminho_orig)}")
        print(f"  Nós em comum:      {len(nos_editado & nos_original)}")
        print(f"{'─' * 60}")
        print(f"  EDITADA  → Dist: {stats_mod['distancia_km']:.3f} km | "
              f"Tempo: {stats_mod['tempo_min']:.2f} min | Vel: {stats_mod['vel_media']:.1f} km/h")
        print(f"  ORIGINAL → Dist: {stats_orig['distancia_km']:.3f} km | "
              f"Tempo: {stats_orig['tempo_min']:.2f} min | Vel: {stats_orig['vel_media']:.1f} km/h")
        print(f"  DIFERENÇA→ Dist: {'+' if diff_dist >= 0 else ''}{diff_dist:.3f} km | "
              f"Tempo: {'+' if diff_tempo >= 0 else ''}{diff_tempo:.2f} min")
        print(f"  Edições ativas: {total_sem} semáforos, {total_lom} lombadas, "
              f"{len(edicoes_usuario['velocidades'])} velocidades")
        print(f"{'=' * 60}\n")

        exibir_mapa_com_painel(caminho, visitados, tempo_execucao,
                               animar=(visualizacao == '2'),
                               caminho_original=caminho_orig,
                               visitados_orig=visitados_orig,
                               tempo_orig=tempo_orig,
                               stats_orig=stats_orig)
    else:
        exibir_mapa_com_painel(caminho, visitados, tempo_execucao,
                               animar=(visualizacao == '2'))


# ==========================================
# 9. MODO DE EDIÇÃO INTERATIVO (inalterado)
# ==========================================
def pedir_velocidade_popup(nome_rua, vel_atual, origem_vel):
    prompt = f"Via: {nome_rua}\nVelocidade atual: {vel_atual:.0f} km/h ({origem_vel})"
    if TK_DISPONIVEL:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            nova_vel = simpledialog.askfloat(
                "Alterar Velocidade", f"{prompt}\n\nNova velocidade (km/h):",
                parent=root, minvalue=1, maxvalue=200, initialvalue=vel_atual
            )
            root.destroy()
            return nova_vel
        except Exception:
            pass
    try:
        nova_vel = input(f"\n🏎️  {prompt}\n   Nova velocidade (km/h): ")
        return float(nova_vel)
    except (ValueError, EOFError):
        return None

def modo_edicao():
    print("\n🛠️ MODO DE EDIÇÃO INTERATIVO")
    print("O mapa vai abrir. Use o TECLADO para selecionar a ferramenta e CLIQUE no mapa para aplicar.")
    print("─" * 50)
    print("  [1] 🚦 Semáforo 30s  [2] 🚦 Semáforo 60s")
    print("  [3] 🏔️  Lombada 40   [4] 🏔️  Lombada 60   [5] 🏔️  Lombada 80")
    print("  [6] 🏎️  Velocidade   [7] 🧹 Limpar edição")
    print("  [Q] Sair do editor")
    print("─" * 50)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.manager.set_window_title('Editor Interativo do Mapa')
    fig.patch.set_facecolor('#111111')
    ox.plot_graph(G, ax=ax, show=False, close=False,
                  node_size=8, edge_color='#444444', bgcolor='#111111')

    estado = {
        'modo': None, 'aberto': True,
        'scat_sem': None, 'scat_lom': None, 'scat_vel': None,
        'texto_status': None, 'marcador_clique': None, 'linha_snap': None,
    }

    nomes_modo = {
        'semaforo30': '🚦 SEMÁFORO 30s',
        'semaforo60': '🚦 SEMÁFORO 60s',
        'lombada40':  '🏔️ PARDAL 40km/h',
        'lombada60':  '🏔️ PARDAL 60km/h',
        'lombada80':  '🏔️ PARDAL 80km/h',
        'velocidade': '🏎️ VELOCIDADE',
        'limpar':     '🧹 LIMPAR EDIÇÃO',
    }

    def atualizar_titulo():
        if estado['modo']:
            titulo = f"EDITOR | Ferramenta: {nomes_modo.get(estado['modo'], estado['modo'])}"
            cor = 'cyan'
        else:
            titulo = "EDITOR | Pressione 1–7 para escolher ferramenta"
            cor = 'white'
        ax.set_title(titulo, color=cor, fontsize=13, fontweight='bold')
        fig.canvas.draw_idle()

    def atualizar_marcadores():
        for chave in ('scat_sem', 'scat_lom', 'scat_vel'):
            if estado[chave]:
                estado[chave].remove()
                estado[chave] = None

        todos_sem = edicoes_usuario['semaforos30'] | edicoes_usuario['semaforos60']
        if todos_sem:
            sx = [G.nodes[n]['x'] for n in todos_sem]
            sy = [G.nodes[n]['y'] for n in todos_sem]
            estado['scat_sem'] = ax.scatter(sx, sy, c='red', s=100, marker='s',
                                             label='🚦 Semáforos', zorder=5,
                                             edgecolors='white', linewidths=0.5)

        todas_lom = edicoes_usuario['lombadas40'] | edicoes_usuario['lombadas60'] | edicoes_usuario['lombadas80']
        if todas_lom:
            lx = [G.nodes[n]['x'] for n in todas_lom]
            ly = [G.nodes[n]['y'] for n in todas_lom]
            estado['scat_lom'] = ax.scatter(lx, ly, c='orange', s=100, marker='^',
                                             label='🏔️ Lombadas', zorder=5,
                                             edgecolors='white', linewidths=0.5)

        nos_vel = {u for (u, v, k) in edicoes_usuario['velocidades']} | \
                  {v for (u, v, k) in edicoes_usuario['velocidades']}
        if nos_vel:
            vx = [G.nodes[n]['x'] for n in nos_vel]
            vy = [G.nodes[n]['y'] for n in nos_vel]
            estado['scat_vel'] = ax.scatter(vx, vy, c='lime', s=60, marker='d',
                                             label='🏎️ Vel. editada', zorder=5,
                                             edgecolors='white', linewidths=0.5)

        handles = [h for h in (estado['scat_sem'], estado['scat_lom'], estado['scat_vel']) if h]
        if handles:
            ax.legend(handles=handles, loc='upper right',
                      facecolor='#222222', labelcolor='white', fontsize=10)
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()
        fig.canvas.draw_idle()

    def mostrar_feedback(msg, cor='lime'):
        if estado['texto_status']:
            estado['texto_status'].remove()
        estado['texto_status'] = ax.text(
            0.5, 0.02, msg, transform=ax.transAxes, color=cor,
            fontsize=11, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#222222',
                      edgecolor=cor, alpha=0.9), zorder=10
        )
        fig.canvas.draw_idle()

    def mostrar_snap(lon_clique, lat_clique, lon_real, lat_real):
        if estado['marcador_clique']:
            estado['marcador_clique'].remove()
        if estado['linha_snap']:
            for l in estado['linha_snap']:
                l.remove()
        estado['marcador_clique'] = ax.scatter(
            [lon_real], [lat_real], c='white', s=200, marker='o',
            edgecolors='yellow', linewidths=2, zorder=8, alpha=0.8
        )
        dist = ((lon_clique - lon_real)**2 + (lat_clique - lat_real)**2)**0.5
        if dist > 1e-6:
            estado['linha_snap'] = ax.plot(
                [lon_clique, lon_real], [lat_clique, lat_real],
                '--', color='yellow', linewidth=1.5, alpha=0.7, zorder=7
            )
        else:
            estado['linha_snap'] = None
        fig.canvas.draw_idle()

    def obter_situacao_local(no_prox, u, v, key):
        dados     = G[u][v][key]
        nome_rua  = dados.get('name', 'Sem nome')
        distancia = dados.get('length', 0)
        tipo_via  = dados.get('highway', 'residential')
        if isinstance(tipo_via, list):
            tipo_via = tipo_via[0]

        if (u, v, key) in edicoes_usuario['velocidades']:
            vel_kmh    = edicoes_usuario['velocidades'][(u, v, key)]
            origem_vel = 'editada pelo usuário'
        else:
            vel_osm = dados.get('maxspeed')
            if vel_osm and not isinstance(vel_osm, list):
                vel_kmh    = float(vel_osm)
                origem_vel = 'do OpenStreetMap'
            else:
                vel_kmh    = VELOCIDADE_PADRAO_KMH.get(tipo_via, 30)
                origem_vel = f'padrão p/ via "{tipo_via}"'

        tem_semaforo, origem_semaforo = False, ''
        if no_prox in edicoes_usuario['semaforos30']:
            tem_semaforo, origem_semaforo = True, '30s (adicionado)'
        elif no_prox in edicoes_usuario['semaforos60']:
            tem_semaforo, origem_semaforo = True, '60s (adicionado)'
        elif G.nodes[no_prox].get('highway') == 'traffic_signals' \
                and no_prox not in edicoes_usuario['removidos']:
            tem_semaforo, origem_semaforo = True, 'original do mapa'

        tem_lombada, origem_lombada = False, ''
        for cat in ['lombadas40', 'lombadas60', 'lombadas80']:
            if no_prox in edicoes_usuario[cat]:
                tem_lombada  = True
                origem_lombada = f'{cat[-2:]}km/h (adicionada)'
                break

        return {
            'nome_rua': nome_rua, 'distancia_m': distancia, 'tipo_via': tipo_via,
            'vel_kmh': vel_kmh, 'origem_vel': origem_vel,
            'tem_semaforo': tem_semaforo, 'origem_semaforo': origem_semaforo,
            'tem_lombada': tem_lombada, 'origem_lombada': origem_lombada,
            'tempo_seg': dados.get('tempo_segundos', 0),
        }

    def imprimir_situacao(info):
        print(f"\n{'─' * 55}")
        print(f"📍 {info['nome_rua']}")
        print(f"   Tipo de via:          {info['tipo_via']}")
        print(f"   Distância do trecho:  {info['distancia_m']:.0f} m")
        print(f"   Velocidade máx:       {info['vel_kmh']:.0f} km/h ({info['origem_vel']})")
        print(f"   Semáforo: {'✅ Sim (' + info['origem_semaforo'] + ')' if info['tem_semaforo'] else '❌ Não'}")
        print(f"   Lombada:  {'✅ Sim (' + info['origem_lombada'] + ')' if info['tem_lombada'] else '❌ Não'}")
        print(f"   Peso total: {info['tempo_seg']:.1f} seg")
        print(f"{'─' * 55}")

    def on_key(event):
        mapa_teclas = {
            '1': 'semaforo30', '2': 'semaforo60',
            '3': 'lombada40',  '4': 'lombada60', '5': 'lombada80',
            '6': 'velocidade', '7': 'limpar',
        }
        if event.key in mapa_teclas:
            estado['modo'] = mapa_teclas[event.key]
        elif event.key in ('q', 'Q', 'escape'):
            estado['aberto'] = False
            plt.close(fig)
            return
        atualizar_titulo()

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if not estado['modo']:
            mostrar_feedback('⚠️ Selecione uma ferramenta primeiro (teclas 1-7)', cor='yellow')
            return

        lon, lat = event.xdata, event.ydata
        no_prox  = nearest_nodes(G, lon, lat)
        u, v, key = nearest_edges(G, lon, lat)
        lon_real  = G.nodes[no_prox]['x']
        lat_real  = G.nodes[no_prox]['y']

        mostrar_snap(lon, lat, lon_real, lat_real)
        info     = obter_situacao_local(no_prox, u, v, key)
        nome_rua = info['nome_rua']
        imprimir_situacao(info)

        modo = estado['modo']

        if modo == 'semaforo30':
            edicoes_usuario['semaforos30'].add(no_prox)
            edicoes_usuario['semaforos60'].discard(no_prox)
            mostrar_feedback(f'🚦 Semáforo 30s adicionado no nó {no_prox}')

        elif modo == 'semaforo60':
            edicoes_usuario['semaforos60'].add(no_prox)
            edicoes_usuario['semaforos30'].discard(no_prox)
            mostrar_feedback(f'🚦 Semáforo 60s adicionado no nó {no_prox}')

        elif modo == 'lombada40':
            edicoes_usuario['lombadas40'].add(no_prox)
            edicoes_usuario['lombadas60'].discard(no_prox)
            edicoes_usuario['lombadas80'].discard(no_prox)
            mostrar_feedback('🏔️ Pardal 40km/h adicionado')

        elif modo == 'lombada60':
            edicoes_usuario['lombadas60'].add(no_prox)
            edicoes_usuario['lombadas40'].discard(no_prox)
            edicoes_usuario['lombadas80'].discard(no_prox)
            mostrar_feedback('🏔️ Pardal 60km/h adicionado')

        elif modo == 'lombada80':
            edicoes_usuario['lombadas80'].add(no_prox)
            edicoes_usuario['lombadas40'].discard(no_prox)
            edicoes_usuario['lombadas60'].discard(no_prox)
            mostrar_feedback('🏔️ Pardal 80km/h adicionado')

        elif modo == 'velocidade':
            mostrar_feedback(f'🏎️ Vel. atual: {info["vel_kmh"]:.0f} km/h — Aguardando entrada...', cor='cyan')
            fig.canvas.draw()
            fig.canvas.flush_events()
            nova_vel = pedir_velocidade_popup(nome_rua, info['vel_kmh'], info['origem_vel'])
            if nova_vel is not None:
                edicoes_usuario['velocidades'][(u, v, key)] = nova_vel
                mostrar_feedback(f'🏎️ Velocidade: {info["vel_kmh"]:.0f} → {nova_vel:.0f} km/h em: {nome_rua}')
            else:
                mostrar_feedback('⚠️ Cancelado ou valor inválido', cor='yellow')
                return

        elif modo == 'limpar':
            removido, detalhes = False, []
            for cat in ['semaforos30', 'semaforos60', 'lombadas40', 'lombadas60', 'lombadas80']:
                if no_prox in edicoes_usuario[cat]:
                    edicoes_usuario[cat].discard(no_prox)
                    removido = True
                    detalhes.append(cat)
            if (u, v, key) in edicoes_usuario['velocidades']:
                del edicoes_usuario['velocidades'][(u, v, key)]
                removido = True
                detalhes.append('velocidade')
            edicoes_usuario['removidos'].add(no_prox)
            if removido:
                mostrar_feedback(f'🧹 Removido: {", ".join(detalhes)} em {nome_rua}')
            else:
                mostrar_feedback(f'🧹 Nenhuma edição para remover em: {nome_rua}', cor='yellow')

        atualizar_pesos_do_grafo()
        atualizar_marcadores()
        salvar_edicoes()

    def on_close(event):
        estado['aberto'] = False

    atualizar_marcadores()
    atualizar_titulo()
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

    salvar_edicoes()
    print("\n✅ Editor fechado. Suas edições foram salvas.")


# ==========================================
# 10. INICIALIZAÇÃO E LOOP PRINCIPAL
# ==========================================
atualizar_pesos_do_grafo()

while True:
    print("\n" + "=" * 40)
    print("🗺️  SISTEMA DE ROTAS - MENU PRINCIPAL  🗺️")
    print("=" * 40)
    print("[1] Traçar Rota (Clicar no Mapa)")
    print("[2] Entrar no Modo de Edição")
    print("[3] Limpar TODAS as edições")
    print("[4] Sair")
    if tem_edicoes_ativas():
        qtd = (len(edicoes_usuario['semaforos30']) + len(edicoes_usuario['semaforos60']) +
               len(edicoes_usuario['lombadas40'])  + len(edicoes_usuario['lombadas60'])  +
               len(edicoes_usuario['lombadas80'])  + len(edicoes_usuario['velocidades']) +
               len(edicoes_usuario['removidos']))
        print(f"    📝 {qtd} edição(ões) ativas no mapa")

    opcao = input("Escolha uma opção: ").strip()

    if opcao == '1':
        modo_selecionar_rota()
    elif opcao == '2':
        modo_edicao()
    elif opcao == '3':
        resetar_todas_edicoes()
    elif opcao == '4':
        salvar_edicoes()
        print("💾 Edições salvas. Até logo!")
        break
    else:
        print("Opção inválida.")

        