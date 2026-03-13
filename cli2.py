import sys
import matplotlib
from scipy import stats
if sys.platform == 'darwin': 
    matplotlib.use('TkAgg')
    matplotlib.rcParams['figure.dpi'] = 100  
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
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
if 'traffic_calming' not in ox.settings.useful_tags_node: # verificar em que impacta no código
    ox.settings.useful_tags_node.append('traffic_calming')
if 'surface' not in ox.settings.useful_tags_way:
    ox.settings.useful_tags_way.append('surface')

G = ox.graph_from_place("Cruzeiro, Distrito Federal, Brazil", network_type="drive")

VELOCIDADE_MAX_MAPA_MS = 80 / 3.6
VELOCIDADE_PADRAO_KMH = {'trunk': 80, 'primary': 60, 'secondary': 60, 'tertiary': 40, 'residential': 40, 'unclassified': 40}

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
    if vel_kmh <= 40:   # Via coletora
        return 1.5
    elif vel_kmh <= 60: # Via arterial
        return 1.2
    else:               # Via expressa
        return 1.0

FRENAGEM_MS2 = 3.0        
ACELERACAO_MS2 = 2.0      
ESPERA_SEMAFORO_S = 30.0   
VEL_LOMBADA_KMH = 20.0    
ZONA_LOMBADA_M = 30.0     

# ==========================================
# 1. PERSISTÊNCIA DE EDIÇÕES (JSON)
# ==========================================
ARQUIVO_EDICOES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edicoes_mapa.json')

def salvar_edicoes():
    dados = {
        'semaforos60': list(edicoes_usuario['semaforos60']),
        'semaforos30': list(edicoes_usuario['semaforos30']),
        'lombadas40': list(edicoes_usuario['lombadas40']),
        'lombadas60': list(edicoes_usuario['lombadas60']),
        'lombadas80': list(edicoes_usuario['lombadas80']),
        'velocidades': {f"{u},{v},{k}": vel for (u, v, k), vel in edicoes_usuario['velocidades'].items()},
        'removidos': list(edicoes_usuario['removidos']),
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
            'lombadas40': set(dados.get('lombadas40', [])),
            'lombadas60': set(dados.get('lombadas60', [])),
            'lombadas80': set(dados.get('lombadas80', [])),
            'velocidades': {},
            'removidos': set(dados.get('removidos', [])),
        }
        for chave_str, vel in dados.get('velocidades', {}).items():
            partes = chave_str.split(',')
            edicoes['velocidades'][(int(partes[0]), int(partes[1]), int(partes[2]))] = vel
        return edicoes
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"⚠️ Erro ao carregar edições salvas: {e}. Usando mapa limpo.")
        return {
            'semaforos60': set(), 'semaforos30': set(), 
            'lombadas40': set(), 'lombadas60': set(), 'lombadas80': set(), 
            'velocidades': {}, 'removidos': set()
        }

edicoes_usuario = carregar_edicoes()
tem_edicoes_salvas = os.path.exists(ARQUIVO_EDICOES) and any([
    edicoes_usuario['semaforos60'], 
    edicoes_usuario['semaforos30'],    
    edicoes_usuario['lombadas40'], 
    edicoes_usuario['lombadas60'], 
    edicoes_usuario['lombadas80'],
    edicoes_usuario['velocidades'], 
    edicoes_usuario['removidos']
])
if tem_edicoes_salvas:
    qtd = (len(edicoes_usuario['semaforos60']) + len(edicoes_usuario['semaforos30']) +
           len(edicoes_usuario['lombadas40']) + len(edicoes_usuario['lombadas60']) + len(edicoes_usuario['lombadas80']) +
           len(edicoes_usuario['velocidades']) + len(edicoes_usuario['removidos']))
    print(f"📂 {qtd} edição(ões) carregada(s) do arquivo anterior.")

# ==========================================
#  MOTOR DE CÁLCULO DE PESOS
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
        vel_ms = max(0.1, vel_kmh / 3.6)
        tempo_base = distancia_m / vel_ms        
        penalidade_total = 0
        
        # Semaforos
        if v in sem60:
            penalidade_total += TABELA_PENALIDADES['semaforo60'] 
        elif v in sem30:
            penalidade_total += TABELA_PENALIDADES['semaforo30']
            
        # Lombadas
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
        edicoes_usuario['semaforos30'] or 
        edicoes_usuario['semaforos60'] or        
        edicoes_usuario['lombadas40'] or 
        edicoes_usuario['lombadas60'] or 
        edicoes_usuario['lombadas80'] or
        edicoes_usuario['velocidades'] or 
        edicoes_usuario['removidos']
    )

def calcular_rota_sem_edicoes(origem, destino):
    backup = {
        'semaforos60': edicoes_usuario['semaforos60'].copy(),
        'semaforos30': edicoes_usuario['semaforos30'].copy(),
        'lombadas40': edicoes_usuario['lombadas40'].copy(),
        'lombadas60': edicoes_usuario['lombadas60'].copy(),
        'lombadas80': edicoes_usuario['lombadas80'].copy(),
        'velocidades': edicoes_usuario['velocidades'].copy(),
        'removidos': edicoes_usuario['removidos'].copy(),
    }
        
    edicoes_usuario['semaforos60'] = set()
    edicoes_usuario['semaforos30'] = set()
    edicoes_usuario['lombadas40'] = set()
    edicoes_usuario['lombadas60'] = set()
    edicoes_usuario['lombadas80'] = set()
    edicoes_usuario['velocidades'] = {}
    edicoes_usuario['removidos'] = set()
    atualizar_pesos_do_grafo()
    
    inicio = time.time()
    caminho_orig, visitados_orig = a_star_animado(origem, destino)
    tempo_orig = time.time() - inicio
    
    stats_orig = calcular_estatisticas_dict(caminho_orig)
    
    for chave, valor in backup.items():
            edicoes_usuario[chave] = valor        
    atualizar_pesos_do_grafo()
    
    return caminho_orig, visitados_orig, tempo_orig, stats_orig

# ==========================================
# ALGORITMOS A*
# ==========================================
def heuristica_tempo(n1, n2):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    distancia_reta = ((x1-x2)**2 + (y1-y2)**2)**0.5 * 111139
    return distancia_reta / VELOCIDADE_MAX_MAPA_MS

def a_star_animado(inicio, fim):
    fila = []
    heapq.heappush(fila, (0, inicio))
    g_custo = {inicio: 0}
    pais = {}
    visitados = []

    while fila:
        _, atual = heapq.heappop(fila)
        visitados.append(atual)

        if atual == fim: break

        for vizinho in G.neighbors(atual):
            peso = G[atual][vizinho][0]['tempo_segundos']
            novo_g = g_custo[atual] + peso

            if vizinho not in g_custo or novo_g < g_custo[vizinho]:
                g_custo[vizinho] = novo_g
                f = novo_g + heuristica_tempo(vizinho, fim)
                pais[vizinho] = atual
                heapq.heappush(fila, (f, vizinho))

    caminho = []
    atual = fim
    while atual in pais:
        caminho.append(atual)
        atual = pais[atual]
    caminho.append(inicio)
    caminho.reverse()
    return caminho, visitados

# ==========================================
# ALGORITMO DIJKSTRA
# ==========================================
def dijkstra_animado(inicio, fim):
    fila = []
    heapq.heappush(fila, (0, inicio))
    g_custo = {inicio: 0}
    pais = {}
    visitados = []

    while fila:
        _, atual = heapq.heappop(fila)
        visitados.append(atual)

        if atual == fim: break

        for vizinho in G.neighbors(atual):
            peso = G[atual][vizinho][0]['tempo_segundos']
            novo_g = g_custo[atual] + peso

            if vizinho not in g_custo or novo_g < g_custo[vizinho]:
                g_custo[vizinho] = novo_g
                pais[vizinho] = atual
                heapq.heappush(fila, (novo_g, vizinho))

    caminho = []
    atual = fim
    while atual in pais:
        caminho.append(atual)
        atual = pais[atual]
    caminho.append(inicio)
    caminho.reverse()
    return caminho, visitados

# ==========================================
#  INTERFACES E PAINEL DE ESTATÍSTICAS
# ==========================================
def calcular_estatisticas_dict(caminho):
    distancia_total = 0
    tempo_total = 0
    semaforos60_passados = 0
    semaforos30_passados = 0
    lombadas40_passadas = 0
    lombadas60_passadas = 0
    lombadas80_passadas = 0

    semaforos60_ativos = edicoes_usuario['semaforos60'].copy()
    semaforos30_ativos = edicoes_usuario['semaforos30'].copy()
    lombadas40_ativas = edicoes_usuario['lombadas40'].copy()
    lombadas60_ativas = edicoes_usuario['lombadas60'].copy()
    lombadas80_ativas = edicoes_usuario['lombadas80'].copy()
    for n, d in G.nodes(data=True):
        if d.get('highway') == 'traffic_signals': semaforos30_ativos.add(n)
        if d.get('traffic_calming') in ['bump', 'hump']: lombadas40_ativas.add(n)

    for i in range(len(caminho)-1):
        u = caminho[i]
        v = caminho[i+1]
        dados_aresta = G[u][v][0]
        distancia_total += dados_aresta['length']
        tempo_total += dados_aresta['tempo_segundos']
        if v in semaforos60_ativos: semaforos60_passados += 1
        if v in semaforos30_ativos: semaforos30_passados += 1
        if v in lombadas40_ativas: lombadas40_passadas += 1
        if v in lombadas60_ativas: lombadas60_passadas += 1
        if v in lombadas80_ativas: lombadas80_passadas += 1
    vel_media = (distancia_total / tempo_total) * 3.6 if tempo_total > 0 else 0
    return {
        'distancia_km': distancia_total / 1000,
        'tempo_min': tempo_total / 60,
        'vel_media': vel_media,
        'semaforos60': semaforos60_passados,
        'semaforos30': semaforos30_passados,
        'lombadas40': lombadas40_passadas,
        'lombadas60': lombadas60_passadas,
        'lombadas80': lombadas80_passadas,
    }

def formatar_estatisticas(stats, visitados, tempo_execucao, titulo="ESTATÍSTICAS DA ROTA"):
    total_sem = stats['semaforos30'] + stats['semaforos60']
    total_lom = stats['lombadas40'] + stats['lombadas60'] + stats['lombadas80']
    return (
        f"{titulo}\n\n"
        f"⏱️ A* Tempo de Busca:\n   {tempo_execucao:.4f} seg\n\n"
        f"🔍 Nós Explorados:\n   {len(visitados)} cruzamentos\n\n"
        f"🛣️ Distância Total:\n   {stats['distancia_km']:.2f} km\n\n"
        f"⏳ Tempo Estimado:\n   {stats['tempo_min']:.1f} minutos\n\n"
        f"🏎️ Velocidade Média:\n   {stats['vel_media']:.1f} km/h\n\n"
        f"🚦 Semáforos: {total_sem} (30s: {stats['semaforos30']}, 60s: {stats['semaforos60']})\n"
        f"🏔️ Lombadas: {total_lom}"
    )

def formatar_comparacao(stats_mod, stats_orig):
    def diff(val_mod, val_orig, fmt=".1f", inverso=False):
        delta = val_mod - val_orig
        if abs(delta) < 0.01: return "="
        sinal = "+" if delta > 0 else ""
        if inverso:
            cor = "⬇️" if delta < 0 else "⬆️"
        else:
            cor = "⬆️" if delta > 0 else "⬇️"
        return f"{cor}{sinal}{delta:{fmt}}"
    
    return (
        f"COMPARAÇÃO\n"
        f"Editado vs Original\n\n"
        f"🛣️ Distância:\n"
        f"   {stats_mod['distancia_km']:.2f} vs {stats_orig['distancia_km']:.2f} km\n"
        f"   {diff(stats_mod['distancia_km'], stats_orig['distancia_km'], '.2f', inverso=True)}\n\n"
        f"⏳ Tempo:\n"
        f"   {stats_mod['tempo_min']:.1f} vs {stats_orig['tempo_min']:.1f} min\n"
        f"   {diff(stats_mod['tempo_min'], stats_orig['tempo_min'], inverso=True)}\n\n"
        f"🏎️ Vel. Média:\n"
        f"   {stats_mod['vel_media']:.1f} vs {stats_orig['vel_media']:.1f}\n"
        f"   {diff(stats_mod['vel_media'], stats_orig['vel_media'])}\n\n"
        f"🚦 Semáforos:\n"
        f"   {stats_mod['semaforos']} vs {stats_orig['semaforos']}\n\n"
        f"🏔️ Lombadas:\n"
        f"   {stats_mod['lombadas']} vs {stats_orig['lombadas']}"
    )

def exibir_mapa_com_painel(caminho, visitados, tempo_execucao, animar=False,
                           caminho_original=None, visitados_orig=None, tempo_orig=None,
                           stats_orig=None):
    comparando = caminho_original is not None
    rotas_identicas = comparando and caminho == caminho_original
    
    if comparando:
        fig = plt.figure(figsize=(18, 8))
        fig.canvas.manager.set_window_title('Resultado da Rota — Comparação com Mapa Original')
        gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1])
        ax_mapa = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
        ax_comp = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(14, 8))
        fig.canvas.manager.set_window_title('Resultado da Rota')
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_mapa = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
    
    ax_info.set_facecolor('#111111')
    ax_info.axis('off')
    stats_mod = calcular_estatisticas_dict(caminho)
    texto_stats = formatar_estatisticas(stats_mod, visitados, tempo_execucao,
                                        titulo="ROTA EDITADA ✏️" if comparando else "ESTATÍSTICAS DA ROTA")
    ax_info.text(0.1, 0.5, texto_stats, color='lime', fontsize=11, va='center', ha='left', family='monospace')
    
    if comparando and stats_orig is not None:
        ax_comp.set_facecolor('#111111')
        ax_comp.axis('off')
        texto_comp = formatar_comparacao(stats_mod, stats_orig)
        if rotas_identicas:
            texto_comp += "\n\n📌 MESMA ROTA\n   As edições não mudaram\n   o caminho, mas alteraram\n   os tempos de percurso."
        ax_comp.text(0.1, 0.5, texto_comp, color='#88CCFF', fontsize=11, va='center', ha='left', family='monospace')
    
    ox.plot_graph(G, ax=ax_mapa, show=False, close=False, node_size=0, edge_color='#555555', edge_alpha=0.5, bgcolor='#111111')
    fig.patch.set_facecolor('#111111')

    if animar:
        coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in visitados]
        scat = ax_mapa.scatter([], [], c='cyan', s=15, zorder=3)
        passo = max(1, len(coords) // 50)
        
        for i in range(1, len(coords) + passo, passo):
            scat.set_offsets(coords[:min(i, len(coords))])
            ax_mapa.set_title(f"A* Explorando... ({min(i, len(coords))} nós)", color='cyan', fontsize=14)
            plt.pause(0.01)
    
    # ---------------------------------------------------------
    # CORREÇÃO AQUI: Controle de Z-Order e Espessura das Rotas
    # ---------------------------------------------------------
    if comparando:
        # 1. Rota ORIGINAL primeiro: Mais GROSSA e com transparência. Serve como "borda" ou "fundo".
        ox.plot_graph_route(G, caminho_original, ax=ax_mapa, route_color='#FF6600',
                           route_linewidth=8, node_size=0, route_alpha=0.5)
        
        # 2. Rota EDITADA por cima: Mais FINA e opaca. Passa por "dentro" da rota original se forem iguais.
        ox.plot_graph_route(G, caminho, ax=ax_mapa, route_color='#00FF88',
                           route_linewidth=3, node_size=0, route_alpha=1.0)
        
        orig_x, orig_y = G.nodes[caminho[0]]['x'], G.nodes[caminho[0]]['y']
        dest_x, dest_y = G.nodes[caminho[-1]]['x'], G.nodes[caminho[-1]]['y']
        ax_mapa.scatter([orig_x], [orig_y], c='white', s=150, marker='o', 
                       edgecolors='black', linewidths=2, zorder=8, label='Origem')
        ax_mapa.scatter([dest_x], [dest_y], c='white', s=150, marker='*', 
                       edgecolors='black', linewidths=2, zorder=8, label='Destino')
    else:
        ox.plot_graph_route(G, caminho, ax=ax_mapa, route_color='red', route_linewidth=4, node_size=0)
    
    if comparando:
        from matplotlib.lines import Line2D
        titulo_mapa = "🟢 Rota Editada  vs  🟠 Rota Original"
        if rotas_identicas:
            titulo_mapa = "Mesmo caminho — só os tempos mudaram"
        ax_mapa.set_title(titulo_mapa, color='white', fontsize=13, fontweight='bold')
        legenda = [
            Line2D([0], [0], color='#00FF88', linewidth=3, label='Rota Editada'),
            Line2D([0], [0], color='#FF6600', linewidth=8, alpha=0.5, label='Rota Original'),
        ]
        ax_mapa.legend(handles=legenda, loc='lower right', facecolor='#222222', labelcolor='white', fontsize=10)
    else:
        ax_mapa.set_title("A* - Busca Concluída!", color='lime', fontsize=16, fontweight='bold')
    
    plt.show()

def modo_selecionar_rota():
    print("\n🗺️ O mapa vai abrir. CLIQUE em 2 pontos: primeiro na ORIGEM, depois no DESTINO.")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title('Selecione Origem e Destino')
    ox.plot_graph(G, ax=ax, show=False, close=False, node_size=5, edge_color='#555555', bgcolor='#111111')
    ax.set_title("CLIQUE NA ORIGEM E DEPOIS NO DESTINO", color='white')
    
    pontos = plt.ginput(2, timeout=0) 
    fechar_janela_seguro(fig)
    
    if len(pontos) < 2:
        print("Você não clicou 2 vezes. Abortando...")
        return
        
    lon_o, lat_o = pontos[0]
    lon_d, lat_d = pontos[1]
    
    no_o = ox.distance.nearest_nodes(G, lon_o, lat_o)
    no_d = ox.distance.nearest_nodes(G, lon_d, lat_d)
    
    visualizacao = input("\nComo deseja visualizar a rota?\n[1] Mapa Estático\n[2] Animação do A*\nEscolha: ")
    
    print("\nCalculando rota, por favor aguarde...")
    inicio_tempo = time.time()
    caminho, visitados = a_star_animado(no_o, no_d)
    tempo_execucao = time.time() - inicio_tempo
    
    if tem_edicoes_ativas():
        print("Calculando rota original (sem edições) para comparação...")
        caminho_orig, visitados_orig, tempo_orig, stats_orig = calcular_rota_sem_edicoes(no_o, no_d)
        
        stats_mod = calcular_estatisticas_dict(caminho)
        nos_editado = set(caminho)
        nos_original = set(caminho_orig)
        nos_em_comum = nos_editado & nos_original
        nos_so_editado = nos_editado - nos_original
        nos_so_original = nos_original - nos_editado
        
        print(f"\n{'=' * 60}")
        print(f"🔍 DEBUG — COMPARAÇÃO DE ROTAS")
        print(f"{'=' * 60}")
        print(f"  Rotas idênticas? {'SIM ✔️' if caminho == caminho_orig else 'NÃO ✖️ (caminhos diferentes)'}")
        print(f"  Nós na rota editada:   {len(caminho)}")
        print(f"  Nós na rota original:  {len(caminho_orig)}")
        print(f"  Nós em comum:          {len(nos_em_comum)}")
        print(f"  Nós só na editada:     {len(nos_so_editado)}")
        print(f"  Nós só na original:    {len(nos_so_original)}")
        print(f"{'─' * 60}")
        print(f"  EDITADA  → Dist: {stats_mod['distancia_km']:.3f} km | Tempo: {stats_mod['tempo_min']:.2f} min | Vel: {stats_mod['vel_media']:.1f} km/h")
        print(f"  ORIGINAL → Dist: {stats_orig['distancia_km']:.3f} km | Tempo: {stats_orig['tempo_min']:.2f} min | Vel: {stats_orig['vel_media']:.1f} km/h")
        diff_dist = stats_mod['distancia_km'] - stats_orig['distancia_km']
        diff_tempo = stats_mod['tempo_min'] - stats_orig['tempo_min']
        print(f"  DIFERENÇA→ Dist: {'+' if diff_dist >= 0 else ''}{diff_dist:.3f} km | Tempo: {'+' if diff_tempo >= 0 else ''}{diff_tempo:.2f} min")
        print(f"  Edições ativas: {len(edicoes_usuario['semaforos'])} semáforos, {len(edicoes_usuario['lombadas'])} lombadas, {len(edicoes_usuario['velocidades'])} velocidades")
        print(f"{'=' * 60}\n")
        
        exibir_mapa_com_painel(caminho, visitados, tempo_execucao, animar=(visualizacao == '2'),
                               caminho_original=caminho_orig, visitados_orig=visitados_orig,
                               tempo_orig=tempo_orig, stats_orig=stats_orig)
    else:
        exibir_mapa_com_painel(caminho, visitados, tempo_execucao, animar=(visualizacao == '2'))

def pedir_velocidade_popup(nome_rua, vel_atual, origem_vel):
    prompt = f"Via: {nome_rua}\nVelocidade atual: {vel_atual:.0f} km/h ({origem_vel})"
    
    if TK_DISPONIVEL:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            nova_vel = simpledialog.askfloat(
                "Alterar Velocidade",
                f"{prompt}\n\nNova velocidade (km/h):",
                parent=root, minvalue=1, maxvalue=200,
                initialvalue=vel_atual
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
    print("  [1] 🚦 Semáforo    [2] 🏔️  Lombada")
    print("  [3] 🏎️  Velocidade  [4] 🧹 Limpar edição")
    print("  [Q] Sair do editor")
    print("─" * 50)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.manager.set_window_title('Editor Interativo do Mapa')
    fig.patch.set_facecolor('#111111')
    ox.plot_graph(G, ax=ax, show=False, close=False, node_size=8, edge_color='#444444', bgcolor='#111111')

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
        'limpar':     '🧹 LIMPAR EDIÇÃO'
    }
    
    def atualizar_titulo():
        if estado['modo']:
            titulo = f"EDITOR | Ferramenta: {nomes_modo[estado['modo']]}"
            cor = 'cyan'
        else:
            titulo = "EDITOR | Pressione 1, 2, 3, 4, 5, 6 ou 7 para escolher ferramenta"
            cor = 'white'
        ax.set_title(titulo, color=cor, fontsize=13, fontweight='bold')
        fig.canvas.draw_idle()

    def atualizar_marcadores():
        if estado['scat_sem']: estado['scat_sem'].remove()
        if estado['scat_lom']: estado['scat_lom'].remove()
        if estado['scat_vel']: estado['scat_vel'].remove()
        estado['scat_sem'] = None
        estado['scat_lom'] = None
        estado['scat_vel'] = None


        todos_semaforos = edicoes_usuario['semaforos30'] | edicoes_usuario['semaforos60'] # Unifica os marcadores de semáforo
        if todos_semaforos:
            sx = [G.nodes[n]['x'] for n in todos_semaforos]
            sy = [G.nodes[n]['y'] for n in todos_semaforos]
            estado['scat_sem'] = ax.scatter(sx, sy, c='red', s=100, marker='s', label='🚦 Semáforos', zorder=5, edgecolors='white', linewidths=0.5)

        todas_lombadas = edicoes_usuario['lombadas40'] | edicoes_usuario['lombadas60'] | edicoes_usuario['lombadas80'] # Unifica os marcadores de lombada
        if todas_lombadas:
            lx = [G.nodes[n]['x'] for n in todas_lombadas]
            ly = [G.nodes[n]['y'] for n in todas_lombadas]
            estado['scat_lom'] = ax.scatter(lx, ly, c='orange', s=100, marker='^', label='🏔️ Lombadas', zorder=5, edgecolors='white', linewidths=0.5)

        nos_vel = set()
        for (u, v, k) in edicoes_usuario['velocidades']:
            nos_vel.add(u)
            nos_vel.add(v)
        if nos_vel:
            vx = [G.nodes[n]['x'] for n in nos_vel]
            vy = [G.nodes[n]['y'] for n in nos_vel]
            estado['scat_vel'] = ax.scatter(vx, vy, c='lime', s=60, marker='d', label='🏎️ Vel. editada', zorder=5, edgecolors='white', linewidths=0.5)

        handles = [h for h in [estado['scat_sem'], estado['scat_lom'], estado['scat_vel']] if h is not None]
        if handles:
            ax.legend(handles=handles, loc='upper right', facecolor='#222222', labelcolor='white', fontsize=10)
        else:
            leg = ax.get_legend()
            if leg: leg.remove()
        fig.canvas.draw_idle()

    def mostrar_feedback(msg, cor='lime'):
        if estado['texto_status']: estado['texto_status'].remove()
        estado['texto_status'] = ax.text(
            0.5, 0.02, msg, transform=ax.transAxes, color=cor, fontsize=11, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#222222', edgecolor=cor, alpha=0.9), zorder=10
        )
        fig.canvas.draw_idle()

    def mostrar_snap(lon_clique, lat_clique, lon_real, lat_real):
        if estado['marcador_clique']: estado['marcador_clique'].remove()
        if estado['linha_snap']:
            for l in estado['linha_snap']: l.remove()
        
        estado['marcador_clique'] = ax.scatter([lon_real], [lat_real], c='white', s=200, marker='o', edgecolors='yellow', linewidths=2, zorder=8, alpha=0.8)
        
        dist = ((lon_clique - lon_real)**2 + (lat_clique - lat_real)**2)**0.5
        if dist > 1e-6:
            estado['linha_snap'] = ax.plot([lon_clique, lon_real], [lat_clique, lat_real], '--', color='yellow', linewidth=1.5, alpha=0.7, zorder=7)
        else:
            estado['linha_snap'] = None
        fig.canvas.draw_idle()

    def obter_situacao_local(no_prox, u, v, key):
        dados = G[u][v][key]
        nome_rua = dados.get('name', 'Sem nome')
        distancia_m = dados.get('length', 0)
        tipo_via = dados.get('highway', 'residential')
        if isinstance(tipo_via, list): tipo_via = tipo_via[0]

        if (u, v, key) in edicoes_usuario['velocidades']:
            vel_kmh = edicoes_usuario['velocidades'][(u, v, key)]
            origem_vel = 'editada pelo usuário'
        else:
            vel_osm = dados.get('maxspeed')
            if vel_osm and not isinstance(vel_osm, list):
                vel_kmh = float(vel_osm)
                origem_vel = 'do OpenStreetMap'
            else:
                vel_kmh = VELOCIDADE_PADRAO_KMH.get(tipo_via, 30)
                origem_vel = f'padrão p/ via "{tipo_via}"'

        tem_semaforo = False
        origem_semaforo = ''
        if no_prox in edicoes_usuario['semaforos30']:
            tem_semaforo, origem_semaforo = True, '30s (adicionado)'
        elif no_prox in edicoes_usuario['semaforos60']:
            tem_semaforo, origem_semaforo = True, '60s (adicionado)'
        elif G.nodes[no_prox].get('highway') == 'traffic_signals' and no_prox not in edicoes_usuario['removidos']:
            tem_semaforo, origem_semaforo = True, 'original do mapa'

        # Verificação de Lombadas/Pardais (Nova Lógica)
        tem_lombada = False
        origem_lombada = ''
        for cat in ['lombadas40', 'lombadas60', 'lombadas80']:
            if no_prox in edicoes_usuario[cat]:
                tem_lombada, origem_lombada = True, f'{cat[-2:]}km/h (adicionada)'
                break

        tempo_seg = dados.get('tempo_segundos', 0)

        return {
            'nome_rua': nome_rua, 'distancia_m': distancia_m, 'tipo_via': tipo_via,
            'vel_kmh': vel_kmh, 'origem_vel': origem_vel, 'tem_semaforo': tem_semaforo, 
            'origem_semaforo': origem_semaforo, 'tem_lombada': tem_lombada, 
            'origem_lombada': origem_lombada, 'tempo_seg': tempo_seg,
        }

    def imprimir_situacao(info):
        print(f"\n{'─' * 55}")
        print(f"📍 {info['nome_rua']}")
        print(f"   Tipo de via: {info['tipo_via']}")
        print(f"   Distância do trecho: {info['distancia_m']:.0f} m")
        print(f"   Velocidade máx: {info['vel_kmh']:.0f} km/h ({info['origem_vel']})")
        
        if info['tem_semaforo']:
            print(f"   Semáforo: ✅ Sim ({info['origem_semaforo']})")
        else: 
            print(f"   Semáforo: ❌ Não")
            
        if info['tem_lombada']:
            print(f"   Lombada/Pardal: ✅ Sim ({info['origem_lombada']})")
        else: 
            print(f"   Lombada/Pardal: ❌ Não")
            
        print(f"   Peso total calculado: {info['tempo_seg']:.1f} seg")
        print(f"{'─' * 55}")

    def on_key(event):
        if event.key == '1': 
            estado['modo'] = 'semaforo30'
        elif event.key == '2': 
            estado['modo'] = 'semaforo60'
        elif event.key == '3': 
            estado['modo'] = 'lombada40'
        elif event.key == '4': 
            estado['modo'] = 'lombada60'
        elif event.key == '5': 
            estado['modo'] = 'lombada80'
        elif event.key == '6': 
            estado['modo'] = 'velocidade'
        elif event.key == '7': 
            estado['modo'] = 'limpar'
        elif event.key in ('q', 'Q', 'escape'):
            estado['aberto'] = False
            plt.close(fig)
            return
        atualizar_titulo()

    def on_click(event):
        if event.inaxes != ax or event.button != 1: return
        if not estado['modo']:
            mostrar_feedback('⚠️ Selecione uma ferramenta primeiro (teclas 1-7)', cor='yellow')
            return

        lon, lat = event.xdata, event.ydata
        no_prox = ox.distance.nearest_nodes(G, lon, lat)
        u, v, key = ox.distance.nearest_edges(G, lon, lat)
        lon_real = G.nodes[no_prox]['x']
        lat_real = G.nodes[no_prox]['y']

        mostrar_snap(lon, lat, lon_real, lat_real)
        info = obter_situacao_local(no_prox, u, v, key)
        imprimir_situacao(info)
        nome_rua = info['nome_rua']

        if estado['modo'] == 'semaforo30':
            edicoes_usuario['semaforos30'].add(no_prox)
            edicoes_usuario['semaforos60'].discard(no_prox) 
            mostrar_feedback(f'🚦 Semáforo 30s adicionado no nó {no_prox}')

        elif estado['modo'] == 'semaforo60':
            edicoes_usuario['semaforos60'].add(no_prox)
            edicoes_usuario['semaforos30'].discard(no_prox)
            mostrar_feedback(f'🚦 Semáforo 60s adicionado no nó {no_prox}')

        elif estado['modo'] == 'lombada40':
            edicoes_usuario['lombadas40'].add(no_prox)
            edicoes_usuario['lombadas60'].discard(no_prox)
            edicoes_usuario['lombadas80'].discard(no_prox)
            mostrar_feedback(f'🏔️ Pardal 40km/h adicionado')
        
        elif estado['modo'] == 'lombada60':
            edicoes_usuario['lombadas60'].add(no_prox)
            edicoes_usuario['lombadas60'].discard(no_prox)
            edicoes_usuario['lombadas80'].discard(no_prox)
            mostrar_feedback(f'🏔️ Pardal 60km/h adicionado')
        
        elif estado['modo'] == 'lombada80':
            edicoes_usuario['lombadas80'].add(no_prox)
            edicoes_usuario['lombadas60'].discard(no_prox)
            edicoes_usuario['lombadas80'].discard(no_prox)
            mostrar_feedback(f'🏔️ Pardal 80km/h adicionado')

        elif estado['modo'] == 'velocidade':
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

        elif estado['modo'] == 'limpar':
            removido = False
            detalhes = []

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
                mostrar_feedback(f'🧹 Nenhuma edição sua p/ remover em: {nome_rua}', cor='yellow')

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


atualizar_pesos_do_grafo()

# ==========================================
# 5️⃣ LOOP PRINCIPAL
# ==========================================
while True:
    print("\n" + "="*40)
    print("🗺️  SISTEMA DE ROTAS - MENU PRINCIPAL  🗺️")
    print("="*40)
    print("[1] Traçar Rota (Clicar no Mapa)")
    print("[2] Entrar no Modo de Edição (Adicionar Semáforos/Pesos)")
    print("[3] Limpar TODAS as edições (Reverter mapa original)")
    print("[4] Sair")
    if tem_edicoes_ativas():
        qtd = (len(edicoes_usuario['semaforos30']) +
               len(edicoes_usuario['semaforos60']) +
               len(edicoes_usuario['lombadas40']) + 
               len(edicoes_usuario['lombadas60']) + 
               len(edicoes_usuario['lombadas80']) + 
               len(edicoes_usuario['velocidades']) + 
               len(edicoes_usuario['removidos']))
        print(f"    📝 {qtd} edição(ões) ativas no mapa")
    
    opcao = input("Escolha uma opção: ")
    
    if opcao == '1': modo_selecionar_rota()
    elif opcao == '2': modo_edicao()
    elif opcao == '3':
        edicoes_usuario = carregar_edicoes()
        atualizar_pesos_do_grafo()
        salvar_edicoes()
        print("🔄 Mapa resetado para os padrões de fábrica!")
    elif opcao == '4':
        salvar_edicoes()
        print("💾 Edições salvas. Até logo!")
        break
    else: print("Opção inválida.")