import sys
import os
import json
import time
import heapq
import tkinter as tk
from tkinter import simpledialog, messagebox
import matplotlib
if sys.platform == 'darwin': 
    matplotlib.use('TkAgg')
    matplotlib.rcParams['figure.dpi'] = 100  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import osmnx as ox
import networkx as nx

# ==========================================
# CONFIGURAÇÕES GERAIS E FÍSICAS
# ==========================================
VELOCIDADE_PADRAO_KMH = {'trunk': 80, 'primary': 60, 'secondary': 50, 'tertiary': 40, 'residential': 30, 'unclassified': 30}
VELOCIDADE_MAX_MAPA_MS = 80 / 3.6
FRENAGEM_MS2 = 3.0        
ACELERACAO_MS2 = 2.0      
ESPERA_SEMAFORO_S = 15.0   
VEL_LOMBADA_KMH = 20.0    
ZONA_LOMBADA_M = 30.0     
ARQUIVO_EDICOES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edicoes_mapa.json')

# ==========================================
# CLASSE 1: MOTOR DE ROTEAMENTO (LÓGICA)
# ==========================================
class MotorRoteamento:
    def __init__(self, local="Cruzeiro, Distrito Federal, Brazil"):
        self.configurar_osmnx()
        print(f"Baixando o mapa de {local}...")
        self.G = ox.graph_from_place(local, network_type="drive")
        self.edicoes = self.carregar_edicoes()
        self.atualizar_pesos()

    def configurar_osmnx(self):
        if 'traffic_calming' not in ox.settings.useful_tags_node:
            ox.settings.useful_tags_node.append('traffic_calming')
        if 'surface' not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way.append('surface')

    def carregar_edicoes(self):
        padrao = {'semaforos': set(), 'lombadas': set(), 'velocidades': {}, 'removidos': set()}
        if not os.path.exists(ARQUIVO_EDICOES): return padrao
        try:
            with open(ARQUIVO_EDICOES, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            edicoes = {
                'semaforos': set(dados.get('semaforos', [])),
                'lombadas': set(dados.get('lombadas', [])),
                'removidos': set(dados.get('removidos', [])),
                'velocidades': {}
            }
            for chave, vel in dados.get('velocidades', {}).items():
                u, v, k = map(int, chave.split(','))
                edicoes['velocidades'][(u, v, k)] = vel
            return edicoes
        except Exception as e:
            print(f"⚠️ Erro ao carregar edições: {e}")
            return padrao

    def salvar_edicoes(self):
        dados = {
            'semaforos': list(self.edicoes['semaforos']),
            'lombadas': list(self.edicoes['lombadas']),
            'removidos': list(self.edicoes['removidos']),
            'velocidades': {f"{u},{v},{k}": vel for (u, v, k), vel in self.edicoes['velocidades'].items()}
        }
        with open(ARQUIVO_EDICOES, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)

    def penalidade_semaforo(self, vel_ms):
        return ESPERA_SEMAFORO_S + (vel_ms / FRENAGEM_MS2) + (vel_ms / ACELERACAO_MS2)

    def penalidade_lombada(self, vel_ms):
        vel_lombada_ms = VEL_LOMBADA_KMH / 3.6
        if vel_ms <= vel_lombada_ms: return 0.0  
        dif_vel = vel_ms - vel_lombada_ms
        t_frenagem = dif_vel / FRENAGEM_MS2
        t_aceleracao = dif_vel / ACELERACAO_MS2
        t_extra_zona = (ZONA_LOMBADA_M / vel_lombada_ms) - (ZONA_LOMBADA_M / vel_ms)
        return t_frenagem + t_aceleracao + t_extra_zona

    def atualizar_pesos(self):
        semaforos = self.edicoes['semaforos'].copy()
        lombadas = self.edicoes['lombadas'].copy()
        
        for n, d in self.G.nodes(data=True):
            if n in self.edicoes['removidos']: continue
            if d.get('highway') == 'traffic_signals': semaforos.add(n)
            if d.get('traffic_calming') in ['bump', 'hump']: lombadas.add(n)

        for u, v, key, d in self.G.edges(keys=True, data=True):
            dist = d.get('length', 0)
            tipo_via = d.get('highway', 'residential')
            if isinstance(tipo_via, list): tipo_via = tipo_via[0]
            
            if (u, v, key) in self.edicoes['velocidades']:
                vel_kmh = self.edicoes['velocidades'][(u, v, key)]
            else:
                vel_kmh = d.get('maxspeed', VELOCIDADE_PADRAO_KMH.get(tipo_via, 30))
                vel_kmh = float(vel_kmh[0]) if isinstance(vel_kmh, list) else float(vel_kmh)
                
            vel_ms = max(0.1, vel_kmh / 3.6)
            tempo_s = dist / vel_ms
            
            if v in semaforos: tempo_s += self.penalidade_semaforo(vel_ms)
            if v in lombadas: tempo_s += self.penalidade_lombada(vel_ms)
            self.G[u][v][key]['tempo_segundos'] = tempo_s

    # --- ALGORITMOS DE BUSCA ---
    def dijkstra(self, inicio, fim):
        fila = [(0, inicio)]
        distancias = {inicio: 0}
        pais = {}
        visitados = []

        while fila:
            custo_atual, atual = heapq.heappop(fila)
            visitados.append(atual)

            if atual == fim: break
            if custo_atual > distancias.get(atual, float('inf')): continue

            for vizinho in self.G.neighbors(atual):
                peso = self.G[atual][vizinho][0]['tempo_segundos']
                novo_custo = custo_atual + peso

                if novo_custo < distancias.get(vizinho, float('inf')):
                    distancias[vizinho] = novo_custo
                    pais[vizinho] = atual
                    heapq.heappush(fila, (novo_custo, vizinho))

        caminho = []
        atual = fim
        while atual in pais:
            caminho.append(atual)
            atual = pais[atual]
        if atual == inicio: caminho.append(inicio)
        caminho.reverse()
        
        return caminho if len(caminho) > 1 else [], visitados

    def heuristica_tempo(self, n1, n2):
        x1, y1 = self.G.nodes[n1]['x'], self.G.nodes[n1]['y']
        x2, y2 = self.G.nodes[n2]['x'], self.G.nodes[n2]['y']
        distancia_reta = ((x1-x2)**2 + (y1-y2)**2)**0.5 * 111139
        return distancia_reta / VELOCIDADE_MAX_MAPA_MS

    def a_star(self, inicio, fim):
        fila = [(0, inicio)]
        g_custo = {inicio: 0}
        pais = {}
        visitados = []

        while fila:
            _, atual = heapq.heappop(fila)
            visitados.append(atual)

            if atual == fim: break

            for vizinho in self.G.neighbors(atual):
                peso = self.G[atual][vizinho][0]['tempo_segundos']
                novo_g = g_custo.get(atual, float('inf')) + peso

                if novo_g < g_custo.get(vizinho, float('inf')):
                    g_custo[vizinho] = novo_g
                    f = novo_g + self.heuristica_tempo(vizinho, fim)
                    pais[vizinho] = atual
                    heapq.heappush(fila, (f, vizinho))

        caminho = []
        atual = fim
        while atual in pais:
            caminho.append(atual)
            atual = pais[atual]
        if atual == inicio: caminho.append(inicio)
        caminho.reverse()
        
        return caminho if len(caminho) > 1 else [], visitados

    # --- ESTATÍSTICAS E EDIÇÕES ---
    def calcular_estatisticas(self, caminho):
        dist_total, tempo_total, sem_pass, lom_pass = 0, 0, 0, 0
        semaforos = self.edicoes['semaforos']
        lombadas = self.edicoes['lombadas']
        
        for i in range(len(caminho)-1):
            u, v = caminho[i], caminho[i+1]
            dados = self.G[u][v][0]
            dist_total += dados['length']
            tempo_total += dados['tempo_segundos']
            if v in semaforos or self.G.nodes[v].get('highway') == 'traffic_signals': sem_pass += 1
            if v in lombadas or self.G.nodes[v].get('traffic_calming') in ['bump', 'hump']: lom_pass += 1

        vel_media = (dist_total / tempo_total) * 3.6 if tempo_total > 0 else 0
        return {
            'dist_km': dist_total / 1000, 'tempo_min': tempo_total / 60,
            'vel_media': vel_media, 'semaforos': sem_pass, 'lombadas': lom_pass
        }

    def alterar_velocidade_trecho(self, p1, p2, nova_vel):
        try:
            caminho_trecho = nx.shortest_path(self.G, p1, p2, weight='length')
            for i in range(len(caminho_trecho)-1):
                u, v = caminho_trecho[i], caminho_trecho[i+1]
                key = list(self.G[u][v].keys())[0]
                self.edicoes['velocidades'][(u, v, key)] = nova_vel
            self.atualizar_pesos()
            self.salvar_edicoes()
            return len(caminho_trecho) - 1
        except nx.NetworkXNoPath:
            return 0


# ==========================================
# CLASSE 2: INTERFACE GRÁFICA (UI)
# ==========================================
class InterfaceMapa:
    def __init__(self, root, motor):
        self.root = root
        self.motor = motor
        self.root.title("Rastreador de Rotas - A* e Dijkstra")
        self.root.geometry("1400x800")
        
        self.modo_atual = None  
        self.origem = None
        self.destino = None
        self.ponto_edicao = None  
        self.animar_var = tk.BooleanVar(value=False)
        
        # Variáveis para o comparativo
        self.ultima_origem = None
        self.ultimo_destino = None
        self.ultimo_algoritmo = None
        
        self.criar_layout()
        self.desenhar_mapa_base()

    def criar_layout(self):
        self.frame_menu = tk.Frame(self.root, width=350, bg="#2c3e50")
        self.frame_menu.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_menu.pack_propagate(False)

        tk.Label(self.frame_menu, text="🗺️ Roteador OSMnx", fg="white", bg="#2c3e50", font=("Arial", 16, "bold")).pack(pady=15)

        # Botões de Rota (Dijkstra e A*)
        tk.Button(self.frame_menu, text="📍 Rota Dijkstra", font=("Arial", 11, "bold"), bg="#27ae60", fg="white", command=lambda: self.ativar_rota('DIJKSTRA')).pack(fill=tk.X, padx=20, pady=2)
        tk.Button(self.frame_menu, text="📍 Rota A*", font=("Arial", 11, "bold"), bg="#2980b9", fg="white", command=lambda: self.ativar_rota('A_STAR')).pack(fill=tk.X, padx=20, pady=2)
        
        # Botão de Comparativo
        self.btn_comparar = tk.Button(self.frame_menu, text="⚖️ Comparar Modelos", font=("Arial", 11, "bold"), bg="#8e44ad", fg="white", state=tk.DISABLED, command=self.abrir_comparacao)
        self.btn_comparar.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Checkbutton(self.frame_menu, text="Animar busca", variable=self.animar_var, bg="#2c3e50", fg="white", selectcolor="#34495e").pack(pady=5)

        tk.Frame(self.frame_menu, height=2, bg="#34495e").pack(fill=tk.X, padx=10, pady=10)
        
        ferramentas = [
            ('semaforo', '🚦 Add Semáforo (1)', '#f39c12'),
            ('lombada', '🏔️ Add Lombada (2)', '#d35400'),
            ('velocidade', '🏎️ Editar Vel. de Trecho (3)', '#8e44ad'),
            ('limpar', '🧹 Remover Edição (4)', '#c0392b')
        ]
        for modo, texto, cor in ferramentas:
            tk.Button(self.frame_menu, text=texto, bg=cor, fg="white", font=("Arial", 10), command=lambda m=modo: self.ativar_ferramenta(m)).pack(fill=tk.X, padx=20, pady=3)

        tk.Button(self.frame_menu, text="🗑️ Limpar TODAS Edições", bg="#7f8c8d", fg="white", command=self.limpar_tudo).pack(fill=tk.X, padx=20, pady=15)

        self.lbl_status = tk.Label(self.frame_menu, text="Aguardando comando...", fg="#f1c40f", bg="#2c3e50", font=("Arial", 11, "bold"), wraplength=300)
        self.lbl_status.pack(pady=10)

        self.console = tk.Text(self.frame_menu, height=20, bg="#1e272e", fg="white", font=("Consolas", 9), state=tk.DISABLED)
        self.console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_mapa = tk.Frame(self.root, bg="#FFFFFF")
        self.frame_mapa.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, msg + "\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def desenhar_mapa_base(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('#FFFFFF') 
        self.ax.set_facecolor('#FFFFFF')
        
        ox.plot_graph(self.motor.G, ax=self.ax, show=False, close=False, node_size=0, edge_color='#AAAAAA', edge_alpha=0.7, bgcolor='#FFFFFF')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_mapa)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', lambda e: self.ativar_ferramenta({'1':'semaforo','2':'lombada','3':'velocidade','4':'limpar'}.get(e.key)))
        self.plotar_marcadores()

    def plotar_marcadores(self):
        while len(self.ax.collections) > 1: self.ax.collections[-1].remove()
        
        def extrair_coords(nos): return zip(*[(self.motor.G.nodes[n]['x'], self.motor.G.nodes[n]['y']) for n in nos])
            
        if self.motor.edicoes['semaforos']:
            sx, sy = extrair_coords(self.motor.edicoes['semaforos'])
            self.ax.scatter(sx, sy, c='red', s=60, marker='s', zorder=5, edgecolors='black')
        if self.motor.edicoes['lombadas']:
            lx, ly = extrair_coords(self.motor.edicoes['lombadas'])
            self.ax.scatter(lx, ly, c='orange', s=60, marker='^', zorder=5, edgecolors='black')

        nos_vel = set()
        for (u, v, k) in self.motor.edicoes['velocidades']: nos_vel.update([u, v])
        if nos_vel:
            vx, vy = extrair_coords(nos_vel)
            self.ax.scatter(vx, vy, c='lime', s=40, marker='d', zorder=5, edgecolors='black')
            
        self.canvas.draw_idle()

    def ativar_rota(self, algoritmo):
        self.modo_atual = f'ROTA_{algoritmo}'
        self.origem = self.destino = self.ponto_edicao = None
        self.lbl_status.config(text=f"📍 Rota {algoritmo}: Clique na ORIGEM")
        self.log(f"Modo {algoritmo} ativado. Selecione a origem.")
        self.ax.clear()
        ox.plot_graph(self.motor.G, ax=self.ax, show=False, close=False, node_size=0, edge_color='#AAAAAA', edge_alpha=0.7, bgcolor='#FFFFFF')
        self.plotar_marcadores()

    def ativar_ferramenta(self, modo):
        if not modo: return
        self.modo_atual = modo
        self.ponto_edicao = None 
        textos = {'semaforo': '🚦 Inserir Semáforo', 'lombada': '🏔️ Inserir Lombada', 'velocidade': '🏎️ Editar Vel.', 'limpar': '🧹 Apagar'}
        self.lbl_status.config(text=f"Ferramenta: {textos[modo]}\nClique no mapa.")
        self.plotar_marcadores()

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1 or not self.modo_atual: return
        
        no_prox = ox.distance.nearest_nodes(self.motor.G, event.xdata, event.ydata)
        nx_coord, ny_coord = self.motor.G.nodes[no_prox]['x'], self.motor.G.nodes[no_prox]['y']
        
        if self.modo_atual.startswith('ROTA_'):
            if not self.origem:
                self.origem = no_prox
                self.ax.scatter([nx_coord], [ny_coord], c='green', s=150, zorder=10, edgecolors='black')
                self.canvas.draw()
                self.lbl_status.config(text="📍 Origem definida! Clique no DESTINO.")
            elif not self.destino:
                self.destino = no_prox
                self.ax.scatter([nx_coord], [ny_coord], c='red', s=150, zorder=10, edgecolors='black')
                self.canvas.draw()
                
                algoritmo_escolhido = self.modo_atual.split('_', 1)[1] # Extrai 'DIJKSTRA' ou 'A_STAR'
                self.lbl_status.config(text=f"⚙️ Calculando {algoritmo_escolhido}...")
                self.root.update()
                self.executar_rota(algoritmo_escolhido)
                
        elif self.modo_atual == 'velocidade':
            if not self.ponto_edicao:
                self.ponto_edicao = no_prox
                self.ax.scatter([nx_coord], [ny_coord], c='blue', s=100, zorder=10, edgecolors='black')
                self.canvas.draw()
                self.lbl_status.config(text="🏎️ Clique no PONTO FINAL.")
            else:
                self.ax.scatter([nx_coord], [ny_coord], c='blue', s=100, zorder=10, edgecolors='black')
                self.canvas.draw()
                self.root.update()
                
                nova_vel = simpledialog.askfloat("Velocidade", "Qual a nova velocidade (km/h)?", parent=self.root)
                if nova_vel:
                    vias = self.motor.alterar_velocidade_trecho(self.ponto_edicao, no_prox, nova_vel)
                    if vias > 0: self.log(f"✅ Velocidade {nova_vel}km/h aplicada em {vias} vias.")
                    else: self.log("❌ Caminho inválido (verifique a mão da rua).")
                
                self.ponto_edicao = None
                self.lbl_status.config(text="Ferramenta: 🏎️ Editar Vel.\nClique no PONTO INICIAL.")
                self.plotar_marcadores() 

        else:
            u, v, key = ox.distance.nearest_edges(self.motor.G, event.xdata, event.ydata)
            self.aplicar_edicao_simples(no_prox, u, v, key)

    def aplicar_edicao_simples(self, no, u, v, key):
        if self.modo_atual == 'semaforo':
            self.motor.edicoes['semaforos'].add(no)
            self.log("🚦 Semáforo adicionado.")
        elif self.modo_atual == 'lombada':
            self.motor.edicoes['lombadas'].add(no)
            self.log("🏔️ Lombada adicionada.")
        elif self.modo_atual == 'limpar':
            self.motor.edicoes['semaforos'].discard(no)
            self.motor.edicoes['lombadas'].discard(no)
            self.motor.edicoes['removidos'].add(no)
            if (u, v, key) in self.motor.edicoes['velocidades']:
                del self.motor.edicoes['velocidades'][(u, v, key)]
            self.log("🧹 Edições removidas do local.")

        self.motor.atualizar_pesos()
        self.motor.salvar_edicoes()
        self.plotar_marcadores()

    def executar_rota(self, algoritmo):
        inicio_t = time.time()
        
        if algoritmo == 'DIJKSTRA':
            caminho, visitados = self.motor.dijkstra(self.origem, self.destino)
        else: # A_STAR
            caminho, visitados = self.motor.a_star(self.origem, self.destino)
            
        if not caminho:
            self.log("❌ Rota não encontrada!")
            self.modo_atual = None
            return

        if self.animar_var.get():
            coords = [(self.motor.G.nodes[n]['x'], self.motor.G.nodes[n]['y']) for n in visitados]
            scat = self.ax.scatter([], [], c='cyan', s=10, zorder=3)
            passo = max(1, len(coords) // 50)
            for i in range(1, len(coords) + passo, passo):
                scat.set_offsets(coords[:min(i, len(coords))])
                self.canvas.draw()
                self.root.update()
                time.sleep(0.01)
        
        ox.plot_graph_route(self.motor.G, caminho, ax=self.ax, route_color='#00AA55', route_linewidth=4, node_size=0, show=False, close=False)
        self.canvas.draw()

        stats = self.motor.calcular_estatisticas(caminho)
        self.log(f"\n--- ROTA {algoritmo} ({time.time() - inicio_t:.2f}s) ---")
        self.log(f"Distância: {stats['dist_km']:.2f} km")
        self.log(f"Tempo: {stats['tempo_min']:.1f} min")
        self.log(f"Semáforos: {stats['semaforos']} | Lombadas: {stats['lombadas']}")
        
        # Salva dados e habilita o botão de comparação
        self.ultima_origem = self.origem
        self.ultimo_destino = self.destino
        self.ultimo_algoritmo = algoritmo
        self.btn_comparar.config(state=tk.NORMAL)
        
        self.modo_atual = None
        self.lbl_status.config(text="✅ Rota concluída.")

    def abrir_comparacao(self):
        if not self.ultima_origem or not self.ultimo_destino: return

        # Descobre qual é o outro algoritmo
        outro_algoritmo = 'A_STAR' if self.ultimo_algoritmo == 'DIJKSTRA' else 'DIJKSTRA'
        self.lbl_status.config(text=f"⚖️ Gerando comparativo com {outro_algoritmo}...")
        self.root.update()

        # Executa o algoritmo comparativo e refaz o original para pegar os visitados
        inicio_t = time.time()
        if outro_algoritmo == 'DIJKSTRA':
            caminho_outro, visitados_outro = self.motor.dijkstra(self.ultima_origem, self.ultimo_destino)
            caminho_orig, visitados_orig = self.motor.a_star(self.ultima_origem, self.ultimo_destino)
        else:
            caminho_outro, visitados_outro = self.motor.a_star(self.ultima_origem, self.ultimo_destino)
            caminho_orig, visitados_orig = self.motor.dijkstra(self.ultima_origem, self.ultimo_destino)
            
        tempo_exec_outro = time.time() - inicio_t

        if not caminho_outro:
            self.log("❌ Rota não encontrada para o modelo comparativo.")
            self.lbl_status.config(text="Erro no comparativo.")
            return

        # Log no terminal
        self.log(f"\n--- COMPARAÇÃO: {outro_algoritmo} ({tempo_exec_outro:.2f}s) ---")
        self.log(f"Nós explorados na busca: {len(visitados_outro)}")

        # Cria a nova janela Toplevel
        janela_comp = tk.Toplevel(self.root)
        janela_comp.title(f"Comparativo: {self.ultimo_algoritmo} vs {outro_algoritmo}")
        janela_comp.geometry("1200x600")

        fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig_comp.patch.set_facecolor('#FFFFFF')

        # Plota os mapas base 
        ox.plot_graph(self.motor.G, ax=ax1, show=False, close=False, node_size=0, edge_color='#AAAAAA', edge_alpha=0.4, bgcolor='#FFFFFF')
        ox.plot_graph(self.motor.G, ax=ax2, show=False, close=False, node_size=0, edge_color='#AAAAAA', edge_alpha=0.4, bgcolor='#FFFFFF')

        # Plota as duas rotas
        ox.plot_graph_route(self.motor.G, caminho_orig, ax=ax1, route_color='#27ae60', route_linewidth=3, node_size=0, show=False, close=False)
        ax1.set_title(f"Original: {self.ultimo_algoritmo}\nNós explorados: {len(visitados_orig)}", fontsize=12)

        ox.plot_graph_route(self.motor.G, caminho_outro, ax=ax2, route_color='#2980b9', route_linewidth=3, node_size=0, show=False, close=False)
        ax2.set_title(f"Comparativo: {outro_algoritmo}\nNós explorados: {len(visitados_outro)}", fontsize=12)

        # Renderiza no Tkinter
        canvas_comp = FigureCanvasTkAgg(fig_comp, master=janela_comp)
        canvas_comp.draw()
        canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.lbl_status.config(text="✅ Comparativo gerado com sucesso.")

    def limpar_tudo(self):
        if messagebox.askyesno("Confirmar", "Apagar TODAS as edições?"):
            for k in self.motor.edicoes: self.motor.edicoes[k].clear()
            self.motor.atualizar_pesos()
            self.motor.salvar_edicoes()
            self.log("🧹 Tudo limpo.")
            self.ax.clear()
            ox.plot_graph(self.motor.G, ax=self.ax, show=False, close=False, node_size=0, edge_color='#AAAAAA', edge_alpha=0.7, bgcolor='#FFFFFF')
            self.plotar_marcadores()

# ==========================================
# INICIALIZAÇÃO
# ==========================================
if __name__ == "__main__":
    motor = MotorRoteamento()
    root = tk.Tk()
    app = InterfaceMapa(root, motor)
    root.mainloop()
