!apt-get install -y -qq glpk-utils
!apt-get install -y -qq coinor-cbc
!pip install osmnx folium matplotlib networkx

import math, time, random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from heapq import heappush, heappop
try:
    import pyomo.environ as pyo
except ImportError:
    pyo = None
    print("Pyomo não instalado (rode o pip install).")
plt.rcParams["figure.figsize"] = (10, 6)

from collections import defaultdict
import random, time

def path_edges(path):
    return {(path[k], path[k+1]) for k in range(len(path) - 1)}

# Instâncias

@dataclass
class VRPInstanceMD:
    n: int                # nº clientes
    m: int                # nº garagens (depots)
    K: int                # nº veículos (total)
    Q: int                # capacidade (se for igual p/ todos)
    coords: np.ndarray    # array (m + n + 1, 2) = depots | clients | sink
    dist: np.ndarray      # matriz de distâncias (N x N)
    depots: list          # lista de índices dos depots
    clients: list         # lista de índices dos clientes
    sink: int             # índice do destino final

def make_instance_md(n=20, m=3, K=8, Q=6, seed=42, sink_center=True):
    """
    Índices:
      depots:  0 .. m-1
      clients: m .. m+n-1
      sink:    m+n
    """
    rng = np.random.default_rng(seed)

    depots = rng.uniform(0, 100, size=(m, 2))
    clients = rng.uniform(0, 100, size=(n, 2))
    if sink_center:
        sink = np.array([[50.0, 50.0]])
    else:
        sink = rng.uniform(0, 100, size=(1, 2))

    coords = np.vstack([depots, clients, sink])

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    depots_idx  = list(range(m))
    clients_idx = list(range(m, m + n))
    sink_idx    = m + n

    return VRPInstanceMD(
        n=n, m=m, K=K, Q=Q,
        coords=coords, dist=dist,
        depots=depots_idx, clients=clients_idx, sink=sink_idx
    )

def plot_routes_md(inst: VRPInstanceMD, routes, title="Rotas (Multi-Depot → Sink)"):
    """
    routes: lista de rotas, cada rota é uma lista de índices no grafo:
      exemplo: [depot, c1, c2, ..., sink]
    """
    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(routes))))

    # plota arestas das rotas
    for idx, r in enumerate(routes):
        if len(r) < 2:
            continue
        r_coords = inst.coords[r]

        plt.plot(
            r_coords[:, 0], r_coords[:, 1],
            c=colors[idx], linewidth=2, label=f"R{idx+1}", zorder=1
        )

        if len(r_coords) >= 2:
            mid = max(0, (len(r_coords) - 2) // 2)
            dx = (r_coords[mid+1, 0] - r_coords[mid, 0]) * 0.01
            dy = (r_coords[mid+1, 1] - r_coords[mid, 1]) * 0.01
            plt.arrow(
                r_coords[mid, 0], r_coords[mid, 1], dx, dy,
                color=colors[idx], head_width=2, length_includes_head=True
            )

    clients_xy = inst.coords[inst.clients]
    plt.scatter(clients_xy[:, 0], clients_xy[:, 1], s=90, zorder=2)
    for j, node in enumerate(inst.clients, start=1):
        x, y = inst.coords[node]
        plt.text(x+1, y+1, str(j), fontsize=9, fontweight="bold")

    depots_xy = inst.coords[inst.depots]
    plt.scatter(depots_xy[:, 0], depots_xy[:, 1], marker="s", s=140, zorder=3, label="Garagens")
    for d in inst.depots:
        x, y = inst.coords[d]
        plt.text(x+1, y+1, f"D{d}", fontsize=9, fontweight="bold")

    sx, sy = inst.coords[inst.sink]
    plt.scatter([sx], [sy], marker="*", s=220, zorder=4, label="Destino (sink)")
    plt.text(sx+1, sy+1, "S", fontsize=10, fontweight="bold")

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def get_solver(max_seconds=300):
    """
    Para o B&P, queremos solver LP/MIP disponível.
    No Colab: CBC e GLPK normalmente ok.
    """
    if pyo is None:
        return None, None

    opt = pyo.SolverFactory("cbc")
    if opt.available():
        opt.options["seconds"] = max_seconds
        return "cbc", opt

    opt = pyo.SolverFactory("glpk")
    if opt.available():
        opt.options["tmlim"] = max_seconds
        return "glpk", opt

    return None, None

# CRIAÇÃO DA INSTÂNCIA
inst = make_instance_md(n=30, m=3, K=8, Q=30, seed=42, sink_center=True)

# plota só os pontos 
plot_routes_md(inst, [], "Instância MD-VRP (Garagens → Escola)")

"""# Branch-and-Price (heurístico de geração de colunas)
Master relaxado com colunas (rotas garagem → clientes → destino); duais (π, μ);
custo reduzido c̄_r = c_r − ∑_i π_i a_{ir} − μ.
Sementes longas; fallback sem limite de veículos durante a fase LP.

# Formulação para Branch-and-Price (Set Partitioning)

O Branch-and-Price utiliza a decomposição de Dantzig–Wolfe, na qual o problema
mestre seleciona rotas completas previamente geradas.

Modelo Matemático — Branch-and-Price (Set Partitioning)

O problema é decomposto em um Problema Mestre (RMP) e um Subproblema
(Pricing).

# Problema Mestre (Set Partitioning)

Seja Ω o conjunto de todas as rotas viáveis do tipo
garagem → clientes → destino final (sink).

Variáveis:

$$\lambda_r = \begin{cases} 1 & \text{se a rota } r \in \Omega \text{ é selecionada} \\ 0 & \text{caso contrário} \end{cases}$$

Formulação:

$$\min \sum_{r \in \Omega} c_r \lambda_r$$

Sujeito a:

$$\sum_{r \in \Omega} a_{ir} \lambda_r = 1 \quad \forall i \in \mathcal{C}$$

$$\sum_{r \in \Omega} \lambda_r \le K$$

$$\lambda_r \in \{0, 1\}$$

Onde:

*   $\mathcal{C}$ é o conjunto de clientes,
*   $a_{ir} = 1$ se a rota $r$ atende o cliente $i$, e $0$ caso contrário,
*   $c_r$ é o custo total da rota $r$,
*   $K$ é o número máximo de veículos disponíveis.

## Pricing Problem (Subproblema)

O objetivo do subproblema é identificar uma nova rota r′ com custo reduzido
negativo. Esse problema é tipicamente modelado como um Elementary Shortest
Path Problem with Resource Constraints (ESPPRC).

#### Função Objetivo: Minimizar o Custo Reduzido (c̄)

$$
\min \ \bar{c}
= \sum_{(i,j) \in A} c_{ij} x_{ij}
- \sum_{i \in \mathcal{C}} \pi_i \left( \sum_{j \in V} x_{ij} \right)
- \sigma
$$

onde o custo reduzido é composto pelo custo real da rota menos os valores duais
$\pi_i$ associados à cobertura dos clientes visitados e menos o dual $\sigma$
da restrição de frota total.

#### Sujeito a:

**Fluxo na Garagem (Origem):**  
O veículo deve sair exatamente uma vez de uma garagem.

$$
\sum_{j \in V \setminus \mathcal{D}} x_{dj} = 1,
\quad \forall d \in \mathcal{D}
$$

**Fluxo no Destino Final:**  
O veículo deve chegar exatamente uma vez ao destino final.

$$
\sum_{i \in V \setminus \{s\}} x_{is} = 1
$$

**Conservação de Fluxo (Clientes):**  
Se o veículo entra em um cliente $k$, ele deve sair de $k$.

$$
\sum_{i \in V} x_{ik} - \sum_{j \in V} x_{kj} = 0,
\quad \forall k \in \mathcal{C}
$$

**Elementaridade (Visitar no máximo uma vez):**  
Cada cliente pode ser visitado no máximo uma vez na mesma rota.

$$
\sum_{i \in V} x_{ik} \le 1,
\quad \forall k \in \mathcal{C}
$$

**Capacidade e Eliminação de Subciclos:**  
As restrições abaixo garantem que a capacidade do veículo não seja violada e
impedem a formação de subciclos desconectados.

$$
u_j \ge u_i + q_j - M(1 - x_{ij}),
\quad \forall (i,j) \in A, \ j \in \mathcal{C}
$$

$$
q_i \le u_i \le Q,
\quad \forall i \in \mathcal{C}
$$

onde $u_i$ representa a carga acumulada ao visitar o nó $i$, $q_i$ é a demanda
do cliente $i$, $Q$ é a capacidade do veículo e $M$ é uma constante grande
(suficientemente grande, tipicamente $M = Q$).

**Restrições de Domínio:**

$$
x_{ij} \in \{0,1\}, \quad \forall (i,j) \in A
$$

$$
u_i \ge 0
$$
"""

def run_grasp_vrp_md(inst, iterations=50, alpha=0.2, respect_K=False, use_2opt=True):
    """
    Warm-start (pool inicial) via GRASP para instância Multi-Depot -> Sink.

    Rotas geradas:
      - path: [depot, ..., sink]
      - covered: clientes atendidos
      - edges: conjunto de arcos (i,j)
      - load: nº de clientes na rota (demanda 1)
      - cost: soma das distâncias
      - depot: garagem inicial
    """
    print(f"--- Warm-Up GRASP MD ({iterations} iterações, alpha={alpha}) ---")

    generated_routes = []
    unique_paths = set()

    depots = list(inst.depots)
    clients_all = list(inst.clients)
    clients_set = set(clients_all)
    sink = inst.sink

    for _ in range(iterations):
        unvisited = set(clients_all)
        num_routes = 0

        while unvisited:
            num_routes += 1
            if respect_K and num_routes > inst.K:
                break

            depot = random.choice(depots)
            path = [depot]
            curr = depot
            load = 0  # demanda=1 por cliente

            while True:
                if load + 1 > inst.Q:
                    break

                candidates = [(inst.dist[curr, node], node) for node in unvisited]
                if not candidates:
                    break

                candidates.sort(key=lambda x: x[0])
                min_dist = candidates[0][0]
                threshold = min_dist * (1 + alpha)
                rcl = [c for c in candidates if c[0] <= threshold]
                _, chosen_node = random.choice(rcl)

                path.append(chosen_node)
                load += 1
                curr = chosen_node
                unvisited.remove(chosen_node)

            # fecha no sink
            path.append(sink)

            # 2-opt aberto (mantém depot e sink)
            final_path = improve_route_2opt_open(path, inst.dist) if use_2opt else path

            covered = set(final_path).intersection(clients_set)
            if len(covered) == 0:
                continue

            path_tuple = tuple(final_path)
            if path_tuple in unique_paths:
                continue
            unique_paths.add(path_tuple)

            cost = sum(inst.dist[final_path[k], final_path[k + 1]] for k in range(len(final_path) - 1))
            edges = path_edges(final_path)

            route_dict = {
                "path": list(final_path),
                "cost": float(cost),
                "edges": edges,
                "covered": covered,
                "load": int(len(covered)),
                "depot": int(final_path[0]),
            }
            generated_routes.append(route_dict)

    print(f"   -> GRASP gerou {len(generated_routes)} colunas únicas.")
    return generated_routes

def improve_route_2opt_open(path, dist_matrix):
    """
    Aplica 2-opt em um caminho ABERTO [start, ..., end],
    mantendo start e end fixos (ex.: [depot, ..., sink]).
    """
    if len(path) < 4:
        return path  # precisa ter pelo menos 2 arestas internas

    best_path = path[:]
    improved = True

    while improved:
        improved = False

        # Não mexe no primeiro (0) nem no último (-1)
        # então i começa em 1 e j vai até len-3
        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                if j - i == 1:
                    continue  # segmento muito curto, não muda nada

                u, v = best_path[i - 1], best_path[i]
                x, y = best_path[j], best_path[j + 1]

                current_cost = dist_matrix[u, v] + dist_matrix[x, y]
                new_cost     = dist_matrix[u, x] + dist_matrix[v, y]

                if new_cost < current_cost - 1e-4:
                    # inverte o segmento i..j
                    best_path[i:j + 1] = best_path[i:j + 1][::-1]
                    improved = True

    return best_path

def solve_exact_pricing_md(inst, pi_cover, pi_fleet, forbidden_arcs, forced_arcs, Q, time_limit=120):
    """
    Pricing exato (MIP) para rota ABERTA: depot -> ...clientes... -> sink
    - pi_cover: dict {cliente_node: dual}
    - pi_fleet: dual da restrição de frota do mestre
    - forbidden_arcs / forced_arcs: sets de arcos (i,j)
    - Q: capacidade (demanda=1 por cliente)
    """
    if pyo is None:
        return None, 0.0

    depots = list(inst.depots)
    clients = list(inst.clients)
    sink = inst.sink
    nodes = depots + clients + [sink]

    forced_out = {}
    forced_in = {}
    for (u, v) in forced_arcs:
        forced_out[u] = v
        forced_in[v] = u

    valid_arcs = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if i == sink:
                continue          # ninguém sai do sink
            if j in depots:
                continue          # ninguém entra em depot
            if (i, j) in forbidden_arcs:
                continue
            if i in forced_out and forced_out[i] != j:
                continue
            if j in forced_in and forced_in[j] != i:
                continue
            valid_arcs.append((i, j))

    m = pyo.ConcreteModel()
    m.A = pyo.Set(initialize=valid_arcs, dimen=2)
    m.x = pyo.Var(m.A, within=pyo.Binary)
    m.u = pyo.Var(clients, within=pyo.NonNegativeReals, bounds=(0, Q))

    # objetivo: custo reduzido
    def red_cost_rule(model):
        real = sum(inst.dist[i, j] * model.x[i, j] for (i, j) in model.A)

        gains = 0.0
        for c in clients:
            dual = pi_cover.get(c, 0.0)
            incoming_c = sum(model.x[i, c] for i in nodes if (i, c) in model.A)
            gains += dual * incoming_c

        return real - gains - float(pi_fleet)

    m.obj = pyo.Objective(rule=red_cost_rule, sense=pyo.minimize)
    m.cons = pyo.ConstraintList()

    # 1 saída total de depots
    m.cons.add(
        sum(m.x[d, j] for d in depots for j in (clients + [sink]) if (d, j) in m.A) == 1
    )

    # 1 entrada no sink
    m.cons.add(
        sum(m.x[i, sink] for i in (depots + clients) if (i, sink) in m.A) == 1
    )

    # conservação + elementaridade em clientes
    for k in clients:
        incoming = sum(m.x[i, k] for i in depots + clients if (i, k) in m.A)
        outgoing = sum(m.x[k, j] for j in clients + [sink] if (k, j) in m.A)
        m.cons.add(incoming == outgoing)
        m.cons.add(incoming <= 1)

    total_visited = sum(
        sum(m.x[i, c] for i in depots + clients if (i, c) in m.A)
        for c in clients
    )
    m.cons.add(total_visited >= 1)

    # MTZ / capacidade (demanda=1)
    for (i, j) in valid_arcs:
        if j in clients:
            if i in clients:
                m.cons.add(m.u[j] >= m.u[i] + 1 - Q * (1 - m.x[i, j]))
            elif i in depots:
                m.cons.add(m.u[j] >= 1 - Q * (1 - m.x[i, j]))

    for c in clients:
        inc = sum(m.x[i, c] for i in depots + clients if (i, c) in m.A)
        m.cons.add(m.u[c] >= inc)
        m.cons.add(m.u[c] <= Q * inc)

    _, solver = get_solver(max_seconds=time_limit)
    if not solver:
        return None, 0.0

    solver.solve(m, tee=False)
    min_rc_val = pyo.value(m.obj)

    if min_rc_val is not None and min_rc_val < -1e-5:
        # reconstrói caminho
        start = None
        for d in depots:
            for j in clients + [sink]:
                if (d, j) in valid_arcs and pyo.value(m.x[d, j]) > 0.5:
                    start = d
                    break
            if start is not None:
                break
        if start is None:
            return None, float(min_rc_val)

        path = [start]
        curr = start
        used = set()

        while curr != sink:
            nxt = None
            for j in clients + [sink]:
                if (curr, j) in valid_arcs and pyo.value(m.x[curr, j]) > 0.5:
                    nxt = j
                    break
            if nxt is None:
                break
            if (curr, nxt) in used:
                break
            used.add((curr, nxt))
            path.append(nxt)
            curr = nxt

        if path[-1] != sink:
            return None, float(min_rc_val)

        covered = set(path).intersection(set(clients))
        if len(covered) == 0:
            return None, float(min_rc_val)

        real_cost = sum(inst.dist[path[k], path[k + 1]] for k in range(len(path) - 1))
        edges = path_edges(path)

        col = {
            "path": path,
            "cost": float(real_cost),
            "edges": edges,
            "covered": covered,
            "load": int(len(covered)),
            "depot": int(path[0]),
        }
        return col, float(min_rc_val)

    return None, float(min_rc_val if min_rc_val is not None else 0.0)

class BPNode:
    def __init__(self, nid, parent=None, forbidden=None, forced=None):
        self.id = nid
        self.parent = parent

        self.forbidden = set(forbidden) if forbidden is not None else set()
        self.forced = set(forced) if forced is not None else set()

        self.lb = -float("inf")     # lower bound do nó (valor do RMP relaxado)
        self.lam_sol = {}           # solução do mestre: {route_id: lambda_value}
        self.x_bar = {}             # arcos agregados derivados do lambda: {(i,j): val}

    def __lt__(self, other):
        return self.lb < other.lb

class BPNode:
    def __init__(self, nid, parent=None, forbidden=None, forced=None):
        self.id = nid
        self.parent = parent
        self.forbidden = set(forbidden) if forbidden is not None else set()
        self.forced = set(forced) if forced is not None else set()
        self.lb = float("inf")
        self.x_sol = defaultdict(float)

    def __lt__(self, other):
        return self.lb < other.lb

def solve_node_md(
    inst,
    node,
    global_routes,
    alpha=0.5,
    gap_tol=0.005,
    time_limit=900,
    max_cg_iter=400,
    BIG=1e6,
    M_PENALTY=100000.0,
    max_new_cols_per_iter=30,
    max_seeds=40,
):
    if pyo is None:
        return False

    start_time = time.time()

    depots = list(inst.depots)
    clients = list(inst.clients)
    clients_set = set(clients)
    sink = inst.sink
    all_nodes = depots + clients + [sink]

    # matriz para heurística respeitar branching
    dist_h = inst.dist.copy()

    for (u, v) in node.forbidden:
        dist_h[u, v] = BIG

    for (u, v) in node.forced:
        for k in all_nodes:
            if k != v:
                dist_h[u, k] = BIG
            if k != u:
                dist_h[k, v] = BIG

    # estabilização de duais
    pi_cov_stab = {c: 0.0 for c in clients}
    pi_flt_stab = 0.0
    first_iter = True

    for iter_cg in range(1, max_cg_iter + 1):
        if (time.time() - start_time) > time_limit:
            break

        # RMP
        rmp = pyo.ConcreteModel()

        valid_idx = []
        for idx, r in enumerate(global_routes):
            if not r["edges"].isdisjoint(node.forbidden):
                continue
            if not node.forced.issubset(r["edges"]):
                continue
            valid_idx.append(idx)

        if len(valid_idx) == 0:
            node.lb = float("inf")
            return True

        rmp.I = pyo.Set(initialize=valid_idx)
        rmp.lam = pyo.Var(rmp.I, within=pyo.NonNegativeReals)

        rmp.slk_cov = pyo.Var(clients, within=pyo.NonNegativeReals)  # penalizado

        rmp.s_flt = pyo.Var(within=pyo.NonNegativeReals)

        rmp.obj_pen = pyo.Objective(
            expr=(
                sum(global_routes[i]["cost"] * rmp.lam[i] for i in rmp.I)
                + sum(M_PENALTY * rmp.slk_cov[c] for c in clients)
            ),
            sense=pyo.minimize
        )

        rmp.c_cov = pyo.ConstraintList()
        for c in clients:
            rmp.c_cov.add(
                sum(rmp.lam[i] for i in rmp.I if c in global_routes[i]["covered"])
                + rmp.slk_cov[c]
                == 1
            )

        # <= K via slack: sum(lam) + s_flt == K  (s_flt >= 0)
        rmp.c_flt = pyo.Constraint(expr=sum(rmp.lam[i] for i in rmp.I) + rmp.s_flt == inst.K)

        rmp.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        _, solver = get_solver(max_seconds=60)
        if not solver:
            node.lb = float("inf")
            return True

        res = solver.solve(rmp, tee=False)
        if res.solver.termination_condition != pyo.TerminationCondition.optimal:
            node.lb = float("inf")
            return True

        real_obj = float(sum(global_routes[i]["cost"] * pyo.value(rmp.lam[i]) for i in valid_idx))
        slack_cov_sum = float(sum(pyo.value(rmp.slk_cov[c]) for c in clients))
        if slack_cov_sum > 1e-6:
            node.lb = float("inf")
            return True

        # duais
        pi_cov_curr = {}
        for idx_c, c in enumerate(clients, start=1):
            pi_cov_curr[c] = float(rmp.dual[rmp.c_cov[idx_c]])
        pi_flt_curr = float(rmp.dual[rmp.c_flt])

        print(f"Node {node.id} / CG {iter_cg}: LB(real)={real_obj:.2f}")

        # estabilização
        if first_iter:
            pi_cov_stab = pi_cov_curr.copy()
            pi_flt_stab = pi_flt_curr
            first_iter = False
        else:
            for c in clients:
                pi_cov_stab[c] = alpha * pi_cov_curr[c] + (1 - alpha) * pi_cov_stab[c]
            pi_flt_stab = alpha * pi_flt_curr + (1 - alpha) * pi_flt_stab

        pi_cov_use = pi_cov_stab
        pi_flt_use = pi_flt_stab

        # Pricing heurístico
        
        new_col = False
        added = 0

        seeds = clients[:]
        random.shuffle(seeds)
        seeds = seeds[:max_seeds]

        for first_client in seeds:
            if added >= max_new_cols_per_iter:
                break

            depot = random.choice(depots)
            if dist_h[depot, first_client] > BIG / 2:
                continue

            path = [depot, first_client]
            vis = {first_client}
            load = 1
            cost_h = dist_h[depot, first_client]
            curr = first_client

            if load > inst.Q:
                continue

            while load < inst.Q:
                close_c = dist_h[curr, sink]
                if close_c < BIG / 2:
                    rc_path = (cost_h + close_c) - sum(pi_cov_use[x] for x in vis) - pi_flt_use
                    if rc_path < -1e-3:
                        raw = path + [sink]
                        opt_p = improve_route_2opt_open(raw, dist_h)

                        covered = set(opt_p).intersection(clients_set)
                        if len(covered) == 0:
                            pass
                        else:
                            edges_opt = path_edges(opt_p)
                            if edges_opt.isdisjoint(node.forbidden) and node.forced.issubset(edges_opt):
                                real_cost = sum(inst.dist[opt_p[k], opt_p[k+1]] for k in range(len(opt_p) - 1))
                                global_routes.append({
                                    "path": opt_p,
                                    "cost": float(real_cost),
                                    "edges": edges_opt,
                                    "covered": covered,
                                    "load": int(len(covered)),
                                    "depot": int(opt_p[0]),
                                })
                                new_col = True
                                added += 1

                # próximo por dist - dual (simples)
                bst_n = None
                bst_val = float("inf")
                for nxt in clients:
                    if nxt not in vis and dist_h[curr, nxt] < BIG / 2:
                        val = dist_h[curr, nxt] - pi_cov_use.get(nxt, 0.0)
                        if val < bst_val:
                            bst_val = val
                            bst_n = nxt

                if bst_n is None:
                    break

                path.append(bst_n)
                vis.add(bst_n)
                cost_h += dist_h[curr, bst_n]
                curr = bst_n
                load += 1

        # Pricing exato 
        
        if not new_col:
            ex_col, min_rc = solve_exact_pricing_md(
                inst,
                pi_cover=pi_cov_curr,
                pi_fleet=pi_flt_curr,
                forbidden_arcs=node.forbidden,
                forced_arcs=node.forced,
                Q=inst.Q,
                time_limit=120
            )

            if ex_col is not None:
                global_routes.append(ex_col)
                new_col = True

            if min_rc > -1e-6 and ex_col is None:
                # nada a adicionar -> encerra CG
                break

        # se não adicionou nada, encerra
        if not new_col:
            break

    # salva LB e x_bar agregado
    node.lb = real_obj

    node.x_sol = defaultdict(float)
    for i in valid_idx:
        val = float(pyo.value(rmp.lam[i]))
        if val > 1e-7:
            p = global_routes[i]["path"]
            for k in range(len(p) - 1):
                node.x_sol[(p[k], p[k+1])] += val

    return True

def solve_full_branch_and_price_md(inst, alpha=0.5, gap_tol=0.01, time_limit=900, max_cg_iter=400):
    if pyo is None:
        return float("inf"), []

    start_time = time.time()

    depots = list(inst.depots)
    clients = list(inst.clients)
    sink = inst.sink

    global_routes = []

    # 1) colunas pendulares iniciais
    for c in clients:
        best_d = None
        best_cost = float("inf")
        for d in depots:
            val = inst.dist[d, c] + inst.dist[c, sink]
            if val < best_cost:
                best_cost = val
                best_d = d

        path = [best_d, c, sink]
        edges = path_edges(path)
        global_routes.append({
            "path": path,
            "cost": float(best_cost),
            "edges": edges,
            "covered": {c},
            "load": 1,
            "depot": int(best_d),
        })

    # 2) warm-up GRASP
    global_routes.extend(run_grasp_vrp_md(inst, iterations=60, alpha=0.2, respect_K=False, use_2opt=True))

    # árvore
    root = BPNode(0, parent=-1, forbidden=set(), forced=set())
    pq = []

    solve_node_md(inst, root, global_routes, alpha=alpha, gap_tol=gap_tol, time_limit=time_limit, max_cg_iter=max_cg_iter)
    heappush(pq, root)

    best_int_val = float("inf")
    cnt = 0

    while pq and (time.time() - start_time) < time_limit:
        node = heappop(pq)

        if node.lb >= best_int_val or node.lb == float("inf"):
            continue

        print(f"\nExplorando Node {node.id}: LB={node.lb:.2f} / bestInt={best_int_val:.2f}")

        # arco fracionário mais perto de 0.5
        frac_arc = None
        closest = float("inf")
        for (u, v), val in node.x_sol.items():
            if abs(val - round(val)) <= 1e-3:
                continue
            if u == sink:
                continue
            if v in depots:
                continue

            d = abs(val - 0.5)
            if d < closest:
                closest = d
                frac_arc = (u, v)

        if frac_arc is None:
            # nó “quase inteiro” no agregado de arcos
            best_int_val = min(best_int_val, node.lb)
            print(f"  -> Nó sem arco fracionário. bestInt={best_int_val:.2f}")
            continue

        u, v = frac_arc
        print(f"  -> Branching no arco ({u},{v})")

        # Filho 0: proíbe
        cnt += 1
        child0 = BPNode(cnt, parent=node.id, forbidden=node.forbidden | {(u, v)}, forced=node.forced)
        solve_node_md(inst, child0, global_routes, alpha=alpha, gap_tol=gap_tol, time_limit=time_limit, max_cg_iter=max_cg_iter)
        if child0.lb < best_int_val:
            heappush(pq, child0)

        # Filho 1: força
        cnt += 1
        child1 = BPNode(cnt, parent=node.id, forbidden=node.forbidden, forced=node.forced | {(u, v)})
        solve_node_md(inst, child1, global_routes, alpha=alpha, gap_tol=gap_tol, time_limit=time_limit, max_cg_iter=max_cg_iter)
        if child1.lb < best_int_val:
            heappush(pq, child1)

    # IP Final com frota <= K
    
    print("\n--- IP Final (Set Partitioning) ---")

    rmp_f = pyo.ConcreteModel()
    I = list(range(len(global_routes)))

    rmp_f.I = pyo.Set(initialize=I)
    rmp_f.lam = pyo.Var(rmp_f.I, within=pyo.Binary)

    rmp_f.s_flt = pyo.Var(within=pyo.NonNegativeReals)

    rmp_f.obj = pyo.Objective(
        expr=sum(global_routes[i]["cost"] * rmp_f.lam[i] for i in rmp_f.I),
        sense=pyo.minimize
    )

    rmp_f.cov = pyo.ConstraintList()
    for c in clients:
        rmp_f.cov.add(
            sum(rmp_f.lam[i] for i in rmp_f.I if c in global_routes[i]["covered"]) == 1
        )

    # <=K via slack: sum(lam) + s_flt == K
    rmp_f.flt = pyo.Constraint(expr=sum(rmp_f.lam[i] for i in rmp_f.I) + rmp_f.s_flt == inst.K)

    _, s = get_solver(max_seconds=600)
    if not s:
        return float("inf"), []

    s.solve(rmp_f, tee=False)

    final_routes = []
    for i in I:
        if pyo.value(rmp_f.lam[i]) > 0.5:
            final_routes.append(global_routes[i]["path"])

    cost_final = float(pyo.value(rmp_f.obj))
    total_time = time.time() - start_time
    print(f"Custo final: {cost_final:.2f} / Tempo: {total_time:.2f}s")

    return cost_final, final_routes

# --- CRIAÇÃO DA INSTÂNCIA (Multi-Depot → Sink) ---
inst = make_instance_md(
    n=15,     # número de clientes
    m=3,      # número de garagens
    K=4,      # frota total
    Q=7,      # capacidade do veículo
    seed=37,
    sink_center=True
)

# Plota apenas os nós (sem rotas ainda)
plot_routes_md(inst, [], "Instância MD-VRP (Garagens → Destino Final)")

print("\n === BRANCH-AND-PRICE (MD → SINK) ===")
cost_bp, routes_bp = solve_full_branch_and_price_md(
    inst,
    alpha=0.5,
    gap_tol=0.01,
    time_limit=3600,
    max_cg_iter=10000
)

plot_routes_md(inst, routes_bp, f"Branch-and-Price MD→Sink (Q={inst.Q}) - Custo {cost_bp:.2f}")

import osmnx as ox
import networkx as nx
import random
import numpy as np
from shapely.geometry import Point
from dataclasses import dataclass

ox.settings.use_cache = True
ox.settings.log_console = False

@dataclass
class RealWorldMDInstance:
    n: int
    m: int
    K: int
    Q: int
    G: any
    depots: list      # índices globais dos depósitos
    clients: list     # índices globais dos clientes
    sink: int         # índice global do sink
    map_idx_to_node: dict  # índice global -> nó OSM
    dist: np.ndarray       # matriz NxN (global indices 0..N-1)


def build_realworld_md_instance(
    n_clients=12,
    m_depots=3,
    city_query="Fortaleza, Ceara, Brazil",
    radius_meters=1500,
    seed=42,
    Q=5,
    K=4,
    school_as_sink=True,  # se True, sink = escola; senão escolhe outro nó
):
    random.seed(seed)
    np.random.seed(seed)

    # 1) Escolhe uma escola (como no seu código)
    tags = {'amenity': 'school'}
    try:
        gdf = ox.features_from_place(city_query, tags)
        school_feature = gdf.sample(1).iloc[0]
        depot_point = school_feature.geometry.centroid
        school_name = school_feature.get('name', 'Escola (Sem Nome)')
    except Exception:
        depot_point = Point(-38.527, -3.730)
        school_name = "Escola Exemplo (Centro)"

    print(f"Escola selecionada: {school_name}")
    print(f"Baixando malha viária (raio {radius_meters}m)...")

    G = ox.graph_from_point(
        (depot_point.y, depot_point.x),
        dist=radius_meters,
        network_type='drive'
    )

    # nó OSM mais próximo da escola
    school_node = ox.nearest_nodes(G, depot_point.x, depot_point.y)

    # 2) Sorteia nós para garagens e clientes
    all_nodes = list(G.nodes)
    if school_node in all_nodes:
        all_nodes.remove(school_node)

    # garante que tem nós suficientes
    needed = m_depots + n_clients + (0 if school_as_sink else 1)
    if len(all_nodes) < needed:
        raise ValueError("Poucos nós no grafo para sortear depots/clients/sink. Aumente radius_meters.")

    chosen = random.sample(all_nodes, m_depots + n_clients + (0 if school_as_sink else 1))

    depot_nodes = chosen[:m_depots]
    client_nodes = chosen[m_depots:m_depots + n_clients]

    if school_as_sink:
        sink_node = school_node
    else:
        sink_node = chosen[-1]

    # 3) Indexação global: [depots | clients | sink]
    # índices globais: 0..m-1 depots; m..m+n-1 clientes; m+n sink
    depots = list(range(m_depots))
    clients = list(range(m_depots, m_depots + n_clients))
    sink = m_depots + n_clients
    N = sink + 1

    map_idx_to_node = {}
    for i, osm_node in enumerate(depot_nodes):
        map_idx_to_node[i] = osm_node
    for k, osm_node in enumerate(client_nodes):
        map_idx_to_node[m_depots + k] = osm_node
    map_idx_to_node[sink] = sink_node

    # 4) Matriz de distâncias (Dijkstra)
    print("Calculando matriz de distâncias reais (Dijkstra)...")
    dist = np.full((N, N), 1e6, dtype=float)
    np.fill_diagonal(dist, 0.0)

    # dica: calcular por fonte melhora performance
    for i in range(N):
        u = map_idx_to_node[i]
        # single_source_dijkstra_path_length é mais rápido para todos destinos
        lengths = nx.single_source_dijkstra_path_length(G, u, weight="length")
        for j in range(N):
            v = map_idx_to_node[j]
            if v in lengths:
                dist[i, j] = float(lengths[v])

    print("Instância MD→Sink pronta!")
    return RealWorldMDInstance(
        n=n_clients, m=m_depots, K=K, Q=Q, G=G,
        depots=depots, clients=clients, sink=sink,
        map_idx_to_node=map_idx_to_node,
        dist=dist
    )

import folium
import networkx as nx

def plot_solution_on_map_md(inst, routes, school_name="Escola (Sink)"):
    """
    Plota rotas abertas (depot -> ... -> sink) no mapa Folium,
    para instância RealWorldMDInstance (multi-depot -> sink).

    inst deve ter:
      - inst.G (grafo OSMnx)
      - inst.depots (lista de índices globais)
      - inst.clients (lista de índices globais)
      - inst.sink (índice global do sink)
      - inst.map_idx_to_node (global idx -> nó OSM)
    routes: lista de paths no formato [depot, ..., sink]
    """
    G = inst.G
    sink_idx = inst.sink
    sink_node = inst.map_idx_to_node[sink_idx]

    # centro do mapa = escola/sink
    lat0 = G.nodes[sink_node]['y']
    lon0 = G.nodes[sink_node]['x']
    m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles='CartoDB positron')

    # --- marcador do SINK (Escola) ---
    folium.Marker(
        location=[lat0, lon0],
        popup=f"<b>Sink (Escola)</b>: {school_name}",
        icon=folium.Icon(color='black', icon='graduation-cap', prefix='fa')
    ).add_to(m)

    # --- marcadores das GARAGENS ---
    for d_idx in inst.depots:
        d_node = inst.map_idx_to_node[d_idx]
        lat = G.nodes[d_node]['y']
        lon = G.nodes[d_node]['x']
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>Garagem</b> (idx={d_idx})",
            icon=folium.Icon(color='blue', icon='home', prefix='fa')
        ).add_to(m)

    # cores
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

    # --- desenha rotas ---
    for ridx, route in enumerate(routes):
        if not route or route[0] not in inst.depots or route[-1] != sink_idx:
            # ignora rota mal formatada
            continue

        col = colors[ridx % len(colors)]
        route_group = folium.FeatureGroup(name=f"Rota {ridx+1} (clientes={len(route)-2})")

        # marcador do primeiro nó (garagem) já existe, mas podemos destacar a saída
        d_node = inst.map_idx_to_node[route[0]]
        folium.CircleMarker(
            location=[G.nodes[d_node]['y'], G.nodes[d_node]['x']],
            radius=6, color=col, fill=True, fill_opacity=0.9,
            popup=f"Saída: Garagem {route[0]}"
        ).add_to(route_group)

        # para cada arco no path: desenha o menor caminho no grafo
        for k in range(len(route) - 1):
            u_idx = route[k]
            v_idx = route[k + 1]
            u_node = inst.map_idx_to_node[u_idx]
            v_node = inst.map_idx_to_node[v_idx]

            # marca clientes (nós intermediários do path)
            if u_idx in inst.clients:
                pt = G.nodes[u_node]
                folium.CircleMarker(
                    location=[pt['y'], pt['x']],
                    radius=5, color=col, fill=True, fill_opacity=0.7,
                    popup=f"Cliente {u_idx}"
                ).add_to(route_group)

            # desenha a geometria do caminho real
            try:
                path_nodes = nx.shortest_path(G, u_node, v_node, weight='length')
                path_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in path_nodes]
                folium.PolyLine(path_coords, color=col, weight=4, opacity=0.8).add_to(route_group)
            except nx.NetworkXNoPath:
                # sem caminho, não desenha esse trecho
                pass

        route_group.add_to(m)

    folium.LayerControl().add_to(m)
    return m

inst = build_realworld_md_instance(
    n_clients=12, m_depots=3,
    city_query="Fortaleza, Ceara, Brazil",
    radius_meters=1500,
    seed=47,
    Q=5, K=4,
    school_as_sink=True
)

cost, routes = solve_full_branch_and_price_md(inst, time_limit=900, gap_tol=0.01)

mapa = plot_solution_on_map_md(inst, routes, school_name="Escola")
mapa
