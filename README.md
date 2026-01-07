# 🚍 Multi-Depot Vehicle Routing Problem with Sink
Branch-and-Price com Geração de Colunas (Python / Pyomo)

A dapta o solver para o Multi-Depot Vehicle Routing Problem com destino final fixo, onde veículos partem de múltiplas garagens e terminam em uma escola.

---

# 📌 Problema Modelado

Garagens (depots): múltiplos pontos de partida

Clientes: devem ser atendidos exatamente uma vez

Escola: destino final comum para todas as rotas

Capacidade: limitada por veículo

Frota total: limitada por 𝐾.

Cada rota tem o formato:

garagem → clientes → escola.

Não é obrigatório utilizar todas as garagens.

---

# 🧠 Formulação Matemática 
Problema Mestre — Set Partitioning

Minimiza o custo total das rotas selecionadas:

```math
\min_{r \in \Omega} \sum_{r \in \Omega} c_r \lambda_r
```

Sujeito a:

```math
\sum_{r \in \Omega} a_{ir} \lambda_r = 1 \quad \forall i \in \mathcal{C}
```

```math
\sum_{r \in \Omega} \lambda_r \le K
```

```math
\lambda_r \in \{0,1\} \quad \forall r \in \Omega
```

Onde:

- $\mathcal{C}$: conjunto de clientes;
- $a_{ir} = 1$ se a rota $r$ atende o cliente $i$, e $0$ caso contrário;
- $c_r$: custo associado à rota $r$;
- $K$: número máximo de veículos disponíveis.

---
​
# Subproblema (Pricing)

Resolve um ESPPRC (Elementary Shortest Path Problem with Resource Constraints), buscando rotas com custo reduzido negativo:

```math
\bar{c}_r = c_r - \sum_{i \in \mathcal{C}} \pi_i a_{ir} - \mu
```
Implementado de duas formas:

- **Heurístico**: abordagem gulosa com refinamento por *2-opt aberto*.
- **Exato**: modelo de Programação Inteira Mista (MIP) implementado em Pyomo, utilizando restrições MTZ e de capacidade.

---

# ⚙️ Estrutura do Código
## Principais Componentes

- **VRPInstanceMD**  
  Estrutura da instância *multi-depot → sink* (sintética).

- **RealWorldMDInstance**  
  Instância baseada em dados reais, construída a partir do **OSMnx**.

- **run_grasp_vrp_md**  
  *Warm-start* via GRASP, gerando rotas longas iniciais.

- **improve_route_2opt_open**  
  Operador de busca local *2-opt* para rotas abertas (depósito → *sink*).

- **solve_exact_pricing_md**  
  Subproblema de *pricing* resolvido exatamente via MIP (ESPPRC).

- **solve_node_md**  
  Resolução de um nó da árvore do **Branch-and-Price**.

- **solve_full_branch_and_price_md**  
  Implementação completa do algoritmo **Branch-and-Price**.

---

🗺️ Instâncias Reais (OpenStreetMap)

O código permite gerar instâncias reais automaticamente:

Escola pública = sink

Garagens e clientes = nós da malha viária

Distâncias = menor caminho real (Dijkstra)

Visualização interativa com Folium:

Garagens

Clientes

Escola (sink)

Rotas reais desenhadas na malha urbana

▶️ Como Executar
1️⃣ Dependências

No Google Colab (ou Ubuntu):

apt-get install -y glpk-utils coinor-cbc
pip install pyomo osmnx folium networkx matplotlib

2️⃣ Executar Instância Sintética
inst = make_instance_md(
    n=15,
    m=3,
    K=4,
    Q=7,
    seed=37,
    sink_center=True
)

cost, routes = solve_full_branch_and_price_md(inst)
plot_routes_md(inst, routes)

3️⃣ Executar Instância Real (OSM)
inst = build_realworld_md_instance(
    n_clients=12,
    m_depots=3,
    city_query="Fortaleza, Ceara, Brazil",
    radius_meters=1500,
    Q=5,
    K=4,
    school_as_sink=True
)

cost, routes = solve_full_branch_and_price_md(inst)
mapa = plot_solution_on_map_md(inst, routes)
mapa

🧪 Características Importantes

✔️ Rotas abertas (não retornam ao depósito)

✔️ Não obriga uso de todas as garagens

✔️ Branching em arcos (estável para B&P)

✔️ Estabilização de duais (smoothing)

✔️ Fallback exato garante correção

✔️ Compatível com dados reais

📊 Visualizações

Matplotlib: instâncias sintéticas

Folium: mapas interativos reais

Setas indicam direção da rota

Escola destacada como sink

🎓 Contexto Acadêmico

Este projeto é adequado para:

Trabalhos acadêmicos em Otimização Combinatória

Pesquisa em Vehicle Routing Problem

Estudos de Branch-and-Price

Aplicações reais em Transporte Escolar

✍️ Autor

Jonas Xavier
Projeto desenvolvido para estudo e aplicação de
Branch-and-Price em VRP Multi-Depot com Sink
