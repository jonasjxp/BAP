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

#🧠 Formulação Matemática 
Problema Mestre — Set Partitioning

Minimiza o custo total das rotas selecionadas:

```math
\min_{r \in \Omega} \sum_{r \in \Omega} c_r \lambda_r
```

Sujeito a:

∑
𝑟
∈
Ω
𝑎
𝑖
𝑟
𝜆
𝑟
=
1
∀
𝑖
∈
𝐶
r∈Ω
∑
	​

a
ir
	​

λ
r
	​

=1∀i∈C
∑
𝑟
∈
Ω
𝜆
𝑟
≤
𝐾
r∈Ω
∑
	​

λ
r
	​

≤K
𝜆
𝑟
∈
{
0
,
1
}
λ
r
	​

∈{0,1}

Onde:

𝐶
C: conjunto de clientes

𝑎𝑖𝑟 = 1
air=1 se a rota 
𝑟 atende o cliente 
𝑖
𝑐
𝑟
: custo da rota
K: número máximo de veículos
	
---​
# Subproblema (Pricing)

Resolve um ESPPRC (Elementary Shortest Path Problem with Resource Constraints), buscando rotas com custo reduzido negativo:

𝑐
ˉ
𝑟
=
𝑐
𝑟
−
∑
𝑖
𝜋
𝑖
𝑎
𝑖
𝑟
−
𝜇
c
ˉ
r
	​

=c
r
	​

−
i
∑
	​

π
i
	​

a
ir
	​

−μ

Implementado de duas formas:

Heurístico (guloso + 2-opt aberto)

Exato (MIP com Pyomo, MTZ + capacidade)

⚙️ Estrutura do Código
Principais Componentes

VRPInstanceMD
Estrutura da instância multi-depot → sink (sintética)

RealWorldMDInstance
Instância baseada em dados reais (OSMnx)

run_grasp_vrp_md
Warm-start via GRASP com rotas longas

improve_route_2opt_open
2-opt para rotas abertas (depot → sink)

solve_exact_pricing_md
Pricing exato via MIP (ESPPRC)

solve_node_md
Resolução de um nó do Branch-and-Price

solve_full_branch_and_price_md
Algoritmo completo de Branch-and-Price

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
