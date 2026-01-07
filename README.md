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

# 🗺️ Instâncias Reais (OpenStreetMap)

O código permite a geração automática de instâncias reais:

- Escola pública definida como *sink*;
- Garagens e clientes modelados como nós da malha viária;
- Distâncias calculadas como o menor caminho real (Dijkstra).

Visualização interativa com **Folium**:

- Garagens;
- Clientes;
- Escola (*sink*);
- Rotas reais desenhadas sobre a malha urbana.

---

# 📊 Visualizações

- **Matplotlib**: visualização de instâncias sintéticas;
- **Folium**: mapas interativos para instâncias reais;
- Setas indicando a direção das rotas;
- Escola destacada como *sink*.

---

🎓 Contexto Acadêmico

Este projeto é adequado para:

Trabalhos acadêmicos em Otimização Combinatória

Pesquisa em Vehicle Routing Problem

Estudos de Branch-and-Price

Aplicações reais em Transporte Escolar

---

# ✍️ Autores

Jonas Xavier
Ranelle Oliveira
Francisco das Chagas
Aplicação de Branch-and-Price em VRP Multi-Depot com Sink.
