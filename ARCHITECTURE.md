# 🏗 OmniIntel — Comprehensive Architectural Blueprint & Mathematical Formulations

> This document provides the rigorous mathematical formulations, algorithmic specifications, memory layout designs, and architectural invariants governing **OmniIntel: Autonomous Multi-Agent Research & Predictive Intelligence Engine**.

---

## 📑 Table of Contents

1. [Architectural Philosophy & System Invariants](#1-architectural-philosophy--system-invariants)
2. [Layer 1: Algorithmic Foundations & Memory Layouts](#2-layer-1-algorithmic-foundations--memory-layouts)
   - [2.1 26-ary Character Prefix Trie](#21-26-ary-character-prefix-trie)
   - [2.2 Contiguous Array Binary Min-Heap Priority Queue](#22-contiguous-array-binary-min-heap-priority-queue)
   - [2.3 Doubly-Linked List + Hash Map LRU Cache](#23-doubly-linked-list--hash-map-lru-cache)
   - [2.4 Directed Acyclic Graph (DAG) & Kahn's Algorithm](#24-directed-acyclic-graph-dag--kahns-algorithm)
3. [Layer 2: PyTorch Deep Learning & Pre-LN Transformer Cross-Encoder](#3-layer-2-pytorch-deep-learning--pre-ln-transformer-cross-encoder)
   - [3.1 Scaled Dot-Product Multi-Head Self-Attention](#31-scaled-dot-product-multi-head-self-attention)
   - [3.2 Pre-LayerNorm vs. Post-LayerNorm Gradient Flow](#32-pre-layernorm-vs-post-layernorm-gradient-flow)
   - [3.3 AdamW Decoupled Weight Decay Formulation](#33-adamw-decoupled-weight-decay-formulation)
   - [3.4 Margin Ranking Loss with Hard Negative Mining](#34-margin-ranking-loss-with-hard-negative-mining)
4. [Layer 3: Knowledge Graph Reasoning & Dual-Channel Hybrid RAG](#4-layer-3-knowledge-graph-reasoning--dual-channel-hybrid-rag)
   - [4.1 Double Adjacency List Representation](#41-double-adjacency-list-representation)
   - [4.2 Dijkstra Logarithmic Confidence Shortest Path](#42-dijkstra-logarithmic-confidence-shortest-path)
   - [4.3 Dual-Channel BM25 + Dense Reciprocal Rank Fusion (RRF)](#43-dual-channel-bm25--dense-reciprocal-rank-fusion-rrf)
   - [4.4 Graph-Associative Query Expansion](#44-graph-associative-query-expansion)
5. [Layer 4: Autonomous Multi-Agent Cognitive System & Dynamic Self-Correction](#5-layer-4-autonomous-multi-agent-cognitive-system--dynamic-self-correction)
   - [5.1 ReAct Cognitive State Machine](#51-react-cognitive-state-machine)
   - [5.2 AST-Sandboxed Python REPL for Deterministic Math](#52-ast-sandboxed-python-repl-for-deterministic-math)
   - [5.3 Closed-Loop Adversarial Reflection & Dynamic DAG Surgery](#53-closed-loop-adversarial-reflection--dynamic-dag-surgery)
6. [Layer 5: Production Gateway, Concurrency & Telemetry](#6-layer-5-production-gateway-concurrency--telemetry)
   - [6.1 Token-Bucket Rate Limiting Formulation](#61-token-bucket-rate-limiting-formulation)
   - [6.2 Asynchronous Event Loop & Lifespan Protocol](#62-asynchronous-event-loop--lifespan-protocol)
   - [6.3 WebSocket Connection Management & Replay Buffer](#63-websocket-connection-management--replay-buffer)

---

## 1. Architectural Philosophy & System Invariants

OmniIntel was engineered around five immutable production invariants:

1. **Zero Hallucination for Numerical & Financial Calculations:**
   Neural language models and transformers must *never* perform arithmetic in generative token space. All quantitative ratios (Gross Margin, P/E, Debt-to-Equity) are executed deterministically inside an AST-sandboxed Python REPL.
2. **Deterministic Pre-LN Stability:**
   Transformers must train reliably without warm-up tricks or auxiliary gradient clipping. Pre-LayerNorm placement guarantees identity gradient highways throughout all residual connections.
3. **Sub-linear Stage-1 Retrieval with Neural Stage-2 Precision:**
   Direct transformer Cross-Encoder evaluation across an entire corpus of $N$ passages requires $O(N)$ transformer forward passes (computationally prohibitive at scale). Retrieval must be strictly two-staged: Stage-1 sub-linear recall ($O(\log N)$) via Inverted Index + Min-Heap Vector Search, followed by Stage-2 deep cross-attention reranking on only top-$K$ candidates ($K \ll N$).
4. **Closed-Loop Adversarial Quality Assurance:**
   No research output is delivered to the user without adversarial verification. A dedicated `CriticAgent` cross-examines citations against retrieved document spans. If claims fail, the system performs dynamic DAG surgery rather than aborting.
5. **Zero External Framework Lock-In:**
   No third-party agent frameworks (LangChain, AutoGen, CrewAI) or HuggingFace model abstractions are used. All data structures, transformers, graph engines, and agent protocols are written natively in pure Python and PyTorch.

---

## 2. Layer 1: Algorithmic Foundations & Memory Layouts

### 2.1 26-ary Character Prefix Trie

The `PrefixTrie` provides $O(L)$ insertion, lookup, and prefix scanning, where $L$ is token length.

```
                         [ROOT]
                        /      \
                      'n'      't'
                      /          \
                    'v'          's'
                    /              \
                  'd'              'm'
                  /                  \
                'a'*                 'c'*
             (NVIDIA)               (TSMC)
```

- **Memory Representation:**
  Each `TrieNode` maintains:
  - `children: Dict[str, TrieNode]` (indexed by character).
  - `is_terminal: bool` (flagging end-of-entity).
  - `payload: Optional[T]` (entity ID, canonical alias, metadata).
- **Linear-Time Entity Extraction ($O(N)$):**
  Given a raw text string $S$ of length $N$, the scanner iterates $i \in [0, N-1]$ and traverses the trie greedily. Longest-match prefixes are emitted in $O(N)$ character steps, replacing multi-pass regular expressions that degrade to $O(M \cdot N)$ on large vocabularies.

### 2.2 Contiguous Array Binary Min-Heap Priority Queue

The `BinaryMinHeap` implements a priority queue over contiguous memory arrays for top-$K$ selection and scheduler dispatch:

- **Heap Property:**
  For any node at index $i$:
  $$\text{parent}(i) = \left\lfloor\frac{i - 1}{2}\right\rfloor, \quad \text{left}(i) = 2i + 1, \quad \text{right}(i) = 2i + 2$$
  $$\text{Invariant: } A[\text{parent}(i)] \le A[i] \quad \forall i > 0$$
- **Bounded Top-$K$ Filtering:**
  When searching $N$ documents for top $K$ items, we maintain a min-heap of size $K$. For each incoming document score $s$:
  $$\text{If } s > A[0], \quad \text{replace } A[0] \leftarrow s \quad \text{and sift-down in } O(\log K)$$
  Total time complexity: $O(N \log K)$, memory bound: $O(K)$.

### 2.3 Doubly-Linked List + Hash Map LRU Cache

The `LRUCache` achieves strict $O(1)$ constant time for both `get()` and `put()` operations:

```
  [Head Sentinel] <───> [Node: doc_nvda] <───> [Node: doc_tsmc] <───> [Tail Sentinel]
         ▲                      ▲                      ▲
         │                      │                      │
   Hash Map Pointers:   {"doc_nvda": Node_1,   "doc_tsmc": Node_2}
```

- **Data Structure:**
  - `_map: Dict[Key, Node]` maps keys directly to heap-allocated node memory addresses ($O(1)$ hash lookup).
  - `_head` and `_tail` sentinel nodes eliminate boundary check branching during insertion/deletion.
  - Thread safety ensured via `threading.RLock()` protecting node pointer splicing during concurrent model inference.

### 2.4 Directed Acyclic Graph (DAG) & Kahn's Algorithm

Given task set $V$ and dependency set $E$ where edge $(u, v) \in E$ denotes that task $u$ must complete before task $v$ can execute:

- **Kahn's Algorithm Formulation ($O(V + E)$):**
  1. Compute in-degree $d_{in}(v) = |\{u \in V \mid (u, v) \in E\}|$ for all $v \in V$.
  2. Initialize queue $Q = \{v \in V \mid d_{in}(v) = 0\}$.
  3. While $Q$ is not empty:
     - Pop all current tasks from $Q$ as a **Parallel Stage Batch** $B_k$.
     - For each task $u \in B_k$, for each neighbor $v$ of $u$:
       - Decrement $d_{in}(v) \leftarrow d_{in}(v) - 1$.
       - If $d_{in}(v) == 0$, add $v$ to next stage queue $Q_{next}$.
     - Advance stage: $Q \leftarrow Q_{next}, k \leftarrow k + 1$.
  4. If total scheduled tasks $|L| \ne |V|$, the graph contains a directed cycle (caught and raised as `CyclicDependencyError`).

---

## 3. Layer 2: PyTorch Deep Learning & Pre-LN Transformer Cross-Encoder

### 3.1 Scaled Dot-Product Multi-Head Self-Attention

Given input embeddings $\mathbf{X} \in \mathbb{R}^{B \times S \times d_{model}}$, linear projections generate Queries, Keys, and Values:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V \quad (\mathbf{W} \in \mathbb{R}^{d_{model} \times d_{model}})$$

For $h$ attention heads with dimension $d_k = d_{model} / h$:

$$\text{Head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i\mathbf{K}_i^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}_i$$

where $\mathbf{M} \in \mathbb{R}^{S \times S}$ is the additive attention mask ($M_{ij} = -\infty$ for padded tokens, $0$ otherwise).

The multi-head output concatenates heads followed by projection $\mathbf{W}_O$:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \left[\text{Head}_1 \mathbin{\Vert} \dots \mathbin{\Vert} \text{Head}_h\right]\mathbf{W}_O$$

### 3.2 Pre-LayerNorm vs. Post-LayerNorm Gradient Flow

In the classic Post-LN transformer (Vaswani et al., 2017):
$$x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$$
The derivative of $x_{l+1}$ with respect to $x_l$ is scaled by the LayerNorm Jacobian $\mathbf{J}_{LN}$, causing vanishing/exploding gradients as depth $L \to \infty$.

**OmniIntel implements Pre-LayerNorm (Pre-LN):**
$$x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))$$

**Mathematical Derivation of Gradient Highway:**
Expanding $x_L$ across $L$ layers:
$$x_L = x_0 + \sum_{l=0}^{L-1} \text{SubLayer}(\text{LayerNorm}(x_l))$$

Taking the gradient of the loss $\mathcal{L}$ with respect to input $x_0$:
$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left(\mathbf{I} + \sum_{l=0}^{L-1} \frac{\partial}{\partial x_0} \text{SubLayer}(\text{LayerNorm}(x_l))\right)$$

Because the identity matrix $\mathbf{I}$ is directly preserved, the gradient $\frac{\partial \mathcal{L}}{\partial x_L}$ flows backward to $x_0$ without being repeatedly scaled down by LayerNorm denominators, guaranteeing stable convergence without artificial learning rate warmups.

### 3.3 AdamW Decoupled Weight Decay Formulation

L2 regularization in standard Adam is mathematically flawed because weight decay is applied *inside* the moving gradient average, scaling inversely with historical gradient magnitudes.

**OmniIntel AdamW Parameter Update Invariant:**
For parameter $\theta_t$, gradient $g_t$, learning rate $\eta$, weight decay $\lambda$, and moment estimates $m_t, v_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Parameter Grouping Rule:**
- Group 1 (2D Tensor Weights): Apply decoupled decay ($\lambda = 0.01$).
- Group 2 (1D Biases & LayerNorm Gains): Decoupled decay strictly zero ($\lambda = 0.0$).

### 3.4 Margin Ranking Loss with Hard Negative Mining

Given query representation $\mathbf{q}$, ground-truth passage $\mathbf{p}^+$, and adversarial distractor $\mathbf{p}^-$:

$$\mathcal{L}(\mathbf{q}, \mathbf{p}^+, \mathbf{p}^-) = \max\left(0, \gamma - \left(f(\mathbf{q}, \mathbf{p}^+) - f(\mathbf{q}, \mathbf{p}^-)\right)\right)$$

where $\gamma = 0.5$ represents the mandatory score gap. The loss forces the Cross-Encoder scoring head to separate relevant passages from false positives by a strictly positive boundary.

---

## 4. Layer 3: Knowledge Graph Reasoning & Dual-Channel Hybrid RAG

### 4.1 Double Adjacency List Representation

The `KnowledgeGraph` stores entities and fact triples $T = (u, r, v, c)$ in dual adjacency maps:

$$\mathbf{Out}[u] = \{(v, r, c) \mid (u, r, v, c) \in T\}$$
$$\mathbf{In}[v] = \{(u, r, c) \mid (u, r, v, c) \in T\}$$

This guarantees $O(1)$ bidirectional lookup for node in-degree, out-degree, and immediate upstream/downstream neighborhood traversal.

### 4.2 Dijkstra Logarithmic Confidence Shortest Path

Let $c(e) \in (0, 1]$ represent the empirical extraction confidence of relation edge $e$. The confidence of a path $P = (e_1, e_2, \dots, e_k)$ is the multiplicative joint probability:

$$C(P) = \prod_{i=1}^k c(e_i)$$

Standard Dijkstra requires non-negative *additive* edge weights: $\sum w(e)$.
We apply the negative natural logarithm mapping:

$$w(e) = -\ln(c(e))$$

Because $c(e) \in (0, 1]$, $-\ln(c(e)) \ge 0$ (non-negative, satisfying Dijkstra's precondition).

Minimizing the additive sum of logarithmic weights directly maximizes multiplicative path confidence:

$$\min \sum_{i=1}^k w(e_i) = \min \sum_{i=1}^k (-\ln(c(e_i))) = \min \left(-\ln \prod_{i=1}^k c(e_i)\right) = \max \prod_{i=1}^k c(e_i) = \max C(P)$$

The path is discovered in $O((|V| + |E|) \log |V|)$ using our binary min-heap priority queue.

### 4.3 Dual-Channel BM25 + Dense Reciprocal Rank Fusion (RRF)

To achieve robust recall across both exact technical identifiers (e.g., "MI300X", "CoWoS", "B200") and conceptual semantics:

- **Lexical BM25 Channel:**
  $$\text{BM25}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  where $k_1 = 1.5, b = 0.75$.
- **Dense Cosine Channel:**
  $$\text{Sim}_{\text{dense}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **Reciprocal Rank Fusion (RRF):**
  Combining rank orders without assuming comparable score distributions:
  $$\text{RRFScore}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + r_m(d)}$$
  where $k = 60$ is the smoothing constant preventing high-rank dominance, and $r_m(d)$ is the 1-based rank of document $d$ in channel $m$.

### 4.4 Graph-Associative Query Expansion

When user query mentions entity $E_{seed}$, the engine extracts its $k$-hop neighbors $\mathcal{N}_k(E_{seed})$ via BFS traversal on the knowledge graph:

$$Q_{\text{expanded}} = Q_{\text{original}} \cup \left\{ \text{name}(e) \mid e \in \mathcal{N}_k(E_{seed}) \land \text{conf}(e) \ge \tau \right\}$$

This automatically bridges conceptual queries (e.g., "TSMC lithography constraints") with upstream supply chain entities ("ASML", "EUV Steppers").

---

## 5. Layer 4: Autonomous Multi-Agent Cognitive System & Dynamic Self-Correction

### 5.1 ReAct Cognitive State Machine

Every agent inherits from `BaseAgent`, enforcing strict state transition invariants:

```
  ┌─────────┐     dispatch()     ┌──────────┐     generate_plan()    ┌────────┐
  │  IDLE   │ ─────────────────▶ │ PLANNING │ ─────────────────────▶ │ ACTING │
  └─────────┘                    └──────────┘                        └────────┘
       ▲                                                                  │
       │                              evaluate()                          │
       │                        ┌─────────────────────────────────────────┘
       │                        ▼
  ┌────────────┐    success    ┌────────────┐
  │ TERMINATED │ ◀──────────── │ REFLECTING │
  └────────────┘               └────────────┘
                                     │ retry / replan
                                     ▼
                               ┌──────────┐
                               │ PLANNING │
                               └──────────┘
```

- Invariant: An agent cannot enter `ACTING` without passing through `PLANNING`.
- Epistemic memory bounds prevent infinite trajectory accumulation.

### 5.2 AST-Sandboxed Python REPL for Deterministic Math

The `ASTSandboxREPL` compiles mathematical code into an Abstract Syntax Tree (AST) before execution.

- **AST Inspection Rules:**
  - Nodes must belong strictly to permitted mathematical subsets: `ast.Expression`, `ast.BinOp`, `ast.UnaryOp`, `ast.Constant`, `ast.Assign`, `ast.Name`.
  - Dangerous nodes (`ast.Import`, `ast.ImportFrom`, `ast.Call` [to unauthorized built-ins], `ast.Attribute`, `ast.While`) immediately raise `SandboxSecurityError`.
- **Namespace Whitelist:**
  - Global scope initialized with `__builtins__ = None`.
  - Permitted functions: `abs`, `round`, `min`, `max`, `sum`, `pow`, `float`, `int`.

### 5.3 Closed-Loop Adversarial Reflection & Dynamic DAG Surgery

```
   [Researcher Output] ──┐
                         ├────▶ [CriticAgent Cross-Examination]
      [Quant Output]   ──┘                    │
                                              ├─────── Verdict: APPROVED ──▶ [Synthesizer]
                                              │
                                              └─────── Verdict: REVISE
                                                          │
                                            ┌─────────────┴─────────────┐
                                            ▼                           ▼
                                   [Dynamic DAG Surgery]       [Targeted Re-Query]
                                   Inject Corrective Task     Refine Grounding Context
```

When `CriticAgent` detects numerical conflict or ungrounded claims:
1. Emits structured `CriticIssue` with claim text, severity, and suggested correction.
2. `MasterOrchestrator` performs dynamic DAG surgery:
   - Identifies downstream dependencies of the failed node.
   - Injects a specialized corrective node into the execution DAG.
   - Recomputes Kahn's topological order on the fly and resumes execution.

---

## 6. Layer 5: Production Gateway, Concurrency & Telemetry

### 6.1 Token-Bucket Rate Limiting Formulation

Each client IP is allocated a bucket with capacity $C = 30$ tokens and refill rate $r = 10.0 \text{ tokens/sec}$.

For request arriving at time $t_{curr}$ with previous update $t_{last}$:

$$\Delta t = t_{curr} - t_{last}$$
$$\text{Tokens}_{\text{new}} = \min(C, \text{Tokens}_{\text{curr}} + \Delta t \cdot r)$$

$$\text{Decision:} \quad \begin{cases}
\text{Tokens}_{\text{new}} \ge 1 \implies \text{Tokens} \leftarrow \text{Tokens}_{\text{new}} - 1, & \text{ALLOW (HTTP 200)} \\
\text{Tokens}_{\text{new}} < 1 \implies \text{RetryAfter} = \frac{1 - \text{Tokens}_{\text{new}}}{r}, & \text{DENY (HTTP 429)}
\end{cases}$$

### 6.2 Asynchronous Event Loop & Lifespan Protocol

FastAPI application lifespan (`@asynccontextmanager`) guarantees orderly initialization and termination:

1. **Startup:**
   - Instantiate in-memory Knowledge Graph and populate foundational entities/triples.
   - Index core SEC filing corpora into Inverted Index and Vector Store.
   - Warm PyTorch Pre-LN Cross-Encoder and initialize inference cache.
   - Initialize `ConnectionManager` and start `EventBus` consumers.
2. **Shutdown:**
   - Flush pending telemetry logs.
   - Terminate WebSocket connection pools gracefully with close frame 1001 (Going Away).
   - Clear GPU / CPU tensor allocations.

### 6.3 WebSocket Connection Management & Replay Buffer

- **Replay Buffer:** Implements a bounded circular buffer (capacity 100) storing recent events. New subscribers connecting with `?replay=true` immediately receive missed events before entering real-time broadcast mode.
- **Isolated Error Boundaries:** During broadcast iteration, individual dead or stalled socket exceptions are captured in isolation, preventing a single slow consumer from blocking broadcast loops to other active clients.

---

## 📐 Summary of Algorithmic Complexity

| Subsystem Component | Algorithm / Structure | Time Complexity | Space Complexity |
|:---|:---|:---:|:---:|
| **Entity Scanner** | 26-ary Character Prefix Trie | $O(N)$ text scanning | $O(V \cdot \Sigma)$ |
| **Priority Queue** | Binary Min-Heap | $O(\log K)$ push/pop | $O(K)$ contiguous array |
| **Inference Cache** | Hash Map + Doubly-Linked List | $O(1)$ get/put | $O(M)$ nodes |
| **Task Planner** | Kahn's Topological Sorting | $O(V + E)$ | $O(V + E)$ in-degree map |
| **Cycle Detector** | Tarjan's SCC / DFS Back-Edge | $O(V + E)$ | $O(V)$ recursion stack |
| **Path Reasoner** | Dijkstra on $-\ln(\text{conf})$ | $O((V + E) \log V)$ | $O(V)$ distance map |
| **Sparse Index** | Inverted Index with BM25 | $O(\|Q\| \cdot \text{avg}(L_{post}))$ | $O(T_{\text{unique}} \cdot D)$ postings |
| **Dense Search** | Vector Store + Min-Heap Top-$K$ | $O(N \cdot d + N \log K)$ | $O(N \cdot d)$ embeddings |
| **Neural Attention** | Multi-Head Self-Attention | $O(S^2 \cdot d)$ | $O(S^2)$ attention matrix |
| **Rate Limiter** | Token-Bucket per IP | $O(1)$ | $O(\text{Active IPs})$ |

---
*OmniIntel Engineering Team — Systems & Machine Learning Architecture.*
