"""HTML Dashboard Router for OmniIntel.

Provides an interactive Web GUI accessible at /ui:
  - Autonomous Multi-Agent Mission Execution & Live Briefing
  - Kahn's DAG Execution Planning with Visual Graphviz DOT
  - Knowledge Graph Multi-Hop Dijkstra Reasoning & Subgraph Explorer
  - Dual-Channel Hybrid Search & Neural Cross-Encoder Reranker
  - Live WebSocket Telemetry Feed
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI Dashboard"])

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniIntel — Autonomous Intelligence Engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: #1f2937;
            --border-color: #374151;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-cyan);
            letter-spacing: -0.02em;
        }
        .brand span { color: var(--text-main); }
        .badge {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }
        .badge::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: var(--accent-green);
            box-shadow: 0 0 6px var(--accent-green);
        }
        .nav-links a {
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.85rem;
            margin-left: 1.5rem;
            transition: color 0.15s;
        }
        .nav-links a:hover { color: var(--accent-cyan); }
        
        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }
        
        .tabs {
            display: flex;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
        }
        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 0.75rem 1.25rem;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab-btn:hover { color: var(--text-main); }
        .tab-btn.active {
            color: var(--accent-cyan);
            border-bottom-color: var(--accent-cyan);
        }
        
        .tab-pane { display: none; }
        .tab-pane.active { display: block; }
        
        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card-desc {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 1.25rem;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }
        input[type="text"], textarea, select {
            width: 100%;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            font-family: inherit;
            font-size: 0.9rem;
            transition: border-color 0.15s;
        }
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: var(--accent-cyan);
        }
        .btn {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: opacity 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .output-box {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.25rem;
            margin-top: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .rendered-markdown {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-top: 1rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .rendered-markdown h1, .rendered-markdown h2, .rendered-markdown h3 {
            color: var(--accent-cyan);
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .rendered-markdown table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        .rendered-markdown th, .rendered-markdown td {
            border: 1px solid var(--border-color);
            padding: 0.5rem 0.75rem;
            text-align: left;
        }
        .rendered-markdown th { background-color: rgba(255, 255, 255, 0.05); }
        .rendered-markdown code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(255, 255, 255, 0.1);
            padding: 0.15rem 0.35rem;
            border-radius: 0.25rem;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 900px) {
            .grid-2 { grid-template-columns: 1fr; }
        }
        
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .status-pill.approved { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .status-pill.revise { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
        
        .meta-stat {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            ⚡ OmniIntel <span>Engine</span>
            <span class="badge" id="system-badge">System Online</span>
        </div>
        <div class="nav-links">
            <a href="/docs" target="_blank">Swagger API</a>
            <a href="/redoc" target="_blank">ReDoc</a>
            <a href="/health/ready" target="_blank">Readiness JSON</a>
            <a href="https://github.com/MallaManoj/OmniIntel-Autonomous-Multi-Agent-Research-Predictive-Intelligence-Engine" target="_blank">GitHub Repo</a>
        </div>
    </header>

    <main>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('mission-tab')">🚀 Autonomous Mission</button>
            <button class="tab-btn" onclick="switchTab('dag-tab')">🗺 Kahn's DAG Planner</button>
            <button class="tab-btn" onclick="switchTab('graph-tab')">🕸 Knowledge Graph & Dijkstra</button>
            <button class="tab-btn" onclick="switchTab('search-tab')">🔍 Hybrid Search & Reranker</button>
            <button class="tab-btn" onclick="switchTab('trie-tab')">⚡ Prefix Trie Scanner</button>
            <button class="tab-btn" onclick="switchTab('ws-tab')">📡 Live WebSocket Feed</button>
        </div>

        <!-- 1. MISSION TAB -->
        <div id="mission-tab" class="tab-pane active">
            <div class="card">
                <div class="card-title">🚀 Autonomous Multi-Agent Research Mission</div>
                <div class="card-desc">Execute autonomous research across Planner, Researcher, Quant (AST REPL), Critic reflection, and Synthesizer agents.</div>
                
                <div class="form-group">
                    <label>Research Directive / Query</label>
                    <input type="text" id="mission-query" value="Compare NVIDIA and TSMC in sovereign AI infrastructure and evaluate gross margins.">
                </div>
                <div class="form-group">
                    <label>Max Self-Correction Replans</label>
                    <input type="number" id="mission-replans" value="2" min="1" max="5" style="width: 120px;">
                </div>
                <button class="btn" id="run-mission-btn" onclick="runMission()">⚡ Launch Multi-Agent Mission</button>

                <div id="mission-loading" style="display: none; margin-top: 1rem; color: var(--accent-cyan);">
                    ⚙️ Executing Kahn's DAG stages, AST financial sandbox, and Critic reflection loop...
                </div>

                <div id="mission-result" style="display: none; margin-top: 1.5rem;">
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
                        <span class="status-pill approved" id="mission-verdict">APPROVED</span>
                        <span class="meta-stat">🎯 Confidence: <strong id="mission-conf">91.1%</strong></span>
                        <span class="meta-stat">⏱ Total Latency: <strong id="mission-lat">--</strong> ms</span>
                        <span class="meta-stat">📦 Stages Executed: <strong id="mission-stages">--</strong></span>
                    </div>
                    <div class="rendered-markdown" id="mission-markdown"></div>
                </div>
            </div>
        </div>

        <!-- 2. DAG PLANNER TAB -->
        <div id="dag-tab" class="tab-pane">
            <div class="card">
                <div class="card-title">🗺 Kahn's Algorithm DAG Decomposition</div>
                <div class="card-desc">Decompose high-level directives into parallel execution stages using topological sorting.</div>

                <div class="form-group">
                    <label>Query Directive</label>
                    <input type="text" id="dag-query" value="Analyze AMD Instinct MI300X datacenter solutions and compute financial margins.">
                </div>
                <button class="btn" onclick="runDagPlan()">📐 Generate DAG Schedule</button>

                <div class="output-box" id="dag-output">// Click "Generate DAG Schedule" to view topological execution batches</div>
            </div>
        </div>

        <!-- 3. KNOWLEDGE GRAPH TAB -->
        <div id="graph-tab" class="tab-pane">
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">🔗 Dijkstra Shortest Path Reasoning</div>
                    <div class="card-desc">Minimize -ln(confidence) to discover highest-probability multi-hop reasoning chains.</div>
                    
                    <div class="form-group">
                        <label>Source Entity</label>
                        <input type="text" id="graph-src" value="nvidia">
                    </div>
                    <div class="form-group">
                        <label>Target Entity</label>
                        <input type="text" id="graph-tgt" value="asml">
                    </div>
                    <button class="btn" onclick="runPathQuery()">🔍 Discover Multi-Hop Path</button>
                    
                    <div class="output-box" id="path-output">// Discovered path reasoning will appear here</div>
                </div>

                <div class="card">
                    <div class="card-title">🌐 Egocentric Subgraph Extraction</div>
                    <div class="card-desc">Extract k-hop local relational topology around a root entity.</div>
                    
                    <div class="form-group">
                        <label>Root Entity ID</label>
                        <input type="text" id="subgraph-root" value="tsmc">
                    </div>
                    <div class="form-group">
                        <label>Max Hops (k)</label>
                        <input type="number" id="subgraph-hops" value="1" min="1" max="3">
                    </div>
                    <button class="btn" onclick="runSubgraph()">🕸 Extract Subgraph</button>
                    
                    <div class="output-box" id="subgraph-output">// Subgraph nodes and fact triples will appear here</div>
                </div>
            </div>
        </div>

        <!-- 4. HYBRID SEARCH TAB -->
        <div id="search-tab" class="tab-pane">
            <div class="card">
                <div class="card-title">🔍 Dual-Channel Hybrid Search & Cross-Encoder Reranker</div>
                <div class="card-desc">Stage-1 Recall (BM25 + Cosine Min-Heap with RRF) + Stage-2 Pre-LN Transformer neural reranking.</div>

                <div class="form-group">
                    <label>Search Query</label>
                    <input type="text" id="search-query" value="CoWoS advanced packaging capacity TSMC NVIDIA">
                </div>
                <div class="form-group" style="display: flex; gap: 1rem;">
                    <div>
                        <label>Top K</label>
                        <input type="number" id="search-topk" value="3" min="1" max="10" style="width: 100px;">
                    </div>
                    <div>
                        <label>Fusion Mode</label>
                        <select id="search-mode" style="width: 150px;">
                            <option value="rrf" selected>Reciprocal Rank Fusion (RRF)</option>
                            <option value="linear">Linear Interpolation</option>
                        </select>
                    </div>
                </div>
                <button class="btn" onclick="runHybridSearch()">⚡ Execute Dual-Channel Search</button>

                <div class="output-box" id="search-output">// Retrieved passages and neural reranking scores will appear here</div>
            </div>
        </div>

        <!-- 5. PREFIX TRIE TAB -->
        <div id="trie-tab" class="tab-pane">
            <div class="card">
                <div class="card-title">⚡ 26-ary Character Prefix Trie Scanner</div>
                <div class="card-desc">Scan unstructured filing text in O(N) linear time without expensive regex passes.</div>

                <div class="form-group">
                    <label>Unstructured Filing Text</label>
                    <textarea id="trie-text" rows="4">NVIDIA Corporation leverages TSMC advanced CoWoS packaging for its Blackwell B200 accelerators. ASML provides EUV lithography systems while AMD competes with its Instinct architecture.</textarea>
                </div>
                <button class="btn" onclick="runTrieScan()">🔎 Scan Entities</button>

                <div class="output-box" id="trie-output">// Linear-time entity matches will appear here</div>
            </div>
        </div>

        <!-- 6. WEBSOCKET TAB -->
        <div id="ws-tab" class="tab-pane">
            <div class="card">
                <div class="card-title">📡 Live Asynchronous WebSocket Stream (/ws/events)</div>
                <div class="card-desc">Real-time broadcast of multi-agent cognitive events and subsystem state transitions.</div>
                
                <div style="margin-bottom: 1rem;">
                    <span class="badge" id="ws-status">Disconnected</span>
                    <button class="btn" style="margin-left: 1rem; padding: 0.4rem 0.8rem; font-size: 0.8rem;" onclick="connectWebSocket()">Reconnect</button>
                </div>

                <div class="output-box" id="ws-log" style="height: 350px;">// WebSocket event stream will display live JSON frames</div>
            </div>
        </div>
    </main>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }

        async function runMission() {
            const query = document.getElementById('mission-query').value;
            const replans = parseInt(document.getElementById('mission-replans').value) || 2;
            const btn = document.getElementById('run-mission-btn');
            const loader = document.getElementById('mission-loading');
            const resultBox = document.getElementById('mission-result');

            btn.disabled = true;
            loader.style.display = 'block';
            resultBox.style.display = 'none';

            try {
                const res = await fetch('/api/v1/mission/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, max_replans: replans })
                });
                const data = await res.json();
                
                document.getElementById('mission-verdict').innerText = data.verdict || 'COMPLETED';
                document.getElementById('mission-conf').innerText = ((data.confidence_score || 0) * 100).toFixed(1) + '%';
                document.getElementById('mission-lat').innerText = data.total_latency_ms || 0;
                document.getElementById('mission-stages').innerText = data.total_stages_executed || 0;
                document.getElementById('mission-markdown').innerHTML = marked.parse(data.dossier_markdown || '');
                resultBox.style.display = 'block';
            } catch (err) {
                alert('Mission execution failed: ' + err);
            } finally {
                btn.disabled = false;
                loader.style.display = 'none';
            }
        }

        async function runDagPlan() {
            const query = document.getElementById('dag-query').value;
            const out = document.getElementById('dag-output');
            out.innerText = 'Planning execution DAG...';
            try {
                const res = await fetch('/api/v1/mission/plan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await res.json();
                out.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                out.innerText = 'Error: ' + err;
            }
        }

        async function runPathQuery() {
            const src = document.getElementById('graph-src').value;
            const tgt = document.getElementById('graph-tgt').value;
            const out = document.getElementById('path-output');
            out.innerText = 'Calculating Dijkstra shortest path...';
            try {
                const res = await fetch(`/api/v1/graph/path?source=${encodeURIComponent(src)}&target=${encodeURIComponent(tgt)}`);
                const data = await res.json();
                out.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                out.innerText = 'Error: ' + err;
            }
        }

        async function runSubgraph() {
            const root = document.getElementById('subgraph-root').value;
            const hops = parseInt(document.getElementById('subgraph-hops').value) || 1;
            const out = document.getElementById('subgraph-output');
            out.innerText = 'Extracting egocentric subgraph...';
            try {
                const res = await fetch('/api/v1/graph/subgraph', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ root_entity_id: root, max_hops: hops })
                });
                const data = await res.json();
                out.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                out.innerText = 'Error: ' + err;
            }
        }

        async function runHybridSearch() {
            const query = document.getElementById('search-query').value;
            const topk = parseInt(document.getElementById('search-topk').value) || 3;
            const mode = document.getElementById('search-mode').value;
            const out = document.getElementById('search-output');
            out.innerText = 'Executing Stage-1 RRF & Stage-2 Pre-LN Cross-Encoder...';
            try {
                const res = await fetch('/api/v1/search/hybrid', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: topk, fusion_mode: mode, use_reranker: true })
                });
                const data = await res.json();
                out.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                out.innerText = 'Error: ' + err;
            }
        }

        async function runTrieScan() {
            const text = document.getElementById('trie-text').value;
            const out = document.getElementById('trie-output');
            out.innerText = 'Scanning text via Prefix Trie...';
            try {
                const res = await fetch('/api/v1/search/scan-entities', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await res.json();
                out.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                out.innerText = 'Error: ' + err;
            }
        }

        // Live WebSocket Telemetry Stream
        let ws;
        function connectWebSocket() {
            const badge = document.getElementById('ws-status');
            const log = document.getElementById('ws-log');
            const wsUrl = `ws://${window.location.host}/ws/events?replay=true`;
            
            try {
                ws = new WebSocket(wsUrl);
                ws.onopen = () => {
                    badge.innerText = 'Connected';
                    badge.style.backgroundColor = 'rgba(16, 185, 129, 0.2)';
                    badge.style.color = '#34d399';
                    log.innerText = `[${new Date().toISOString()}] Connected to ${wsUrl}\\n` + log.innerText;
                };
                ws.onmessage = (event) => {
                    const parsed = JSON.parse(event.data);
                    log.innerText = `[${new Date().toISOString()}] ${JSON.stringify(parsed, null, 2)}\\n` + log.innerText;
                };
                ws.onclose = () => {
                    badge.innerText = 'Disconnected';
                    badge.style.backgroundColor = 'rgba(244, 63, 94, 0.2)';
                    badge.style.color = '#fb7185';
                };
            } catch (e) {
                console.error(e);
            }
        }
        window.addEventListener('load', connectWebSocket);
    </script>
</body>
</html>
"""

@router.get("/ui", response_class=HTMLResponse, summary="OmniIntel Web UI Dashboard")
async def get_ui_dashboard() -> HTMLResponse:
    """Render the interactive OmniIntel Web UI single-page dashboard."""
    return HTMLResponse(content=HTML_CONTENT)
