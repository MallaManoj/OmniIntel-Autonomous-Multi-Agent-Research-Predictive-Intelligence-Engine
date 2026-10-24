# OmniIntel Neural Reranker Benchmark Report

Quantitative comparison between baseline Vector Cosine Retrieval and the OmniIntel PyTorch Cross-Encoder.

## 📊 Retrieval Performance Summary

| Evaluation Metric | Baseline Vector Search | OmniIntel Cross-Encoder | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Reciprocal Rank (MRR)** | 0.8667 | **1.0000** | **+15.4%** |
| **NDCG@1** | 0.8000 | **1.0000** | **+20.0%** |
| **NDCG@3** | 0.9000 | **1.0000** | **+11.1%** |
| **Hit Rate@1** | 0.8000 | **1.0000** | **+20.0%** |

## ⚡ Latency & Caching Efficiency

| Serving Mode | Avg Latency / Query | Cache Hit Ratio | Hardware Accelerator |
| :--- | :--- | :--- | :--- |
| **Cold Neural Forward Pass** | 248.11 ms | 0.0% | Apple Silicon GPU (arm) |
| **LRU Cached Retrieval** | **0.03 ms** | **100.0%** | RAM ($O(1)$ Hash + DLL) |

> **Placement Conclusion:** The PyTorch Cross-Encoder resolves semantic ambiguities that fool baseline vector search, achieving a **+15.4% increase in MRR**, while our custom LRU Cache enables **9099.4x sub-millisecond retrieval** on repeated queries.
