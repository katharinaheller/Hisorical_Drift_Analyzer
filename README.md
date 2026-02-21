# Historical Drift Analyzer (HDA)

A modular Retrieval-Augmented Generation (RAG) pipeline for analyzing **semantic drift over time** in scientific literature.  
The system combines **temporal-aware retrieval**, **intent-specific reranking**, and **reproducible evaluation** to study how conceptual meanings evolve across decades.

---

## Research Objective

The project investigates how historically informed retrieval and reranking strategies affect:

- semantic relevance of retrieved evidence,
- temporal coverage for diachronic analysis,
- faithfulness of generated answers to cited sources.

The core research question is how a **modular RAG architecture** can be designed to support **reproducible, methodologically transparent analysis of semantic change** in scientific corpora.

---

## System Architecture

The pipeline separates **offline corpus processing** (ingestion, embedding, indexing) from **online query processing** (retrieval, reranking, generation).

![Overall pipeline architecture](assets/architecture_pipeline.png)

**Key components:**
- PDF ingestion and metadata extraction (publication year as temporal anchor)
- Text chunking with configurable overlap
- Dense embeddings + FAISS vector store
- Intent classification (conceptual vs. chronological queries)
- Intent-specific reranking:
  - Semantic reranker for conceptual coherence
  - Temporal reranker for diachronic coverage
- LLM-based answer synthesis with explicit source citations

---

## Evaluation Methodology

The evaluation follows a two-level framework:

1) **Intrinsic Retrieval Quality**  
   - Metric: NDCG@k  
   - Ground truth derived from embedding-based semantic ordering  
   - Evaluated for multiple top-k configurations

2) **Extrinsic Answer Faithfulness**  
   - Claim-based faithfulness metric  
   - Measures how strongly generated statements are supported by retrieved evidence

### NDCG Distribution

![NDCG distribution across top-k](assets/ndcg_distribution.png)

### Faithfulness Comparison Across Models

![Faithfulness comparison across models](assets/faithfulness_bands.png)

The results demonstrate stable retrieval relevance and consistent faithfulness behavior across different top-k settings and model backends.

---

## Reproducibility

All pipeline stages are deterministic and configurable.

```bash
# create environment
poetry install

# build embeddings and FAISS index
python scripts/build_index.py

# run evaluation
python scripts/run_evaluation.py

All intermediate artifacts (retrieved chunks, reranking scores, evaluation logs) are persisted to enable full auditability of experimental results.

Project structure

.
├── src/                # core pipeline modules
├── scripts/            # indexing and evaluation runners
├── configs/            # experiment configurations
├── tests/              # unit and integration tests
├── assets/             # figures used in README
└── README.md

Status

This repository reflects the final evaluated research prototype used in the concept study and evaluation report.
The architecture is intentionally modular to support future extensions (e.g., alternative retrievers, GraphRAG-style components, additional rerankers).