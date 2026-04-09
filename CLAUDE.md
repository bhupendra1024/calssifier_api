# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Context Overlap Analyzer** — detects semantic overlap between image classification contexts using CLIP embeddings. The system helps identify overlapping or ambiguous classification labels in a text-to-image verification workflow (agricultural/solar domain).

This project is in the specification phase. The spec is in [specs.md](specs.md) and the architecture diagram is in [download.png](download.png).

## What to Build

### Core Pipeline
1. **Ingest** 13 classification contexts (key-value pairs: label → description text) from `specs.md §3.1`
2. **Embed** all context descriptions using a CLIP text encoder
3. **Compute** pairwise cosine similarity → produce a 13×13 similarity matrix
4. **Store** embeddings and similarity scores in PostgreSQL
5. **Visualize** the matrix in a UI (context × context heatmap)

### Classification Contexts (13 labels)
Solar Panel with Farmer, Foundation with Farmer, Controller with Farmer, Front side Panel module, Water Discharge with Farmer, Inside IMEI Code, Adhar Card, Pump Serial Number, Motor Serial Number, Controller Serial Number, Water filling with civit mat, Earthing and Lighting, Outside contour.

## Architecture

```
contexts (JSON)
    └─► CLIP text encoder → embeddings (pgvector in Postgres)
                                └─► pairwise cosine similarity
                                        └─► 13×13 matrix → UI heatmap
```

### Key Design Decisions from Spec
- **Local-first**: all processing runs locally, no external API calls
- **PostgreSQL** for storing embeddings (use `pgvector` extension) and similarity results
- **Output format**: similarity matrix suitable for graph plotting (rows and columns are context labels)
- **UI**: a context × context grid where cells show similarity scores (e.g., color-coded heatmap)

## Technology Choices (from spec)
- CLIP embeddings for semantic similarity (e.g., `openai/clip-vit-base-patch32` via `transformers` or `sentence-transformers`)
- PostgreSQL + `pgvector` extension for embedding storage
- Python is the natural fit for the ML pipeline

## Development Notes
- The input contexts live in `specs.md §3.1` as a JSON block — extract them from there or hardcode them into a config/constants file
- Similarity scores range [−1, 1]; cosine similarity between CLIP text embeddings is typically [0, 1]
- The system supports iterative refinement: context descriptions can be updated and re-embedded without rebuilding from scratch
