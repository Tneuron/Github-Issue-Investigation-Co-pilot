# GitHub Issue Investigation Copilot

An evidence-grounded AI agent that investigates GitHub issues against a repository’s source code.

Given a GitHub repository and issue number, it:

1. Fetches the issue through GitHub MCP
2. Clones the repository at a pinned commit
3. Parses and chunks the codebase
4. Builds a FAISS vector index
5. Retrieves code using the issue title
6. Expands relevant symbols through a code graph
7. Produces a report using verified repository facts

## What it can verify

This project currently works best for **static code issues**, such as:

* incorrect assignments
* `None` propagation
* wrong return-value usage
* missing condition handling
* incorrect function calls
* data-flow bugs between functions

It does not yet fully investigate runtime-only, CI, dependency, environment, distributed-system, or historical regression issues because those require external evidence such as logs, test runs, package metadata, CI output, or Git history.

## Architecture

```text
User input: owner/repo + issue number
        ↓
GitHub MCP Server
        ↓
Issue title + body
        ↓
Git clone at commit SHA
        ↓
FileParser → StructuralChunker → CodeEmbedder → FAISS
        ↓
HybridRetriever (issue title only)
        ↓
Graph expansion + verified semantic facts
        ↓
LLM Investigator
        ↓
Evidence-grounded investigation report
```

The issue title is used for retrieval. The issue body is used as investigation context, not as verified repository evidence.

## Project structure

```text
app.py                 # Main entry point
github_mcp_client.py   # MCP client for GitHub issue data
repo_manager.py        # Clones and checks out repositories

file_parser.py         # Reads repository files
chunker.py             # Creates structural code chunks
embedder.py            # Generates embeddings
indexer.py             # Saves/loads FAISS index
retriever.py           # Retrieves relevant code chunks
graph_builder.py       # Builds symbol/data-flow graph
context_builder.py     # Expands investigation context
investigator.py        # Produces verified issue report
llm.py                 # Ollama model wrapper
```

## Requirements

* Python 3.10+
* Docker
* Git
* Ollama
* GitHub personal access token

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the model:

```bash
ollama pull qwen2.5-coder:7b
```

## Docker setup

Docker runs the GitHub MCP server. Your Python app starts the MCP server as a temporary container and communicates with it through stdin/stdout.

Install Docker on Ubuntu:

```bash
sudo apt update
sudo apt install docker.io
```

Start Docker:

```bash
sudo systemctl enable --now docker
```

Allow your user to run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verify Docker:

```bash
docker --version
docker run hello-world
```

Do not run the Python project using `sudo`.

## GitHub token setup

Create a GitHub personal access token and add it to a `.env` file in the project root:

```env
GITHUB_TOKEN=your_token_here
```

For a classic token, use:

```text
repo
read:org
```

Add this to `.gitignore`:

```gitignore
.env
repositories/
vector_store/
data/
```

## Run

```bash
python3 app.py
```

Then enter:

```text
Enter GitHub repository (owner/repo): Dao-AILab/flash-attention
Enter GitHub issue number: 2566
```

The application will:

```text
fetch issue → clone repo → ingest code → build index → investigate issue
```

## Example github issue and report output

```text
================================================================================
GITHUB ISSUE
================================================================================
Repository : Dao-AILab/flash-attention
Issue      : #2651
Title      : FlashAttentionBackwardSm80: compute_softmax_scale_log2 overwrites softmax_scale with None → DSLRuntimeError
```

```text
================================================================================
FINAL INVESTIGATION REPORT
================================================================================
Evidence Found : True
Confidence     : 0.90

Summary
--------------------------------------------------------------------------------
Repository evidence supports the reported overwrite. When const_expr(score_mod is None), compute_softmax_scale_log2 returns None as its second result, and FlashAttentionBackwardSm80.__call__ assigns that result to local softmax_scale.

Root Cause
--------------------------------------------------------------------------------
FlashAttentionBackwardSm80.__call__ assigns compute_softmax_scale_log2.Return[1] to softmax_scale. Under const_expr(score_mod is None), that return value is None, so local softmax_scale is overwritten with None.

Reasoning
--------------------------------------------------------------------------------
The conditional return and propagation facts verify the overwrite. The supplied context does not independently verify the later kernel argument path or the exact reported DSLRuntimeError.

Relevant Files
--------------------------------------------------------------------------------
• flash_attn/cute/flash_fwd_sm90.py
• flash_attn/cute/utils.py

Relevant Functions
--------------------------------------------------------------------------------
• FlashAttentionForwardSm90.__call__
• compute_softmax_scale_log2

Execution Flow
--------------------------------------------------------------------------------
1. compute_softmax_scale_log2
   Under const_expr(score_mod is None), compute_softmax_scale_log2.Return[1] is None.
   ↓
2. FlashAttentionForwardSm90.__call__
   The caller assigns compute_softmax_scale_log2.Return[1] to softmax_scale.

Evidence
--------------------------------------------------------------------------------
• compute_softmax_scale_log2
  Conditional Return: when const_expr(score_mod is None), compute_softmax_scale_log2.Return[1] = None

• FlashAttentionForwardSm90.__call__
  Propagation: None -> compute_softmax_scale_log2.Return[1] -> softmax_scale
```

## Current limitations

The agent does not yet collect:

* GitHub Actions logs
* runtime traces
* dependency/version conflicts
* environment details
* commit history and regression diffs
* distributed-system traces

These are planned future extensions through additional MCP tools and evidence collectors.

## Future work

* Fetch issue comments
* Fetch linked pull requests
* Fetch CI logs
* Support repository-specific FAISS indexes
* Add Git history and regression analysis
* Add runtime/log evidence parsing
* Add a web interface

