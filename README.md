# GitHub Issue Investigation Copilot

An evidence-grounded copilot for investigating GitHub issues against a repository’s source code.

Instead of treating an issue as a prompt and generating a speculative answer, this project retrieves relevant repository symbols, extracts static semantic relationships, validates evidence references, and produces a report that distinguishes between:

* **verified repository findings**
* **unverified issue claims**
* **missing evidence required for a stronger conclusion**

The system is designed to be conservative: if the repository context cannot prove a root cause, it reports insufficient evidence rather than inventing one.

---

## Why this project?

GitHub issues often contain suspected causes, stack traces, proposed fixes, and environment details. Those claims may be correct, but they are not proof.

This project investigates an issue using repository evidence:

```text
GitHub issue
    ↓
Hybrid retrieval over repository symbols
    ↓
Graph-based context expansion
    ↓
Static semantic and data-flow fact extraction
    ↓
Evidence validation
    ↓
Deterministic findings + grounded report
```

The result is not just “an LLM opinion.” It is a report tied to extracted repository facts.

---

## Current Capabilities

### 1. Hybrid code retrieval

The issue title is used to retrieve relevant repository symbols through a hybrid retrieval layer. The retriever identifies likely functions, methods, classes, and files involved in the issue.

### 2. Repository graph construction

The project builds a graph over indexed code symbols and their relationships. This allows the investigation pipeline to expand from directly retrieved symbols into nearby callers, callees, helpers, and related implementation context.

### 3. Static semantic fact extraction

The evidence layer extracts facts such as:

```text
Data Flow:
source → target

Conditional Return:
when condition, function.Return[index] = value

Argument Flow:
variable → called_function.Argument[index]

Keyword Flow:
variable → called_function.keyword_argument
```

These facts are treated as verified repository evidence.

### 4. Evidence-grounded investigation

The investigator validates every evidence reference returned by the LLM. A report cannot cite symbols or fact IDs that are not present in the supplied repository context.

### 5. Deterministic verified findings

For supported static patterns, the project creates findings deterministically instead of allowing the LLM to freely rewrite critical technical details.

Current supported pattern:

```text
conditional helper return
    ↓
returned value propagates into caller local variable
    ↓
verified local overwrite finding
```

For example:

```text
Conditional Return:
when const_expr(score_mod is None),
compute_softmax_scale_log2.Return[1] = None

Propagation:
None → compute_softmax_scale_log2.Return[1] → softmax_scale
```

This allows the system to verify that `softmax_scale` is overwritten with `None`, while still stating that a later runtime error is not independently verified unless the full downstream path is present.

### 6. Evidence-completeness confidence

Confidence is based on the completeness of verified evidence, not on an LLM-generated guess.

A finding with a verified local overwrite but no verified downstream runtime path receives lower confidence than one where the full causal chain is available.

---

## Example Investigation Output

```text
Evidence Found : True
Confidence     : 0.88

Summary
Repository evidence supports the reported overwrite. When
const_expr(score_mod is None), compute_softmax_scale_log2 returns None in
return slot 1, and FlashAttentionBackwardSm80.__call__ assigns that result
to local softmax_scale.

Root Cause
FlashAttentionBackwardSm80.__call__ assigns
compute_softmax_scale_log2.Return[1] to softmax_scale. Under
const_expr(score_mod is None), that return value is None, so local
softmax_scale is overwritten with None.

Reasoning
The conditional return and propagation facts verify the overwrite. The
supplied context does not independently verify the later kernel argument path
or the exact reported runtime error.

Evidence
• compute_softmax_scale_log2
  Conditional Return: when const_expr(score_mod is None),
  compute_softmax_scale_log2.Return[1] = None

• FlashAttentionBackwardSm80.__call__
  Propagation: None → compute_softmax_scale_log2.Return[1] → softmax_scale
```

---

## What It Works Best For

The current version is strongest for issues whose cause can be verified from static source-code relationships inside the indexed repository:

* incorrect return-value propagation
* `None` or null overwrites
* incorrect tuple unpacking
* wrong argument propagation
* local state mutation mistakes
* missing branch handling visible in source
* direct caller/callee API misuse
* defects where both the cause and effect path exist in the repository

---

## Current Limitations

The project does not claim to solve every GitHub issue from source code alone.

It may provide only partial findings or insufficient-evidence reports for issues involving:

* environment variables and runtime configuration
* GPU, driver, CUDA, or hardware-specific behavior
* filesystem caches and cross-process behavior
* race conditions and distributed systems
* external dependency internals
* build, linker, ABI, packaging, or CI-only failures
* performance regressions requiring benchmark data
* historical regressions requiring commit comparison
* bugs requiring logs, reproduction inputs, or runtime traces

For these cases, the correct output is not a hallucinated root cause. The system should identify relevant repository entry points and explain what additional evidence is required.

---

## Architecture

```text
┌─────────────────────┐
│ GitHub Issue Input  │
│ title + body        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hybrid Retriever    │
│ FAISS + symbol data │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Repository Graph    │
│ context expansion   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Context Planner     │
│ selects symbols     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Evidence Builder    │
│ semantic facts      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Investigator        │
│ validates evidence  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Finding Engine      │
│ deterministic rules │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Investigation Report│
└─────────────────────┘
```

---

## Project Structure

```text
.
├── chunker.py              # Extracts repository code chunks and symbols
├── indexer.py              # Builds and loads FAISS vector indexes
├── retriever.py            # Hybrid retrieval over repository chunks
├── graph_builder.py        # Builds graph relationships between symbols
├── context_builder.py      # Context expansion, planning, and prompt creation
├── evidence_builder.py     # Extracts static semantic and data-flow facts
├── investigator.py         # Validates evidence and builds investigation results
├── llm.py                  # Ollama LLM wrapper
├── test.py                 # Local investigation runner
├── data/
│   └── chunks.pkl
└── vector_store/
    └── code.index
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv giisinco
source giisinco/bin/activate
```

On Windows:

```bash
giisinco\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install [Ollama](https://ollama.com?utm_source=chatgpt.com), then pull the model used by the project:

```bash
ollama pull qwen2.5-coder:7b
```

Start the Ollama server if it is not already running:

```bash
ollama serve
```

---

## Running an Investigation

Update the issue title and body in `test.py`, then run:

```bash
python3 test.py
```

The runner prints:

1. retrieved symbols
2. planner-selected symbols
3. expanded investigation context
4. prompt size
5. raw LLM output
6. final validated investigation report

---

## Planned MCP Integration

The next major milestone is Model Context Protocol integration.

Instead of manually entering issue text and indexing a repository locally, the system will accept:

```text
repository: owner/repository
issue number: 123
```

MCP tools will retrieve:

* issue title and body
* labels and comments
* linked pull requests
* repository metadata
* default branch
* commit SHA used for analysis
* repository tree
* relevant source files on demand

Planned flow:

```text
User enters repository + issue number
    ↓
MCP GitHub tools fetch issue and repository metadata
    ↓
Repository is pinned to a commit SHA
    ↓
Code is indexed and investigated
    ↓
Evidence-grounded report is returned
```

Pinning the investigation to a commit SHA is important because issue discussions may refer to code that differs from the latest branch.

---

## Future Work

* MCP-based GitHub issue and repository ingestion
* commit-SHA-pinned investigations
* Git history support for regression analysis
* dependency-aware investigation boundaries
* runtime log and stack-trace ingestion
* CI failure analysis
* benchmark-aware performance investigation
* broader deterministic finding patterns
* evaluation dataset across repositories and issue categories
* web interface and API deployment

---

## Design Principles

* **Evidence before explanation**
  A root cause should be supported by repository facts.

* **No fabricated citations**
  The LLM cannot cite symbols or facts that are absent from the supplied context.

* **Deterministic handling for critical findings**
  Conditions, variable names, return slots, and confidence should not be altered by free-form generation.

* **Useful uncertainty**
  “Insufficient evidence” is a valid and useful result when source code alone cannot verify a runtime or dependency-level claim.

* **Reproducibility**
  Future MCP-based investigations will pin reports to a repository commit SHA.

---

## Requirements

See `requirements.txt` for the Python dependencies.

---

