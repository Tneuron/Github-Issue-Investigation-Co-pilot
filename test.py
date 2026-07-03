from dataclasses import dataclass
from graph_builder import GraphBuilder
from indexer import FaissIndexer
from context_builder import (
    ContextBuilder,
    ContextPlanner,
    PromptBuilder,
)
from llm import OllamaLLM
from investigator import Investigator
from retriever import HybridRetriever

@dataclass
class IssueContext:
    title: str
    body: str

indexer = FaissIndexer()
index, metadata = indexer.load()

builder = GraphBuilder()
graph = builder.build_graph(metadata)

retriever = HybridRetriever()
context_builder = ContextBuilder(graph)
prompt_builder = PromptBuilder(graph)

llm = OllamaLLM("Qwen2.5-Coder:7B")
planner = ContextPlanner(llm)
investigator = Investigator(llm, graph)

issue = IssueContext(
    title=(
        "FlashAttentionBackwardSm80: compute_softmax_scale_log2 overwrites softmax_scale with None → DSLRuntimeError"
    ),
    body="""
Summary

flash_bwd.py overwrites softmax_scale with None when score_mod is None, causing a DSLRuntimeError because the kernel requires a non-None Float32 for dK scaling.
Environment

    flash-attn-4: v4.0.0b16
    nvidia-cutlass-dsl: 4.5.2
    CUDA: 12.8
    GPU: NVIDIA GB10 (sm_121 — consumer Blackwell / DGX Spark)

Root Cause

In FlashAttentionBackwardSm80.__call__ (around line 440):

softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
    softmax_scale, score_mod, score_mod_type
)

utils.compute_softmax_scale_log2 returns (log2_value, None) when score_mod is None (the common case). This assignment clobbers the original Float32 value of softmax_scale with None.

Later in the kernel, softmax_scale is used unconditionally for dK scaling:

# kernel requires Float32, gets None → DSLRuntimeError

The error message is: DSLRuntimeError: argument expects Float32 but got NoneType.
Fix

Use a throwaway variable to preserve the original softmax_scale:

# flash_bwd.py ~line 440
# Old:
softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(...)
# New:
softmax_scale_log2, _ = utils.compute_softmax_scale_log2(...)

softmax_scale_log2 is needed for the kernel's log2 path; softmax_scale (the original Float32) must remain unchanged.
Verification

Applied locally on GB10 (SM_121). DSLRuntimeError for softmax_scale argument no longer occurs. MHA backward tests pass numerically.
Related

Found during SM_121 (GB10 consumer Blackwell) bringup. This bug affects all architectures using FlashAttentionBackwardSm80 when score_mod is None (standard attention without custom score modifiers).
""",
)
retrieval_query = f"""
{issue.title}

{issue.body}
"""
results = retriever.search(issue.title.strip(), k=3)
start_ids = []
for result in results:
    chunk = result["chunk"]
    node_id = builder.symbol_lookup.get(
        (
            chunk.file_path,
            chunk.name
        )
    )
    if node_id is not None:
        start_ids.append(node_id)

print("=" * 80)
print("RETRIEVER")
print("=" * 80)
for result in results:
    chunk = result["chunk"]
    print(
        f"{result['score']:.2f}",
        chunk.file_path,
        chunk.name
    )

context = context_builder.expand(
    start_ids,
    depth=0,
)

print("\nDirect Retrieved Context\n")
for node in context:
    print("-", node.chunk.file_path, node.chunk.name)
selected = planner.select_symbols(
    issue,
    context,
)
selected = [
    item
    for item in selected
    if item.node.chunk.chunk_type in ("function", "method")
]

print("\nPlanner Selected\n")
for item in selected:
    print(item.node.chunk.name)
    print("Reason:", item.reason)
    print()

expanded = []
visited = set()
MAX_INVESTIGATION_NODES = 6

def add_node(node):
    if node.chunk.id in visited:
        return False

    if len(expanded) >= MAX_INVESTIGATION_NODES:
        return False

    expanded.append(node)
    visited.add(node.chunk.id)
    return True

selected_ids = [item.node_id for item in selected]
selected_context = context_builder.expand(
    selected_ids,
    depth=0,
)
for node in selected_context:
    add_node(node)
if not expanded:
    for node in context:
        if node.chunk.chunk_type in ("function", "method"):
            add_node(node)


print("\nExpanded Investigation Context\n")
for node in expanded:
    print("-", node.chunk.file_path, node.chunk.name)


# ============================================================
# FIRST INVESTIGATION PASS
# ============================================================

prompt = prompt_builder.build(issue, expanded)

print("=" * 80)
print("PROMPT SIZE")
print("=" * 80)
print(len(prompt), "characters")
print(prompt.count("\n"), "lines")
print()


# ============================================================
# CONTROLLED FOLLOW-UP INVESTIGATION
#
# The LLM is allowed to ask for more context, but:
# - only exact symbol matches are added
# - only function/method nodes are added
# - context never grows beyond MAX_INVESTIGATION_NODES
# ============================================================

MAX_ROUNDS = 3
current_context = list(expanded)

for round_id in range(MAX_ROUNDS):
    prompt = prompt_builder.build(issue, current_context)
    result = investigator.investigate(prompt, current_context)

    if result.has_enough_evidence:
        break

    requested_symbols = []

    for item in result.additional_context_required:
        if isinstance(item, dict):
            symbol = (
                item.get("repository_symbol")
                or item.get("symbol")
                or item.get("name")
            )
        else:
            symbol = item

        if symbol:
            requested_symbols.append(symbol)

    if not requested_symbols:
        break

    existing_ids = {node.chunk.id for node in current_context}
    added_anything = False

    for requested_symbol in requested_symbols:
        for node in graph.values():
            if len(current_context) >= MAX_INVESTIGATION_NODES:
                break

            if node.chunk.id in existing_ids:
                continue

            if node.chunk.name != requested_symbol:
                continue

            if node.chunk.chunk_type not in ("function", "method"):
                continue

            current_context.append(node)
            existing_ids.add(node.chunk.id)
            added_anything = True
            break

    if not added_anything:
        break


print("\n" + "=" * 80)
print("📋 FINAL INVESTIGATION REPORT")
print("=" * 80)

print(f"Evidence Found : {result.has_enough_evidence}")
print(f"Confidence     : {result.confidence:.2f}")

print("\nSummary")
print("-" * 80)
print(result.summary)
print()

print("Root Cause")
print("-" * 80)
if isinstance(result.root_cause, dict):
    print(result.root_cause.get("reason", "No verified root cause."))
elif result.root_cause:
    print(result.root_cause)
else:
    print("No verified root cause.")

print("\nReasoning")
print("-" * 80)
print(result.reasoning)

print("\nRelevant Files")
print("-" * 80)
for file in result.relevant_files:
    print(f"• {file}")

print("\nRelevant Functions")
print("-" * 80)
for fn in result.relevant_functions:
    print(f"• {fn}")
print()

print("Execution Flow")
print("-" * 80)
if result.execution_flow:
    for index, step in enumerate(result.execution_flow, start=1):
        print(f"{index}. {step['repository_symbol']}")
        print(f"   {step['observation']}")
        if index < len(result.execution_flow):
            print("   ↓")
else:
    print("No verified execution flow.")

print("\nEvidence")
print("-" * 80)

if result.evidence:
    for item in result.evidence:
        symbol = (item.get("symbol")or item.get("repository_symbol"))
        observation = (item.get("observation") or item.get("fact_id")
        )
        print(f"• {symbol}")
        print(f"  {observation}")
        print()
else:
    print("No verified evidence.")


if result.additional_context_required:
    print("\nAdditional Context Required")
    print("-" * 80)
    for item in result.additional_context_required:
        if isinstance(item, dict):
            print(f"• {item['repository_symbol']}")
            print(f"  Reason : {item['reason']}")
        else:
            print(f"• {item}")
