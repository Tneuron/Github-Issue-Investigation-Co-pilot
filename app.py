import argparse
import asyncio
from dataclasses import dataclass

from file_parser import FileParser
from chunker import StructuralChunker
from embedder import CodeEmbedder
from indexer import FaissIndexer
from graph_builder import GraphBuilder
from context_builder import (
    ContextBuilder,
    ContextPlanner,
    PromptBuilder,
)
from llm import OllamaLLM
from investigator import Investigator
from retriever import HybridRetriever
from github_mcp_client import GitHubMCPClient
from repo_manager import RepoManager

@dataclass
class IssueContext:
    title: str
    body: str

def parse_repo_name(repo_name: str) -> tuple[str, str]:
    if "/" not in repo_name:
        raise ValueError(
            "Repository must be in owner/repo format. "
            "Example: Dao-AILab/flash-attention"
        )
    owner, repo = repo_name.split("/", 1)
    if not owner or not repo:
        raise ValueError(
            "Repository must be in owner/repo format. "
            "Example: Dao-AILab/flash-attention"
        )
    return owner, repo

async def fetch_github_context(repo_name: str, issue_number: int,):
    owner, repo = parse_repo_name(repo_name)
    github = GitHubMCPClient()
    github_issue = await github.get_issue(
        owner=owner,
        repo=repo,
        issue_number=issue_number,
    )
    commit_sha = await github.get_repository_commit(
        owner=owner,
        repo=repo,
    )
    repo_manager = RepoManager()
    repo_path = repo_manager.prepare_repo(
        owner=owner,
        repo=repo,
        commit_sha=commit_sha,
    )
    issue = IssueContext(
        title=github_issue.title,
        body=github_issue.body,
    )

    return issue, commit_sha, repo_path


def ingest_repository(repo_path: str):
    print("\n" + "=" * 80)
    print("REPOSITORY INGESTION")
    print("=" * 80)
    print(f"Repository path: {repo_path}")

    parser = FileParser(str(repo_path))
    documents = parser.parse_repository()
    print(f"Documents: {len(documents)}")
    chunker = StructuralChunker()
    all_chunks = []
    for document in documents:
        chunks = chunker.chunk_document(document)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        raise ValueError("No chunks were created. Check FileParser and StructuralChunker.")

    embedder = CodeEmbedder()
    print("Embedding chunks...")
    embedded_chunks = embedder.embed_chunks(all_chunks)
    indexer = FaissIndexer()
    print("Saving FAISS index...")
    indexer.save(embedded_chunks)
    print("Repository ingestion complete.")
    return indexer


def build_fact_lookup(current_context):
    fact_lookup = {}
    for node in current_context:
        for fact in getattr(node, "facts", []):
            fact_lookup[fact.id] = {
                "observation": fact.observation,
            }
    return fact_lookup


def run_investigation(issue: IssueContext):

    print("\n" + "=" * 80)
    print("LOADING INVESTIGATION COMPONENTS")
    print("=" * 80)

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

    results = retriever.search(issue.title.strip(), k=3)
    start_ids = []
    for result in results:
        chunk = result["chunk"]
        node_id = builder.symbol_lookup.get((chunk.file_path, chunk.name))
        if node_id is not None:
            start_ids.append(node_id)

    print("\n" + "=" * 80)
    print("RETRIEVER")
    print("=" * 80)
    for result in results:
        chunk = result["chunk"]
        print(
            f"{result['score']:.2f}",
            chunk.file_path,
            chunk.name,
        )
    context = context_builder.expand(start_ids, depth=0,)

    print("\nDirect Retrieved Context\n")
    for node in context:
        print("-", node.chunk.file_path, node.chunk.name)
    selected = planner.select_symbols(issue, context)
    selected = [item for item in selected if item.node.chunk.chunk_type in ("function", "method")]

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
            return
        if len(expanded) >= MAX_INVESTIGATION_NODES:
            return
        expanded.append(node)
        visited.add(node.chunk.id)

    for item in selected:
        add_node(item.node)
    for item in selected:
        neighbours = context_builder.expand([item.node_id], depth=1)
        for node in neighbours:
            add_node(node)
    for node in context:
        add_node(node)

    print("\nExpanded Investigation Context\n")
    for node in expanded:
        print("-", node.chunk.file_path, node.chunk.name)

    MAX_ROUNDS = 3
    current_context = expanded
    for round_id in range(MAX_ROUNDS):
        print(f"\nInvestigation round: {round_id + 1}")
        prompt = prompt_builder.build(issue, current_context)
        result = investigator.investigate(prompt, current_context,)
        if result.has_enough_evidence:
            break

        requested = []
        for item in result.additional_context_required:
            if isinstance(item, dict):
                symbol = (item.get("repository_symbol") or item.get("symbol"))
                if symbol:
                    requested.append(symbol)
            elif isinstance(item, str):
                requested.append(item)
        if not requested:
            break

        ids = []
        for node in graph.values():
            if node.chunk.name in requested:
                ids.append(node.chunk.id)
        if not ids:
            break
        extra_nodes = context_builder.expand(ids, depth=0)
        existing_ids = {node.chunk.id for node in current_context}
        for node in extra_nodes:
            if node.chunk.id not in existing_ids:
                current_context.append(node)
                existing_ids.add(node.chunk.id)
    fact_lookup = build_fact_lookup(current_context)

    print("\n" + "=" * 80)
    print("FINAL INVESTIGATION REPORT")
    print("=" * 80)

    print(f"Evidence Found : {result.has_enough_evidence}")
    print(f"Confidence     : {result.confidence:.2f}")

    print("\nSummary")
    print("-" * 80)
    print(result.summary or "No verified summary.")
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
    print(result.reasoning or "No additional reasoning provided.")

    print("\nRelevant Files")
    print("-" * 80)
    if result.relevant_files:
        for file_path in result.relevant_files:
            print(f"• {file_path}")
    else:
        print("No verified relevant files.")

    print("\nRelevant Functions")
    print("-" * 80)
    if result.relevant_functions:
        for function_name in result.relevant_functions:
            print(f"• {function_name}")
    else:
        print("No verified relevant functions.")

    print("\nExecution Flow")
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
            symbol = (
                item.get("repository_symbol")
                or item.get("symbol")
            )
            fact_id = item.get("fact_id")
            observation = item.get("observation")
            if not observation and fact_id:
                fact = fact_lookup.get(fact_id, {})
                observation = fact.get("observation")
            print(f"• {symbol}")
            if observation:
                print(f"  {observation}")
            elif fact_id:
                print(f"  Fact ID: {fact_id}")
            print()
    else:
        print("No verified evidence.")

    if result.additional_context_required:
        print("\nAdditional Context Required")
        print("-" * 80)
        for item in result.additional_context_required:
            if isinstance(item, dict):
                symbol = (
                    item.get("repository_symbol")
                    or item.get("symbol")
                    or "Unknown symbol"
                )
                reason = item.get("reason", "No reason provided.")
                print(f"• {symbol}")
                print(f"  Reason: {reason}")
            else:
                print(f"• {item}")

def main():
    print("=" * 80)
    print("GitHub Issue Investigation Copilot")
    print("=" * 80)

    repo_name = input("\nEnter GitHub repository (owner/repo): ").strip()
    while not repo_name or "/" not in repo_name:
        print("Invalid format. Use owner/repo, "
            "for example: Dao-AILab/flash-attention")
        repo_name = input("Enter GitHub repository (owner/repo): ").strip()
    issue_input = input("Enter GitHub issue number: ").strip()
    while not issue_input.isdigit():
        print("Issue number must be a number, for example: 2566")
        issue_input = input("Enter GitHub issue number: ").strip()

    issue_number = int(issue_input)
    issue, commit_sha, repo_path = asyncio.run(
        fetch_github_context(
            repo_name=repo_name,
            issue_number=issue_number,
        )
    )

    print("\n" + "=" * 80)
    print("GITHUB ISSUE")
    print("=" * 80)
    print(f"Repository : {repo_name}")
    print(f"Issue      : #{issue_number}")
    print(f"Title      : {issue.title}")
    print(f"Commit SHA : {commit_sha}")
    print(f"Repo Path  : {repo_path}")

    ingest_repository(repo_path)
    run_investigation(issue)

if __name__ == "__main__":
    main()
