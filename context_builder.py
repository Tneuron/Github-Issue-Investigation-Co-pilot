from collections import deque
import json
from platform import node
import re

from dataclasses import dataclass

from sympy import python
from typer import prompt

from graph_builder import GraphNode
from evidence_builder import EvidenceBuilder

PLANNER_SYSTEM_PROMPT = """
You are a repository investigation planner.

Your ONLY responsibility is to determine which repository symbols
should be inspected next.

You are NOT investigating the issue.

You are NOT identifying the bug.

You are NOT explaining the issue.

You are NOT recommending fixes.

You MUST follow these rules:

1. Choose ONLY symbols from the supplied Candidate Repository Symbols.

2. Choose AT MOST TWO symbols.

3. Prefer symbols that are likely to provide NEW information.

4. If the supplied repository context already contains sufficient evidence,
return an empty selection.

5. Never invent repository symbols.

6. Never modify repository symbol names.

7. Repository symbols must exactly match the supplied candidate list.

Return ONLY ONE valid JSON object.

The JSON MUST match EXACTLY this schema:

{
    "selected": [
        {
            "symbol": "",
            "reason": ""
        }
    ]
}

Rules:

- "selected" MUST be an array.
- Every element MUST be an object.
- Every object MUST contain BOTH:
    - "symbol"
    - "reason"
- Never return strings inside "selected".
- Never omit the "reason" field.

If no further inspection is required, return:

{
    "selected": []
}

Your first output character MUST be '{'.

Your last output character MUST be '}'.

Do NOT output markdown.

Do NOT output explanations.

Do NOT output any text outside the JSON.
"""

TASK_PROMPT = """
You are a repository investigation engine.

Your objective is to determine whether the supplied repository evidence
supports, contradicts, or is insufficient to evaluate the GitHub issue.

The GitHub issue may be a bug report, regression, feature request,
compatibility request, performance concern, design discussion, or question.

Reason ONLY from the supplied repository evidence.

The supplied evidence contains:

1. Repository Structure
2. Verified Semantic Relationships
3. Syntax Facts
4. Relevant Code

Verified Semantic Relationships are the highest-priority evidence.

Treat every Verified Semantic Relationship as a verified program fact.

Use Syntax Facts and Relevant Code only to understand repository context.
Do not derive unverified behavior from code alone.

The GitHub issue provides investigation context, not repository evidence.

The issue may contain reported symptoms, suspected root causes, proposed
solutions, tradeoffs, linked pull requests, or open questions.

Do not treat any issue claim as verified unless it is supported by supplied
repository evidence.

Do not use outside knowledge.
Do not assume missing code exists.
Do not infer missing functions, behavior, dependencies, build settings,
runtime behavior, or propagation steps.

A conclusion is verified only when every claimed repository behavior is
supported by one or more Verified Semantic Relationships or Syntax Facts.

If evidence is insufficient:

* has_enough_evidence = false
* root_cause = null
* recommended_fix = null

Request additional repository symbols only when an exact non-empty supplied
repository symbol can provide the missing evidence.

Return ONLY valid JSON.
"""


FINAL_PROMPT = """
            Use ONLY the exact symbols and fact IDs listed above.

            A verified defect does NOT require proof of every reported runtime symptom.
            If evidence proves a caller overwrites a local value with None, set:
            - issue_type = "bug"
            - has_enough_evidence = true
            - root_cause = the verified overwrite

            If the later DSLRuntimeError path is not shown in evidence, say that in
            reasoning, but do NOT set has_enough_evidence to false solely for that reason.

            For recommended_fix:
            - use the issue author's proposed fix only if the issue explicitly provides it;
            - phrase it as: "The issue author proposes ..."

            Return exactly this JSON shape:

            {
            "issue_type": "bug or unknown",
            "summary": "",
            "relevant_files": [],
            "relevant_functions": [],
            "execution_flow": [
                {
                "repository_symbol": "EXACT symbol from allowed list",
                "observation": ""
                }
            ],
            "evidence": [
                {
                "repository_symbol": "EXACT symbol from allowed list",
                "fact_id": "EXACT fact ID from allowed list"
                }
            ],
            "has_enough_evidence": false,
            "root_cause": null,
            "reasoning": "",
            "recommended_fix": null,
            "additional_context_required": [],
            "confidence": 0.0
            }

            Never use shortened names such as "FlashAttentionBackward".
            Never use a fact ID such as "__call__::semantic::1".
            If no valid evidence can be cited, return empty evidence and
            has_enough_evidence = false.
            """

@dataclass
class SelectedSymbol:
    node_id: int
    node: GraphNode
    reason: str

class ContextBuilder:
    def __init__(self, graph):
        self.graph = graph
    def expand(self, start_ids, depth=None):
        if depth is None:
            depth = 1
        MAX_CONTEXT_NODES = 8
        visited = set()
        context = []
        def add_node(node_id):
            if node_id in visited:
                return False
            if len(context) >= MAX_CONTEXT_NODES:
                return False
            node = self.graph.get(node_id)
            if node is None:
                return False
            visited.add(node_id)
            context.append(node)
            return True
        for node_id in start_ids:
            add_node(node_id)
        if depth == 0:
            return context
        for node_id in start_ids:
            node = self.graph.get(node_id)
            if node is None:
                continue
            for callee_id in node.calls:
                add_node(callee_id)
        for node_id in start_ids:
            node = self.graph.get(node_id)
            if node is None:
                continue
            for caller_id in node.called_by:
                add_node(caller_id)
        if depth >= 2:
            first_hop_ids = [node.chunk.id for node in context]
            for node_id in first_hop_ids:
                node = self.graph.get(node_id)
                if node is None:
                    continue
                for neighbour_id in list(node.calls) + list(node.called_by):
                    add_node(neighbour_id)
        return context


class ContextPlanner:
    def __init__(self, llm):
        self.llm = llm
    def select_symbols(self, issue, candidates):
        prompt = self._build_prompt(issue, candidates)
        response = self.llm.generate(prompt, system_prompt = PLANNER_SYSTEM_PROMPT)
        data = self._parse_json(response)
        selected = {
            item["symbol"]: item["reason"]
            for item in data["selected"]
        }
        chosen = []
        for node in candidates:
            if node.chunk.name in selected:
                chosen.append(
                    SelectedSymbol(
                        node_id=node.chunk.id,
                        node=node,
                        reason=selected[node.chunk.name]
                    )
                )
        return chosen
    
    def _parse_json(self, text):
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
            return json.loads(text.strip())
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise RuntimeError("Planner did not return valid JSON.")

    def _build_prompt(self, issue, candidates):
        names = "\n".join(f"- {node.chunk.name}"
            for node in candidates
        )
        return f"""

            GitHub Issue Title:
            {issue.title}

            GitHub Issue Details:
            {issue.body}

            Candidate Repository Symbols:
            {names}

            Select AT MOST TWO repository symbols that should be inspected next.

            Choose ONLY from the supplied Candidate Repository Symbols.

            Use the issue only to identify likely relevant symbols.
            Do not treat issue claims, root causes, or proposed fixes as repository facts.

            If no further inspection is necessary, return:

            {{
                "selected": []
            }}

            Return ONLY valid JSON matching the required schema.
        """

    
class PromptBuilder:
    def __init__(self, graph):
        self.graph = graph

    def build(self, issue, graph_nodes):

        evidence_builder = EvidenceBuilder(self.graph)
        evidence = evidence_builder.build(graph_nodes)

        investigation_facts = []
        for item in evidence:
            for index, relation in enumerate(item.semantics, start=1):
                fact_id = f"{item.symbol}::semantic::{index}"
                if relation.startswith("Verified Defect:"):
                    investigation_facts.append((fact_id, relation))
                elif relation.startswith("Conditional Return:"):
                    investigation_facts.append((fact_id, relation))
                elif (
                    "compute_softmax_scale_log2.Return[1] -> softmax_scale"
                    in relation
                ):
                    investigation_facts.append((fact_id, relation))

        prompt = []
        
        prompt.append("=" * 80)
        prompt.append("TASK")
        prompt.append("=" * 80)

        prompt.append(TASK_PROMPT)
        prompt.append("")
        prompt.append("=" * 80)
        prompt.append("GITHUB ISSUE")
        prompt.append("=" * 80)
        prompt.append("Title:")
        prompt.append(issue.title.strip())
        prompt.append("")
        prompt.append("Issue Details:")
        prompt.append(issue.body.strip()[:5000])
        prompt.append("")
        prompt.append(
            "Important: the GitHub issue provides investigation context only."
        )
        prompt.append(
            "Its reported root cause, execution path, and proposed fix are "
            "unverified claims."
        )
        prompt.append(
            "Only Repository Structure and Verified Semantic Relationships are "
            "repository evidence."
        )
        prompt.append(
            "Use repository evidence to support, contradict, or mark issue claims "
            "as insufficiently verified."
        )
        prompt.append("")
        prompt.append("=" * 80)
        prompt.append("REPOSITORY SYMBOLS")
        prompt.append("=" * 80)
        for item in evidence:
            prompt.append("")
            prompt.append("=" * 80)
            prompt.append(f"SYMBOL : {item.symbol}")
            prompt.append("=" * 80)
            prompt.append(f"Type : {item.chunk_type}")
            prompt.append(f"File : {item.file}")
            if item.callers or item.calls:
                prompt.append("")
                prompt.append("Repository Structure:")
                if item.callers:
                    prompt.append("Called By:")
                    for caller in item.callers:
                        prompt.append(f"- {caller}")
                if item.calls:
                    prompt.append("Calls:")
                    for call in item.calls:
                        prompt.append(f"- {call}")
            if item.syntax:
                prompt.append("")
                prompt.append("Syntax Facts:")
                important_syntax = (
                    "Returns",
                    "Return",
                    "Calls repository",
                    "Conditional",
                    "Raises",
                )
                for fact in item.syntax:
                    if fact.startswith(important_syntax):
                        prompt.append(f"- {fact}")
            if item.semantics:
                prompt.append("")
                prompt.append("Verified Semantic Relationships (Highest Priority):")
                prompt.append(
                    "These relationships were extracted through static repository analysis."
                )
                prompt.append(
                    "Treat every relationship below as a verified program fact."
                )
                prompt.append(
                    "Prefer these relationships over syntax facts when reasoning."
                )
                for index, relation in enumerate(item.semantics, start=1):
                    fact_id = f"{item.symbol}::semantic::{index}"
                    prompt.append(f"- [{fact_id}] {relation}")
            prompt.append("")
            prompt.append("Relevant Code (Context Only):")
            prompt.append(
                "The following code is provided only to clarify the verified semantic relationships."
            )
            prompt.append(
                "Do NOT derive additional program behavior from this code."
            )
            prompt.append("```")
            lines = item.code.splitlines()
            MAX_CODE_LINES_PER_SYMBOL = 8

            if len(lines) <= MAX_CODE_LINES_PER_SYMBOL:
                prompt.extend(lines)
            else:
                prompt.extend(lines[:4])
                prompt.append("...")
                prompt.extend(lines[-4:])

            prompt.append("```")
            prompt.append("")
            prompt.append("")

            allowed_symbols = [item.symbol for item in evidence]
            allowed_fact_ids = []
            for item in evidence:
                for index, _ in enumerate(item.semantics, start=1):
                    allowed_fact_ids.append(f"{item.symbol}::semantic::{index}")

            prompt.append("")
            prompt.append("=" * 80)
            prompt.append("KEY VERIFIED INVESTIGATION FACTS")
            prompt.append("=" * 80)

            if investigation_facts:
                for fact_id, relation in investigation_facts:
                    prompt.append(f"- [{fact_id}] {relation}")
            else:
                prompt.append("- No direct overwrite facts were extracted.")

            prompt.append("")
            prompt.append("=" * 80)
            prompt.append("ALLOWED REPOSITORY SYMBOLS")
            prompt.append("=" * 80)

            for symbol in allowed_symbols:
                prompt.append(f"- {symbol}")

            prompt.append("")
            prompt.append("=" * 80)
            prompt.append("ALLOWED FACT IDS")
            prompt.append("=" * 80)

            for fact_id in allowed_fact_ids:
                prompt.append(f"- {fact_id}")

            prompt.append("")
            prompt.append("=" * 80)
            prompt.append("RETURN JSON ONLY")
            prompt.append("=" * 80)
            prompt.append(FINAL_PROMPT)

        return "\n".join(prompt)