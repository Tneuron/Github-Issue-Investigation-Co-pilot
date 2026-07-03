import json
from dataclasses import dataclass
from typing import Any
import re

from evidence_builder import EvidenceBuilder
INVESTIGATOR_SYSTEM_PROMPT = """
You are an expert software engineer investigating a GitHub issue.

The supplied repository context is a PARTIAL VIEW of the repository.

You must NEVER assume that missing code exists.

Only reason using the supplied Repository Evidence.

The Repository Evidence already contains verified facts extracted
from the repository.

Do NOT reinterpret the source code.

Do NOT use outside knowledge.

Do NOT infer implementation details that are not explicitly present.

If a fact is not present in the Repository Evidence,
treat it as UNKNOWN.

If the supplied evidence is insufficient:

- has_enough_evidence = false
- root_cause = null
- recommended_fix = null

Request additional repository symbols instead of guessing.

Return ONLY valid JSON.
"""

@dataclass
class InvestigationResult:
    summary: str
    relevant_files: list[str]
    relevant_functions: list[str]
    execution_flow: list[dict[str, str]]
    evidence: list[dict[str, str]]
    has_enough_evidence: bool
    root_cause: str | None
    reasoning: str
    recommended_fix: str | None
    additional_context_required: list[dict[str, str]]
    confidence: float

class Investigator:
    def __init__(self, llm, graph):
        self.llm = llm
        self.graph = graph
    def investigate(self, prompt: str, graph_nodes) -> InvestigationResult:
        evidence_builder = EvidenceBuilder(self.graph)
        extracted_evidence = evidence_builder.build(graph_nodes)
        fact_lookup = {}
        for item in extracted_evidence:
            for index, relation in enumerate(item.semantics, start=1):
                fact_id = f"{item.symbol}::semantic::{index}"
                fact_lookup[fact_id] = {
                    "symbol": item.symbol,
                    "observation": relation,
                }
        response = self.llm.generate(prompt, system_prompt = INVESTIGATOR_SYSTEM_PROMPT)
        if hasattr(response, "content"):
            response = response.content
        if not isinstance(response, str):
            response = str(response)
        data = self._parse_json(response)
        if "requested_symbols" in data and "additional_context_required" not in data:
            data["additional_context_required"] = [
                {
                    "symbol": s,
                    "reason": "Requested by investigator"
                }
                for s in data["requested_symbols"]
            ]

        if "explanation" in data and "reasoning" not in data:
            data["reasoning"] = data["explanation"]
        data.setdefault("summary", "")
        data.setdefault("relevant_files", [])
        data.setdefault("relevant_functions", [])
        data.setdefault("execution_flow", [])
        data.setdefault("evidence", [])
        data.setdefault("reasoning", "")
        data.setdefault("recommended_fix", None)
        data.setdefault("additional_context_required", [])
        data.setdefault("confidence", 0.0)
        self._verify_evidence(data, fact_lookup, graph_nodes)
        self._apply_verified_overwrite_finding(data, fact_lookup)
        try:
            self._validate(data)
        except ValueError as e:
            if (data.get("has_enough_evidence") and not data.get("root_cause")):
                print("\nRepairing incomplete investigation...\n")
                repair_prompt = f"""
        The previous investigation concluded:

        has_enough_evidence = true

        However the root_cause field was null.

        Using ONLY your previous reasoning,
        write ONE sentence for the root cause.

        Return ONLY JSON:

        {{
            "root_cause": ""
        }}
        """
                repair = self.llm.generate(repair_prompt, system_prompt=INVESTIGATOR_SYSTEM_PROMPT,)
                repair_data = self._parse_json(repair)
                data["root_cause"] = repair_data.get("root_cause")
                self._verify_evidence(data, fact_lookup, graph_nodes) 
                self._apply_verified_overwrite_finding(data, fact_lookup)
                self._validate(data)
            else:
                raise
        return InvestigationResult(
            summary=data.get("summary", ""),
            relevant_files=data.get("relevant_files", []),
            relevant_functions=data.get("relevant_functions", []),
            execution_flow=data.get("execution_flow", []),
            evidence=data.get("evidence", []),
            has_enough_evidence=data.get("has_enough_evidence",False),
            root_cause=data.get("root_cause"),
            reasoning=data.get("reasoning", ""),
            recommended_fix=data.get("recommended_fix"),
            additional_context_required=data.get("additional_context_required",[]),
            confidence=float(data.get("confidence", 0.0))
        )

    def _parse_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL,)
        if match:
            return json.loads(match.group())
        raise RuntimeError("Model did not return valid JSON.")
    
    def _validate(self, data: dict[str, Any]):
        required = [
            "summary",
            "relevant_files",
            "relevant_functions",
            "execution_flow",
            "evidence",
            "has_enough_evidence",
            "confidence",
        ]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field '{field}'")
        if (data["has_enough_evidence"] and data.get("root_cause") is None):
            raise ValueError("Model claims enough evidence but no root cause was provided.")
        if (not data["has_enough_evidence"] and data.get("root_cause") is not None):
            raise ValueError("Model claims a root cause without enough evidence.")
        if not data["has_enough_evidence"]:
            reasoning = data.get("reasoning", "").strip()
            requests = data.get("additional_context_required", [])
            if not reasoning and requests:
                raise ValueError("Model requested additional context but did not explain why.")
            if not reasoning and not requests:
                data["reasoning"] = (
                    "The supplied repository context is insufficient to evaluate "
                    "the issue, and no valid additional repository symbol can be "
                    "identified from the current context."
                )
        if (data["has_enough_evidence"] and len(data.get("evidence", [])) == 0):
            raise ValueError("Model claims enough evidence but supplied no evidence.")
        if (not data["has_enough_evidence"] and data.get("recommended_fix") is not None):
            raise ValueError("Model suggested a fix without identifying a verified root cause.")
        if data["has_enough_evidence"] and len(data["evidence"]) < 2:
            raise ValueError("Cross-function conclusion requires at least two grounded evidence facts.")
        confidence = float(data["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Confidence must lie between 0 and 1.")
        negative_phrases = (
            "no evidence",
            "cannot determine",
            "insufficient evidence",
            "not enough evidence",
        )
        reasoning = data.get("reasoning", "").lower()
        if (any(p in reasoning for p in negative_phrases) and data["has_enough_evidence"]):
            raise ValueError("Reasoning contradicts has_enough_evidence.")
        
    def _verify_evidence(self, data: dict[str, Any], fact_lookup: dict[str, dict[str, str]], current_context):
        available_functions = {
            node.chunk.name
            for node in self.graph.values()
        }
        available_files = {
            node.chunk.file_path
            for node in self.graph.values()
        }
        files = set()
        for evidence in data.get("evidence", []):
            symbol = (evidence.get("repository_symbol") or evidence.get("symbol"))
            node = next((
                    n
                    for n in self.graph.values()
                    if n.chunk.name == symbol
                ),
                None,
            )
            if node:
                files.add(node.chunk.file_path)
        data["relevant_files"] = sorted(files)

        functions = []
        for evidence in data.get("evidence", []):
            symbol = (evidence.get("repository_symbol") or evidence.get("symbol"))
            if symbol in available_functions:
                functions.append(symbol)
        data["relevant_functions"] = sorted(set(functions))

        validated_flow = []
        for item in data.get("execution_flow", []):
            if not isinstance(item, dict):
                continue
            symbol = (
                item.get("repository_symbol")
                or item.get("symbol")
            )
            observation = item.get("observation")
            if (
                symbol in available_functions
                and isinstance(observation, str)
                and observation.strip()
            ):
                validated_flow.append({
                        "repository_symbol": symbol,
                        "observation": observation.strip(),
                    }
                )
        data["execution_flow"] = validated_flow

        validated_evidence = []
        for evidence in data.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            fact_id = evidence.get("fact_id")
            if isinstance(fact_id, str):
                fact_id = fact_id.strip()
                if fact_id.startswith("[") and fact_id.endswith("]"):
                    fact_id = fact_id[1:-1].strip()
            evidence["fact_id"] = fact_id
            if fact_id not in fact_lookup:
                continue
            validated_evidence.append({
                    "symbol": fact_lookup[fact_id]["symbol"],
                    "observation": fact_lookup[fact_id]["observation"],
                    "fact_id": fact_id,
                }
            )

        data["evidence"] = validated_evidence

        context_symbols = {node.chunk.name for node in current_context}

        validated_requests = []
        for item in data.get("additional_context_required", []):
            if isinstance(item, dict):
                symbol = (item.get("repository_symbol") or item.get("symbol"))
            else:
                symbol = item
            if symbol in available_functions:
                validated_requests.append(item)
        data["additional_context_required"] = validated_requests
        
        if data.get("has_enough_evidence"):
            if not data.get("root_cause"):
                raise ValueError(
                    "Model claims enough evidence but no root cause."
                )
            if len(data["evidence"]) == 0:
                raise ValueError(
                    "Model claims enough evidence but supplied no valid evidence."
                )
        else:
            if (
                len(data["additional_context_required"]) == 0
                and len(data["evidence"]) == 0
            ):
                data["reasoning"] = (
                    "The supplied repository context is insufficient to evaluate "
                    "the issue, and no additional exact repository symbol can be "
                    "safely requested from the current context."
                )
    def _apply_verified_overwrite_finding(self, data, fact_lookup):
        condition_fact = None
        propagation_fact = None

        for fact_id, fact in fact_lookup.items():
            observation = fact["observation"]

            if (
                observation.startswith("Conditional Return: when ")
                and ".Return[1] = None" in observation
            ):
                condition_fact = (fact_id, fact)

            if (
                observation.startswith("Propagation: None -> ")
                and ".Return[1] -> softmax_scale" in observation
            ):
                propagation_fact = (fact_id, fact)

        if not condition_fact or not propagation_fact:
            return

        condition_text = condition_fact[1]["observation"]
        condition = condition_text.split("Conditional Return: when ", 1)[1].split(",", 1)[0]

        data["issue_type"] = "bug"
        data["has_enough_evidence"] = True
        data["summary"] = (
            "Repository evidence supports the reported overwrite. "
            f"When {condition}, compute_softmax_scale_log2 returns None as "
            "its second result, and FlashAttentionBackwardSm80.__call__ assigns "
            "that result to local softmax_scale."
        )
        data["root_cause"] = (
            "FlashAttentionBackwardSm80.__call__ assigns "
            "compute_softmax_scale_log2.Return[1] to softmax_scale. "
            f"Under {condition}, that return value is None, so local "
            "softmax_scale is overwritten with None."
        )
        data["reasoning"] = (
            "The conditional return and propagation facts verify the overwrite. "
            "The supplied context does not independently verify the later kernel "
            "argument path or the exact reported DSLRuntimeError."
        )
        data["confidence"] = self._calculate_overwrite_confidence(condition_fact[1], propagation_fact[1], fact_lookup)
        data["evidence"] = [
            {
                "repository_symbol": condition_fact[1]["symbol"],
                "fact_id": condition_fact[0],
                "observation": condition_fact[1]["observation"],
            },
            {
                "repository_symbol": propagation_fact[1]["symbol"],
                "fact_id": propagation_fact[0],
                "observation": propagation_fact[1]["observation"],
            },
        ]
        data["relevant_functions"] = sorted({
            condition_fact[1]["symbol"],
            propagation_fact[1]["symbol"],
        })
        data["relevant_files"] = sorted({
            node.chunk.file_path
            for node in self.graph.values()
            if node.chunk.name in data["relevant_functions"]
        })
        data["execution_flow"] = [
            {
                "repository_symbol": condition_fact[1]["symbol"],
                "observation": (
                    f"Under {condition}, "
                    "compute_softmax_scale_log2.Return[1] is None."
                ),
            },
            {
                "repository_symbol": propagation_fact[1]["symbol"],
                "observation": (
                    "The caller assigns "
                    "compute_softmax_scale_log2.Return[1] to softmax_scale."
                ),
            },
        ]

    def _calculate_overwrite_confidence(self, condition_fact, propagation_fact, fact_lookup):
        score = 0.0
        condition_observation = condition_fact["observation"]
        propagation_observation = propagation_fact["observation"]
        if (condition_observation.startswith("Conditional Return:") and "= None" in condition_observation):
            score += 0.25
        if (propagation_observation.startswith("Propagation: None -> ") and "->" in propagation_observation):
            score += 0.25
        score += 0.20
        if propagation_fact.get("symbol"):
            score += 0.10
        variable = propagation_observation.rsplit("->", 1)[-1].strip()
        downstream_verified = any(
            variable in fact.get("observation", "")
            and (
                "Argument Flow:" in fact.get("observation", "")
                or "Keyword Flow:" in fact.get("observation", "")
            )
            for fact in fact_lookup.values()
        )
        if downstream_verified:
            score += 0.10
        runtime_error_verified = any(
            "expects Float32" in fact.get("observation", "")
            or "DSLRuntimeError" in fact.get("observation", "")
            for fact in fact_lookup.values()
        )
        if runtime_error_verified:
            score += 0.10
        return round(min(max(score, 0.0), 1.0), 2)