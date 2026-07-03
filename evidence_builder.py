from dataclasses import dataclass
from collections import defaultdict
from graph_builder import GraphNode
import ast
import re

@dataclass
class Evidence:
    symbol: str
    file: str
    chunk_type: str
    calls: list[str]
    callers: list[str]
    imports: list[str]
    syntax: list[str]
    semantics: list[str]
    code: str

class EvidenceBuilder:
    def __init__(self, graph):
        self.graph = graph

    def _canonical_symbol(self, expression):
        available_symbols = {node.chunk.name for node in self.graph.values()}
        if expression in available_symbols:
            return expression
        short_name = expression.split(".")[-1]
        if short_name in available_symbols:
            return short_name
        return expression
    
    def build(self, graph_nodes):
        evidence = []
        for node in graph_nodes:
            chunk = node.chunk
            syntax = []
            syntax.extend(self._graph_observations(node))
            syntax.extend(self._extract_syntax(node))
            semantics = self._extract_semantics(node)
            syntax = list(dict.fromkeys(syntax))
            semantics = list(dict.fromkeys(semantics))
            syntax_priority = (
                "Parameters",
                "Returns",
                "Return",
                "Calls repository",
                "Calls function",
                "Argument",
                "Conditional",
                "Raises",
                "Reads",
                "Imports",
            )
            syntax.sort(key=lambda x: next((i for i, p in enumerate(syntax_priority) if x.startswith(p)), 999))
            evidence.append(
                Evidence(
                    symbol=chunk.name,
                    file=chunk.file_path,
                    chunk_type=chunk.chunk_type,
                    calls=[
                        self.graph[x].chunk.name
                        for x in sorted(node.calls)
                    ],
                    callers=[
                        self.graph[x].chunk.name
                        for x in sorted(node.called_by)
                    ],
                    imports=sorted(node.imports),
                    syntax=syntax,
                    semantics=semantics,
                    code=chunk.code
                )
            )

        all_semantics = []
        for item in evidence:
            for relation in item.semantics:
                all_semantics.append((item.symbol, relation))
        edge_owner = {}
        plain_semantics = []
        for symbol, relation in all_semantics:
            plain_semantics.append(relation)
            if not relation.startswith("Data Flow:"):
                continue
            flow = relation[len("Data Flow:"):].strip()
            if "->" not in flow:
                continue
            source, target = [
                value.strip()
                for value in flow.split("->", 1)
            ]
            edge_owner[(source, target)] = symbol
        propagation_facts = self._build_propagation_facts(plain_semantics)
        for propagation in propagation_facts:
            flow = propagation[len("Propagation:"):].strip()
            parts = [part.strip() for part in flow.split("->")]
            if len(parts) != 3:
                continue
            source, middle, target = parts
            owner = edge_owner.get((middle, target))
            if owner is None:
                continue
            for item in evidence:
                if item.symbol == owner:
                    item.semantics.append(propagation)
                    break
        verified_defects = self._build_verified_defect_facts(evidence)
        for item in evidence:
            item.semantics.extend(verified_defects.get(item.symbol, []))
            item.semantics = list(dict.fromkeys(item.semantics))
        return evidence

    def _extract_syntax(self, node):
        observations = []
        try:
            tree = ast.parse(node.chunk.code)
        except Exception:
            return observations
        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, n):
                if n.args.args:
                    observations.append(
                        "Parameters: " +
                        ", ".join(arg.arg for arg in n.args.args)
                    )
                self.generic_visit(n)
            def visit_Return(self, n):
                try:
                    if n.value is None:
                        observations.append("Returns nothing.")
                    elif isinstance(n.value, ast.Tuple):
                        observations.append(f"Returns tuple with {len(n.value.elts)} values.")
                        for idx, value in enumerate(n.value.elts):
                            observations.append(f"Return[{idx}] = {ast.unparse(value)}")
                    else:
                        observations.append(f"Returns: {ast.unparse(n.value)}")
                except Exception:
                    pass

            def visit_Assign(self, n):
                try:
                    if (len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple)):
                        observations.append("Tuple assignment")
                    else:
                        for target in n.targets:
                            observations.append(f"Assignment: {ast.unparse(target)}")
                except Exception:
                    pass
                self.generic_visit(n)

            def visit_Call(self, n):
                try:
                    func = ast.unparse(n.func)
                    observations.append(f"Calls function: {func}")
                    for idx, arg in enumerate(n.args):
                        observations.append(f"Argument[{idx}] = {ast.unparse(arg)}")
                except Exception:
                    pass
                self.generic_visit(n)
            def visit_If(self, n):
                try:
                    observations.append(f"Conditional: {ast.unparse(n.test)}")
                except Exception:
                    pass
                self.generic_visit(n)
            def visit_Raise(self, n):
                try:
                    if n.exc:
                        observations.append(f"Raises: {ast.unparse(n.exc)}")
                except Exception:
                    pass
                self.generic_visit(n)
            def visit_Attribute(self, n):
                try:
                    text = ast.unparse(n)
                    if text.startswith("self."):
                        observations.append(f"Reads: {text}")
                except Exception:
                    pass
                self.generic_visit(n)
        Visitor().visit(tree)
        return observations
    
    def _extract_semantics(self, node):
        semantics = []
        variable_sources = {}
        try:
            tree = ast.parse(node.chunk.code)
        except Exception:
            return semantics
        class Visitor(ast.NodeVisitor):
            def __init__(self, outer):
                self.outer = outer 
                self.current_function = None 
                self.condition_stack = []
            def visit_FunctionDef(self, n):
                self.current_function = n.name
                self.generic_visit(n)
            def visit_Assign(self, n):
                try:
                    if (len(n.targets) == 1 and isinstance(n.targets[0], (ast.Tuple, ast.List)) and isinstance(n.value, ast.Call)):
                        func = self.outer._canonical_symbol(ast.unparse(n.value.func))
                        for idx, target in enumerate(n.targets[0].elts):
                            target_name = ast.unparse(target)
                            if target_name in variable_sources:
                                semantics.append(
                                    f"Overwrite: "
                                    f"{target_name} "
                                    f"(previously {variable_sources[target_name]})"
                                )
                            variable_sources[target_name] = (f"{func}.Return[{idx}]")
                            semantics.append(
                                f"Data Flow: "
                                f"{func}.Return[{idx}]"
                                f" -> "
                                f"{target_name}"
                            )
                    elif (len(n.targets) == 1 and isinstance(n.value, ast.Call)):
                        func = self.outer._canonical_symbol(ast.unparse(n.value.func))
                        target = ast.unparse(n.targets[0])
                        if target in variable_sources:
                            semantics.append(
                                f"Overwrite: "
                                f"{target} "
                                f"(previously {variable_sources[target]})"
                            )
                        variable_sources[target] = f"{func}.Return"
                        semantics.append(f"Data Flow: {func}.Return -> {target}")
                    elif (len(n.targets) == 1 and isinstance(n.value, ast.Name)):
                        source = n.value.id
                        target = ast.unparse(n.targets[0])
                        if source == target:
                            return
                        if target in variable_sources:
                            semantics.append(
                                f"Overwrite: "
                                f"{target} "
                                f"(previously {variable_sources[target]})"
                            )
                        if source in variable_sources:
                            variable_sources[target] = variable_sources[source]
                        else:
                            variable_sources[target] = source
                        semantics.append(
                            f"Data Flow: "
                            f"{source}"
                            f" -> "
                            f"{target}"
                        )
                except Exception:
                    pass
                self.generic_visit(n)
            def visit_Call(self, n):
                try:
                    func = self.outer._canonical_symbol(ast.unparse(n.func))
                    for idx, arg in enumerate(n.args):
                        if (
                            isinstance(arg, ast.Name)
                            and arg.id in variable_sources
                        ):
                            semantics.append(
                                f"Data Flow: "
                                f"{variable_sources[arg.id]}"
                                f" -> "
                                f"{func}.Argument[{idx}]"
                            )
                        else:
                            semantics.append(
                                f"Argument Flow: "
                                f"{ast.unparse(arg)}"
                                f" -> "
                                f"{func}.Argument[{idx}]"
                            )
                    for kw in n.keywords:
                        if kw.arg is None:
                            continue
                        value = ast.unparse(kw.value)
                        if (
                            isinstance(kw.value, ast.Name)
                            and value in variable_sources
                        ):
                            value = variable_sources[value]
                        semantics.append(
                            f"Keyword Flow: "
                            f"{value}"
                            f" -> "
                            f"{func}.{kw.arg}"
                        )
                except Exception:
                    pass
                self.generic_visit(n)
            def visit_Return(self, n):
                try:
                    if self.current_function is None:
                        return
                    if isinstance(n.value, ast.Name):
                        source = n.value.id
                        if source in variable_sources:
                            source = variable_sources[source]
                        semantics.append(
                            f"Data Flow: {source} -> {self.current_function}.Return"
                        )
                    elif isinstance(n.value, ast.Tuple):
                        for idx, value in enumerate(n.value.elts):
                            source = ast.unparse(value)
                            if (isinstance(value, ast.Name)and source in variable_sources):
                                source = variable_sources[source]
                            semantics.append(
                                f"Data Flow: {source} -> {self.current_function}.Return[{idx}]"
                            )
                            if self.condition_stack:
                                semantics.append( f"Conditional Return: when {' and '.join(self.condition_stack)}, "
                                                  f"{self.current_function}.Return[{idx}] = {source}" )
                except Exception:
                    pass
                self.generic_visit(n)

            def visit_If(self, n):
                try:
                    condition = ast.unparse(n.test)
                except Exception:
                    condition = "<unknown condition>"
                self.condition_stack.append(condition)
                for statement in n.body:
                    self.visit(statement)
                self.condition_stack.pop()
                for statement in n.orelse:
                    self.visit(statement)

            def visit_Attribute(self, n):
                self.generic_visit(n)
            def visit_AnnAssign(self, n):
                try:
                    if isinstance(n.target, ast.Attribute):
                        if n.value is not None:
                            semantics.append(
                                f"State Update: {ast.unparse(n.value)} -> {ast.unparse(n.target)}"
                            )
                except Exception:
                    pass
                self.generic_visit(n)
        Visitor(self).visit(tree)
        semantic_priority = (
            "Data Flow",
            "Overwrite",
            "State Update",
            "Argument Flow",
            "Keyword Flow",
        )
        semantics = list(dict.fromkeys(semantics))
        semantics.sort(
            key=lambda x: next(
                (
                    i
                    for i, p in enumerate(semantic_priority)
                    if x.startswith(p)
                ),
                999,
            )
        )
        return semantics

    def _build_verified_defect_facts(self, evidence):

        defect_facts = defaultdict(list)

        conditional_none_returns = set()
        tuple_assignments = []

        for item in evidence:
            for relation in item.semantics:
                if (relation.startswith("Conditional Return:") and ".Return[" in relation and relation.endswith("= None")):
                    match = re.search(
                        r"when (.*?),\s*([A-Za-z_][\w.]*)\.Return\[(\d+)\] = None$",
                        relation,
                    )
                    if match:
                        condition = match.group(1).strip()
                        function_name = match.group(2).strip()
                        return_index = int(match.group(3))
                        conditional_none_returns.add((condition, function_name, return_index))
                elif relation.startswith("Data Flow:") and ".Return[" in relation:
                    match = re.search(
                        r"Data Flow:\s*([A-Za-z_][\w.]*)\.Return\[(\d+)\]\s*->\s*([A-Za-z_]\w*)$",
                        relation,
                    )
                    if match:
                        function_name = match.group(1).strip()
                        return_index = int(match.group(2))
                        target_variable = match.group(3).strip()
                        tuple_assignments.append((
                                item.symbol,
                                function_name,
                                return_index,
                                target_variable,
                            )
                        )

        for (
            caller_symbol,
            helper_symbol,
            return_index,
            target_variable,
        ) in tuple_assignments:
            for (
                condition,
                returned_helper_symbol,
                none_return_index,
            ) in conditional_none_returns:
                if (
                    helper_symbol == returned_helper_symbol
                    and return_index == none_return_index
                ):
                    defect_facts[caller_symbol].append(
                        "Verified Defect: "
                        f"When {condition}, "
                        f"{helper_symbol}.Return[{return_index}] is None, "
                        f"and {caller_symbol} assigns that value to "
                        f"{target_variable}. "
                        f"Therefore {caller_symbol} overwrites local "
                        f"{target_variable} with None."
                    )
        return defect_facts

    def _build_propagation_facts(self, semantics):
        edges = []
        for relation in semantics:
            if not relation.startswith("Data Flow:"):
                continue
            flow = relation[len("Data Flow:"):].strip()
            if "->" not in flow:
                continue
            source, target = [
                value.strip()
                for value in flow.split("->", 1)
            ]
            if source == target:
                continue
            edges.append((source, target))
        outgoing = {}
        for source, target in edges:
            outgoing.setdefault(source, set()).add(target)
        propagation_facts = set()
        for source, middle in edges:
            for target in outgoing.get(middle, set()):
                if target == source:
                    continue
                propagation_facts.add(
                    f"Propagation: {source} -> {middle} -> {target}"
                )
        return sorted(propagation_facts)

    def _graph_observations(self, node):
        obs = []
        for callee in sorted(node.calls):
            obs.append(
                f"Calls repository symbol: {self.graph[callee].chunk.name}"
            )
        for caller in sorted(node.called_by):
            obs.append(
                f"Called by repository symbol: {self.graph[caller].chunk.name}"
            )
        for imp in sorted(node.imports):
            obs.append(
                f"Imports module: {imp}"
            )
        return obs