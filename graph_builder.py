from dataclasses import dataclass, field
from tree_sitter_language_pack import get_parser

from chunker import CodeChunk

@dataclass
class GraphNode:
    chunk: CodeChunk
    calls: set[str] = field(default_factory=set)
    called_by: set[str] = field(default_factory=set)
    contains: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    inherits: set[str] = field(default_factory=set)

class GraphBuilder:
    def __init__(self):
        self.graph = {}
        self.id_lookup = {}
        self.symbol_lookup = {}
        self.parsers = {}

    def build_graph(self, chunks):
        for chunk in chunks:
            node = GraphNode(chunk)
            self.graph[chunk.id] = node
            self.id_lookup[chunk.id] = node
            self.symbol_lookup[(chunk.file_path, chunk.name)] = chunk.id
        self._build_contains_edges()
        self._build_call_edges()
        self._build_import_edges()
        return self.graph

    def _build_contains_edges(self):
        for node in self.graph.values():
            chunk = node.chunk
            if chunk.class_name is None:
                continue
            parent_id = self.symbol_lookup.get((chunk.file_path, chunk.class_name))
            if parent_id is None:
                continue
            self.graph[parent_id].contains.add(chunk.id)

    def _build_import_edges(self):
        for node in self.graph.values():
            chunk = node.chunk
            try:
                parser = self._get_parser(chunk.language)
            except Exception:
                continue
            code = chunk.code
            if chunk.language in {"javascript", "typescript", "tsx"}:
                if chunk.chunk_type == "method":
                    code = f"class Dummy {{\n{code}\n}}"
            source_bytes = code.encode("utf-8")
            tree = parser.parse(code)
            self._extract_imports(
                tree.root_node(),
                chunk,
                node,
                source_bytes
            )
            imports_by_file = {}
            for node in self.graph.values():
                if node.chunk.chunk_type == "imports":
                    imports_by_file[node.chunk.file_path] = node.imports
            for node in self.graph.values():
                if node.chunk.chunk_type == "imports":
                    continue
                node.imports.update(
                    imports_by_file.get(
                        node.chunk.file_path,
                        set(),
                    )
                )

    def _extract_imports(self, node, chunk, graph_node, source_bytes):
        IMPORT_NODES = {
            "python": {
                "import_statement",
                "import_from_statement",
            },
            "javascript": {
                "import_statement",
            },
            "typescript": {
                "import_statement",
            },
            "tsx": {
                "import_statement",
            },
            "cpp": {
                "preproc_include",
            },
        }
        if node.kind() in IMPORT_NODES.get(chunk.language, set()):
            imported = self._extract_import_name(node, source_bytes, chunk.language)
            if imported:
                graph_node.imports.add(imported)
        for i in range(node.child_count()):
            self._extract_imports(
                node.child(i),
                chunk,
                graph_node,
                source_bytes,
            )

    def _extract_import_name(self, node, source_bytes, language):
        if language == "cpp":
            text = source_bytes[
                node.start_byte(): node.end_byte()].decode()
            text = text.replace("#include", "").strip()
            return text.strip("<>\"")
        if language == "python":
            text = source_bytes[node.start_byte(): node.end_byte()].decode()
            if text.startswith("import "):
                return text.replace("import", "").strip()
            if text.startswith("from "):
                text = text.replace("from ", "")
                module = text.split(" import ")[0]
                return module
            return None
        if language in {"javascript","typescript","tsx",}:
            text = source_bytes[node.start_byte():node.end_byte()].decode()
            if "from" in text:
                return (
                    text.split("from")[-1].replace('"', "").replace("'", "").replace(";", "").strip()
                )
        return None
        
    def _collect_local_variables(self, node, source_bytes):
        variables = {}
        stack = [node]
        while stack:
            current = stack.pop()
            if current.kind() == "declaration":
                type_name = None
                variable_name = None
                for i in range(current.child_count()):
                    child = current.child(i)
                    if child.kind() in {
                        "type_identifier",
                        "primitive_type",
                        "qualified_identifier",
                    }:
                        type_name = source_bytes[child.start_byte():child.end_byte()].decode()
                    elif child.kind() == "identifier":
                        variable_name = source_bytes[child.start_byte():child.end_byte()].decode()
                    elif child.kind() == "init_declarator":
                        for j in range(child.child_count()):
                            grandchild = child.child(j)
                            if grandchild.kind() == "identifier":
                                variable_name = source_bytes[grandchild.start_byte():grandchild.end_byte()].decode()
                if type_name and variable_name:
                    variables[variable_name] = type_name
            for i in range(current.child_count() - 1, -1, -1):
                stack.append(current.child(i))
        return variables

    def get_node(self, node_id):
        return self.graph.get(node_id)
    
    def _get_parser(self, language):
        if language not in self.parsers:
            self.parsers[language] = get_parser(language)
        return self.parsers[language]
    
    def _build_call_edges(self):
        for node in self.graph.values():
            chunk = node.chunk
            if chunk.chunk_type not in {"function", "method"}:
                continue
            try:
                parser = self._get_parser(chunk.language)
            except Exception:
                continue
            source_bytes = chunk.code.encode("utf-8")
            code = chunk.code
            if chunk.language in {"javascript", "typescript", "tsx"}:
                if chunk.chunk_type == "method":
                    code = f"class Dummy {{\n{chunk.code}\n}}"
            tree = parser.parse(code)
            locals = self._collect_local_variables(
                tree.root_node(),
                source_bytes,
            )
            self._extract_calls(
                tree.root_node(),
                chunk,
                node,
                source_bytes,
                locals,
            )
        self._build_called_by_edges()
        

    def _dump_tree(self, node, depth=0):
        print("  " * depth + node.kind())
        for i in range(node.child_count()):
            self._dump_tree(node.child(i), depth + 1)
    
    def _extract_calls(self, node, chunk, graph_node, source_bytes, locals):
        if node.kind() in {"call_expression", "call"}:
            function_name = self._extract_call_name(node, source_bytes, chunk.language, locals)
            if function_name:
                self._connect_call(graph_node, chunk, function_name)
        for i in range(node.child_count()):
            self._extract_calls(
                node.child(i),
                chunk,
                graph_node,
                source_bytes,
                locals
            )

    def _connect_call(self, graph_node, chunk, function_name):
        candidates = [function_name]
        if chunk.class_name:
            candidates.append(
                f"{chunk.class_name}.{function_name}"
            )
        for candidate in candidates:
            target = self.symbol_lookup.get((chunk.file_path, candidate))
            if target:
                graph_node.calls.add(target)
                return
        for (file_path, symbol_name), target in self.symbol_lookup.items():
            if symbol_name in candidates:
                graph_node.calls.add(target)
            
    def _build_called_by_edges(self):
        for node in self.graph.values():
            for called in node.calls:
                self.graph[
                    called
                ].called_by.add(node.chunk.id)

    def _extract_call_name(self, node, source_bytes, language, locals):
        if language == "cpp":
            function = node.child_by_field_name("function")
            if function is None:
                return None
            if function.kind() == "identifier":
                return source_bytes[function.start_byte():function.end_byte()].decode()
            if function.kind() == "field_expression":
                object_node = function.child_by_field_name("argument")
                field_node = function.child_by_field_name("field")
                if object_node and field_node:
                    obj = source_bytes[object_node.start_byte():object_node.end_byte()].decode()
                    method = source_bytes[field_node.start_byte():field_node.end_byte()].decode()
                    if obj in locals:
                        return f"{locals[obj]}.{method}"
                    return method
        if language == "python":
            function = node.child_by_field_name("function")
            if function and function.kind() == "attribute":
                for i in range(function.child_count() - 1, -1, -1):
                    child = function.child(i)
                    if child.kind() == "identifier":
                        return source_bytes[child.start_byte(): child.end_byte()].decode("utf-8")
        IDENTIFIERS = {
            "identifier",
            "field_identifier",
        }
        stack = [node]
        while stack:
            current = stack.pop()
            if current.kind() in IDENTIFIERS:
                return source_bytes[current.start_byte(): current.end_byte()].decode("utf-8")
            for i in range(current.child_count() - 1, -1, -1):
                stack.append(current.child(i))
        return None
    
    def trace_execution_path(self, node_id, visited=None,):
        if visited is None:
            visited=set()
        if node_id in visited:
            return []
        visited.add(node_id)
        path=[self.graph[node_id]]
        for child in self.graph[node_id].calls:
            path.extend(self.trace_execution_path(child, visited))
        return path