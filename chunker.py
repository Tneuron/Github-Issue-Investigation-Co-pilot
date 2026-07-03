from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from tree_sitter_language_pack import get_parser

from file_parser import Document

DEBUG = True

@dataclass
class CodeChunk:
    id: str
    file_path: str
    language: str
    name: str
    chunk_type: str
    node_type: str
    class_name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    code: str = ""


class StructuralChunker:

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".cs": "c_sharp",
        ".php": "php",
        ".rb": "ruby",
    }

    def chunk_document(self, document: Document) -> List[CodeChunk]:
        language = self.LANGUAGE_MAP.get(document.extension)
        if language is None:
            return []
        try:
            parser = get_parser(language)
        except Exception as e:
            print(f"Failed to load parser: {e}")
            return []
        source_bytes = document.content.encode("utf-8")
        tree = parser.parse(document.content)
        chunks = []
        self._extract_import_chunk(
            tree.root_node(),
            source_bytes,
            document,
            language,
            chunks,
        )
        if language == "python":
            self._extract_python(
                tree.root_node(),
                source_bytes,
                document,
                chunks,
                []
            )
        elif language in ("javascript", "typescript", "tsx"):
            self._extract_javascript(
                tree.root_node(),
                source_bytes,
                document,
                language,
                chunks,
                []
            )
        elif language in ("cpp", "c"):
            self._extract_cpp(
                tree.root_node(),
                source_bytes,
                document,
                language,
                chunks,
                []
            )
        elif language == "go":
            self._extract_go(
                tree.root_node(),
                source_bytes,
                document,
                chunks,
                []
            )
        elif language == "rust":
            self._extract_rust(
                tree.root_node(),
                source_bytes,
                document,
                chunks,
                []
            )
        elif language == "java":
            self._extract_java(
                tree.root_node(),
                source_bytes,
                document,
                chunks,
                []
            )
        if not chunks:
            chunks.append(
                CodeChunk(
                    id=document.file_path,
                    file_path=document.file_path,
                    language=language,
                    name=Path(document.file_path).stem,
                    chunk_type="file",
                    node_type="file",
                    start_line=1,
                    end_line=document.content.count("\n")+1,
                    code=document.content,
                )
            )
        return chunks

    def _extract_code(self, node, source_bytes):
        return source_bytes[
            node.start_byte():
            node.end_byte()
        ].decode(
            "utf-8",
            errors="ignore"
        )

    def _node_text(self, node, source_bytes):
        return source_bytes[
            node.start_byte():
            node.end_byte()
        ].decode(
            "utf-8",
            errors="ignore"
        )
    
    def _make_chunk(self, document, language, node, name, chunk_type, class_stack, source_bytes):
        if chunk_type == "method":
            qualified_name = ".".join(class_stack + [name])
            class_name = ".".join(class_stack)
        else:
            qualified_name = name
            class_name = None
        return CodeChunk(
            id=f"{document.file_path}:{qualified_name}",
            file_path=document.file_path,
            language=language,
            name=qualified_name,
            chunk_type=chunk_type,
            node_type=node.kind(),
            class_name=class_name,
            start_line=node.start_position().row + 1,
            end_line=node.end_position().row + 1,
            code=self._extract_code(node, source_bytes)
        )

    def _extract_import_chunk(self, root, source_bytes, document, language, chunks):
        IMPORT_TYPES = {
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

            "c": {
                "preproc_include",
            },
        }

        wanted = IMPORT_TYPES.get(language, set())
        import_nodes = []
        self._collect_import_nodes(root, wanted, import_nodes,)
        if not import_nodes:
            return
        code = "\n".join(self._extract_code(node, source_bytes) for node in import_nodes)
        chunks.append(
            CodeChunk(
                id=f"{document.file_path}:imports",
                file_path=document.file_path,
                language=language,
                name=f"{Path(document.file_path).stem}.imports",
                chunk_type="imports",
                node_type="imports",
                class_name=None,
                start_line=import_nodes[0].start_position().row + 1,
                end_line=import_nodes[-1].end_position().row + 1,
                code=code,
            )
        )

    def _collect_import_nodes(self, node, wanted, result):
        if node.kind() in wanted:
            result.append(node)
        for i in range(node.child_count()):
            self._collect_import_nodes(
                node.child(i),
                wanted,
                result,
            )

    def _extract_python(self, node, source_bytes, document, chunks, class_stack,):
        kind = node.kind()
        if kind == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._node_text(name_node, source_bytes)
                chunks.append(
                    self._make_chunk(
                        document,
                        "python",
                        node,
                        name,
                        "class",
                        class_stack,
                        source_bytes,
                    )
                )
                class_stack.append(name)
                for i in range(node.child_count()):
                    self._extract_python(
                        node.child(i),
                        source_bytes,
                        document,
                        chunks,
                        class_stack,
                    )
                class_stack.pop()
                return
        elif kind == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._node_text(name_node, source_bytes)
                chunk_type = ("method" if class_stack else "function")
                chunks.append(
                    self._make_chunk(
                        document,
                        "python",
                        node,
                        name,
                        chunk_type,
                        class_stack,
                        source_bytes,
                    )
                )
        for i in range(node.child_count()):
            self._extract_python(
                node.child(i),
                source_bytes,
                document,
                chunks,
                class_stack,
            )

    def _extract_javascript(self, node, source_bytes, document, language, chunks, class_stack,):
        kind = node.kind()
        if kind == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._node_text(name_node, source_bytes,)
                chunks.append(
                    self._make_chunk(
                        document,
                        language,
                        node,
                        name,
                        "class",
                        class_stack,
                        source_bytes,
                    )
                )
                class_stack.append(name)
                for i in range(node.child_count()):
                    self._extract_javascript(
                        node.child(i),
                        source_bytes,
                        document,
                        language,
                        chunks,
                        class_stack,
                    )
                class_stack.pop()
                return
        elif kind in ("function_declaration", "method_definition",):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._node_text(name_node, source_bytes,)
                chunk_type = ("method" if class_stack else "function")
                chunks.append(
                    self._make_chunk(
                        document,
                        language,
                        node,
                        name,
                        chunk_type,
                        class_stack,
                        source_bytes,
                    )
                )
        for i in range(node.child_count()):
            self._extract_javascript(
                node.child(i),
                source_bytes,
                document,
                language,
                chunks,
                class_stack,
            )
    
    def _extract_cpp(self, node, source_bytes, document, language, chunks, class_stack,):
        kind = node.kind()
        if kind in ("class_specifier", "struct_specifier"):
            class_name = None
            for i in range(node.child_count()):
                child = node.child(i)
                if child.kind() == "type_identifier":
                    class_name = self._node_text(child, source_bytes)
                    break
            if class_name:
                chunks.append(
                    self._make_chunk(
                        document,
                        language,
                        node,
                        class_name,
                        "class",
                        class_stack,
                        source_bytes,
                    )
                )
                class_stack.append(class_name)
                for i in range(node.child_count()):
                    self._extract_cpp(
                        node.child(i),
                        source_bytes,
                        document,
                        language,
                        chunks,
                        class_stack,
                    )
                class_stack.pop()
                return
        elif kind == "function_definition":
            function_name = self._extract_cpp_function_name(node, source_bytes)
            if function_name:
                chunk_type = ("method" if class_stack else "function")
                chunk = self._make_chunk(
                        document,
                        language,
                        node,
                        function_name,
                        chunk_type,
                        class_stack,
                        source_bytes,
                    )
                self._analyze_cpp_function(
                    node,
                    source_bytes
                )
                chunks.append(chunk)
        for i in range(node.child_count()):
            self._extract_cpp(
                node.child(i),
                source_bytes,
                document,
                language,
                chunks,
                class_stack,
            )

    def _analyze_cpp_function(self, function_node, source_bytes):
        local_variables = {}
        resolved_calls = []
        stack = [function_node]
        while stack:
            node = stack.pop()
            if node.kind() == "declaration":
                type_name = None
                variable_name = None
                for i in range(node.child_count()):
                    child = node.child(i)
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
                                variable_name = source_bytes[grandchild.start_byte(): grandchild.end_byte()].decode()
                if type_name and variable_name:
                    local_variables[variable_name] = type_name
            elif node.kind() == "call_expression":
                function = node.child_by_field_name("function")
                if function is None:
                    continue
                if function.kind() == "identifier":
                    resolved_calls.append(
                        source_bytes[function.start_byte(): function.end_byte()].decode()
                    )
                elif function.kind() == "field_expression":
                    object_name = None
                    method_name = None
                    for i in range(function.child_count()):
                        child = function.child(i)
                        if child.kind() == "identifier":
                            if object_name is None:
                                object_name = source_bytes[child.start_byte(): child.end_byte()].decode()
                            else:
                                method_name = source_bytes[child.start_byte(): child.end_byte()].decode()
                    if object_name and method_name:
                        if object_name in local_variables:
                            resolved_calls.append(f"{local_variables[object_name]}.{method_name}")
                        else:
                            resolved_calls.append(method_name)
            for i in range(node.child_count() - 1, -1, -1):
                stack.append(node.child(i))
        return resolved_calls

    def _extract_cpp_function_name(self, node, source_bytes,):
        declarator = None
        for i in range(node.child_count()):
            child = node.child(i)
            if child.kind() == "function_declarator":
                declarator = child
                break
        if declarator is None:
            return None
        return self._find_cpp_identifier(declarator, source_bytes)
    
    def _find_cpp_identifier(self, node, source_bytes,):
        if node.kind() in (
            "field_identifier",
            "identifier",
            "destructor_name",
            "operator_name",
        ):
            return self._node_text(node, source_bytes)
        for i in range(node.child_count()):
            result = self._find_cpp_identifier(node.child(i), source_bytes,)
            if result:
                return result
        return None
    