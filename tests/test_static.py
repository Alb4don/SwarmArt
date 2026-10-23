import ast
import importlib
import io
import tokenize
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "swarmart"
SOURCES = sorted(PACKAGE.glob("*.py"))
EXTERNAL_ROOTS = ("numpy", "pygame", "PIL", "PySide6", "shiboken6")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_comments_or_docstrings(path):
    source = path.read_text(encoding="utf-8")
    comments = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT]
    assert not comments, f"comments found in {path.name}"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            assert ast.get_docstring(node) is None, f"docstring in {path.name}:{getattr(node, 'name', 'module')}"


def _resolve(obj, parts):
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, list(reversed(parts))
    return None, None


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_external_symbol_exists(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in EXTERNAL_ROOTS:
                    module = importlib.import_module(alias.name)
                    aliases[alias.asname or alias.name.split(".")[0]] = (
                        module if alias.asname else importlib.import_module(alias.name.split(".")[0])
                    )
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in EXTERNAL_ROOTS:
            module = importlib.import_module(node.module)
            for alias in node.names:
                if not hasattr(module, alias.name):
                    importlib.import_module(f"{node.module}.{alias.name}")
                assert hasattr(module, alias.name), f"{node.module}.{alias.name} does not exist"
                aliases[alias.asname or alias.name] = getattr(module, alias.name)
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not isinstance(getattr(node, "ctx", None), ast.Store):
            root, parts = _attribute_chain(node)
            if root in aliases and parts:
                _resolve(aliases[root], parts)
                checked += 1
    if aliases:
        assert checked > 0


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_internal_imports_resolve(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            module = importlib.import_module(f"swarmart.{node.module}" if node.module else "swarmart")
            for alias in node.names:
                assert hasattr(module, alias.name), f"swarmart.{node.module}.{alias.name} missing"


def test_no_unsafe_calls():
    banned = {"eval", "exec", "compile", "__import__"}
    banned_modules = {"pickle", "marshal", "shelve", "subprocess"}
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in banned, f"{path.name} calls {node.func.id}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not banned_modules & {n.split(".")[0] for n in names}, path.name
