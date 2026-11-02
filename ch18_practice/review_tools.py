"""Tools for the code-review assistant (the through-line project of Chapter 18).

Each tool is an ordinary Python function plus a JSON schema in the format the
DeepSeek API (OpenAI-compatible) expects in the `tools` parameter. Results are
returned as JSON strings, because a tool message carries text.

Safety: the model only ever names a file; the harness resolves the path and
refuses anything outside SANDBOX. Never let a model choose arbitrary paths,
shell commands, or code to execute.

See Chapter 18 of "Secrets of DeepSeek AI".
"""
import ast
import json
import os

SANDBOX = os.path.realpath(os.path.dirname(__file__))
MAX_BYTES = 200_000                      # a review tool reads source files


def _safe_read(path):
    # realpath also resolves symbolic links that point outside the sandbox
    full = os.path.realpath(os.path.join(SANDBOX, path))
    if not full.startswith(SANDBOX + os.sep):
        raise PermissionError(f"path outside sandbox: {path}")
    if os.path.getsize(full) > MAX_BYTES:
        raise ValueError(f"file too large for review: {path}")
    with open(full, encoding="utf-8") as f:
        return f.read()


def sandbox_relpath(path):
    """Path of a file given on the command line, relative to SANDBOX.
    Raises ValueError if the file is missing or lies outside the sandbox."""
    full = os.path.realpath(path)
    if not os.path.isfile(full):
        raise ValueError(f"no such file: {path}")
    if not full.startswith(SANDBOX + os.sep):
        raise ValueError(f"file must be inside the sandbox {SANDBOX}")
    return os.path.relpath(full, SANDBOX)


def list_functions(path):
    """Names, line numbers, argument counts and docstring presence."""
    tree = ast.parse(_safe_read(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({
                "name": node.name,
                "line": node.lineno,
                "n_args": len(node.args.args),
                "has_docstring": ast.get_docstring(node) is not None,
                "length": (node.end_lineno or node.lineno) - node.lineno + 1,
            })
    return json.dumps(out, ensure_ascii=False)


def static_checks(path):
    """A handful of classic Python smells, found with the standard ast module."""
    tree = ast.parse(_safe_read(path))
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({"line": node.lineno, "kind": "bare-except",
                           "hint": "catch a specific exception class"})
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    issues.append({"line": node.lineno, "kind": "mutable-default",
                                   "hint": f"default argument of {node.name}() is shared between calls"})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("eval", "exec"):
            issues.append({"line": node.lineno, "kind": "eval-exec",
                           "hint": f"{node.func.id}() on external data is a code-injection risk"})
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comp, ast.Constant) \
                        and comp.value is None:
                    issues.append({"line": node.lineno, "kind": "eq-none",
                                   "hint": "use 'is None' / 'is not None'"})
    issues.sort(key=lambda i: i["line"])
    return json.dumps(issues, ensure_ascii=False)


TOOLS = [
    {"type": "function", "function": {
        "name": "list_functions",
        "description": "List the functions defined in a Python file of the project.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string",
                                               "description": "file path relative to the project root"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "static_checks",
        "description": "Run simple static checks (bare except, mutable defaults, eval/exec, == None).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]

REGISTRY = {"list_functions": list_functions, "static_checks": static_checks}


if __name__ == "__main__":
    print(list_functions("sample_buggy.py"))
    print(static_checks("sample_buggy.py"))
