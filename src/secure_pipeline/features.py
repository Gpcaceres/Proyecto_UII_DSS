import ast
import re
from typing import Dict, List, Optional

DANGEROUS_CALLS = {
    "eval",
    "exec",
    "os.system",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_output",
    "pickle.loads",
    "yaml.load",
}

SANITIZATION_CALLS = {
    "shlex.quote",
    "yaml.safe_load",
    "json.loads",
    "escape",
    "replace",
    "os.path.join",
}

DANGEROUS_CALLS_C = {
    "strcpy",
    "strcat",
    "sprintf",
    "vsprintf",
    "gets",
    "scanf",
    "system",
    "popen",
}

SANITIZATION_CALLS_C = {
    "snprintf",
    "vsnprintf",
    "strncpy",
    "strlcpy",
    "fgets",
    "strncat",
}


def _ast_depth(node: ast.AST, current: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return current
    return max(_ast_depth(child, current + 1) for child in children)


def extract_features(source: str, language: Optional[str] = None) -> Dict[str, float]:
    lang = (language or "python").lower()

    tokens = re.findall(r"\w+", source)
    token_count = len(tokens)

    if lang in {"c", "cpp", "c++"}:
        depth = _brace_depth(source)
        dangerous = len(re.findall(_call_regex(DANGEROUS_CALLS_C), source))
        sanitized = len(re.findall(_call_regex(SANITIZATION_CALLS_C), source))
    else:
        try:
            tree = ast.parse(source)
            depth = _ast_depth(tree)
        except SyntaxError:
            depth = 0
            tree = None

        dangerous = 0
        sanitized = 0

        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fullname = _resolve_call_name(node)
                    if fullname in DANGEROUS_CALLS:
                        dangerous += 1
                    if fullname in SANITIZATION_CALLS:
                        sanitized += 1

        dangerous += len(re.findall(r"\\b(eval|exec|subprocess\\.run|os\\.system)\\b", source))

    return {
        "token_count": float(token_count),
        "avg_token_length": float(sum(len(t) for t in tokens) / token_count) if token_count else 0.0,
        "ast_depth": float(depth),
        "dangerous_calls": float(dangerous),
        "sanitization_calls": float(sanitized),
    }


def feature_names() -> List[str]:
    return [
        "token_count",
        "avg_token_length",
        "ast_depth",
        "dangerous_calls",
        "sanitization_calls",
    ]


def _resolve_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        value = node.func.value
        if isinstance(value, ast.Name):
            return f"{value.id}.{node.func.attr}"
        if isinstance(value, ast.Attribute):
            return f"{_attr_to_str(value)}.{node.func.attr}"
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _attr_to_str(attr: ast.Attribute) -> str:
    parts: List[str] = []
    current = attr
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        if isinstance(current.value, ast.Name):
            parts.append(current.value.id)
            break
        current = current.value  # type: ignore[assignment]
    return ".".join(reversed(parts))


def _brace_depth(source: str) -> int:
    depth = 0
    max_depth = 0
    for char in source:
        if char == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == "}":
            depth = max(depth - 1, 0)
    return max_depth


def _call_regex(names: set[str]) -> str:
    escaped = [re.escape(name) for name in names]
    return r"\b(" + "|".join(escaped) + r")\s*\("
