"""Feature extraction from source code and error messages."""
from typing import Any, Dict
import ast
import io
import keyword
import tokenize
import re


def _safe_parse(code: str):
    """Try to parse code into an AST without executing it.

    Returns (ast_tree_or_None, parse_error_or_None)
    """
    try:
        tree = ast.parse(code)
        return tree, None
    except Exception as e:
        return None, e


def _count_tokens(code: str) -> int:
    # simple word tokenization (identifiers, keywords, numbers)
    tokens = re.findall(r"\w+", code)
    return len(tokens)


def extract_features(code: str, error_message: str = None) -> Dict[str, Any]:
    """Return a feature dict for a single sample.

    Features are static, textual, and AST-based where possible. This function
    never executes the code. It is defensive against malformed snippets.
    """
    features: Dict[str, Any] = {}
    if code is None:
        code = ''
    # basic text features
    features['n_chars'] = len(code)
    features['n_lines'] = code.count('\n') + (1 if code.strip() else 0)
    features['n_tokens'] = _count_tokens(code)

    # comments and blank lines
    try:
        g = tokenize.generate_tokens(io.StringIO(code).readline)
        n_comments = 0
        n_newlines = 0
        n_indents = 0
        n_tabs = 0
        for toknum, tokval, _, _, _ in g:
            if toknum == tokenize.COMMENT:
                n_comments += 1
            if toknum == tokenize.NL or toknum == tokenize.NEWLINE:
                n_newlines += 1
            if toknum == tokenize.INDENT:
                n_indents += 1
            if toknum == tokenize.OP and '\t' in tokval:
                n_tabs += 1
        features['n_comments'] = n_comments
        features['n_newline_tokens'] = n_newlines
        features['n_indent_tokens'] = n_indents
        features['n_tab_chars'] = code.count('\t')
    except Exception:
        # tokenization failed; set defaults
        features['n_comments'] = 0
        features['n_newline_tokens'] = 0
        features['n_indent_tokens'] = 0
        features['n_tab_chars'] = code.count('\t')

    # bracket/parenthesis counts
    features['n_paren_open'] = code.count('(')
    features['n_paren_close'] = code.count(')')
    features['n_bracket_open'] = code.count('[')
    features['n_bracket_close'] = code.count(']')
    features['n_brace_open'] = code.count('{')
    features['n_brace_close'] = code.count('}')

    # operator-like characters (simple heuristic)
    features['n_ops_chars'] = sum(code.count(c) for c in ['+', '-', '*', '/', '%', '=', '>', '<', '!'])

    # keyword counts (how many Python keywords appear)
    kwlist = set(keyword.kwlist)
    code_words = set(re.findall(r"\b\w+\b", code))
    features['n_keywords_present'] = sum(1 for w in code_words if w in kwlist)

    # AST features
    tree, err = _safe_parse(code)
    features['ast_parse_success'] = 1 if tree is not None else 0
    features['ast_syntax_error'] = 0 if tree is not None else 1
    n_funcs = 0
    n_assign_targets = 0
    n_names = 0
    var_names = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                n_funcs += 1
            if isinstance(node, ast.Assign):
                # count targets as proxies for variable writes
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        var_names.add(t.id)
                        n_assign_targets += 1
            if isinstance(node, ast.Name):
                n_names += 1
                if isinstance(node.ctx, ast.Store):
                    var_names.add(node.id)
    features['n_functions'] = n_funcs
    features['n_assignments'] = n_assign_targets
    features['n_names_total'] = n_names
    features['n_unique_var_names'] = len(var_names)

    # indentation characteristics: average leading spaces of non-blank lines
    indents = []
    for ln in code.splitlines():
        if ln.strip():
            leading = len(ln) - len(ln.lstrip(' '))
            indents.append(leading)
    features['avg_leading_spaces'] = sum(indents) / len(indents) if indents else 0
    features['has_tab_indentation'] = 1 if '\t' in code else 0

    # optional error_message feature
    if error_message:
        features['error_msg_len'] = len(str(error_message))
    else:
        features['error_msg_len'] = 0

    return features


def extract_features(code: str, error_message: str = None) -> Dict[str, Any]:
    """Return a feature dict for a single sample.

    Start with simple features: length, number of lines, token counts, presence of keywords.
    Replace with AST-based features later.
    """
    # kept for backward compatibility: call implementation below
    return _extract_features_impl(code, error_message)


def _extract_features_impl(code: str, error_message: str = None) -> Dict[str, Any]:
    # Implementation is defined above (duplicate name avoided)
    # Reuse the logic by calling the module-level functions already computed.
    # For historical callers, the top-level function routes here.
    # The actual implementation is the body present earlier (we call it again).
    # To avoid duplicating code, call the parsing/count helpers above.
    features: Dict[str, Any] = {}
    if code is None:
        code = ''
    # basic text features
    features['n_chars'] = len(code)
    features['n_lines'] = code.count('\n') + (1 if code.strip() else 0)
    features['n_tokens'] = _count_tokens(code)

    # comments and blank lines
    try:
        g = tokenize.generate_tokens(io.StringIO(code).readline)
        n_comments = 0
        n_newlines = 0
        n_indents = 0
        for toknum, tokval, _, _, _ in g:
            if toknum == tokenize.COMMENT:
                n_comments += 1
            if toknum == tokenize.NL or toknum == tokenize.NEWLINE:
                n_newlines += 1
            if toknum == tokenize.INDENT:
                n_indents += 1
        features['n_comments'] = n_comments
        features['n_newline_tokens'] = n_newlines
        features['n_indent_tokens'] = n_indents
        features['n_tab_chars'] = code.count('\t')
    except Exception:
        features['n_comments'] = 0
        features['n_newline_tokens'] = 0
        features['n_indent_tokens'] = 0
        features['n_tab_chars'] = code.count('\t')

    # bracket/parenthesis counts
    features['n_paren_open'] = code.count('(')
    features['n_paren_close'] = code.count(')')
    features['n_bracket_open'] = code.count('[')
    features['n_bracket_close'] = code.count(']')
    features['n_brace_open'] = code.count('{')
    features['n_brace_close'] = code.count('}')

    # operator-like characters (simple heuristic)
    features['n_ops_chars'] = sum(code.count(c) for c in ['+', '-', '*', '/', '%', '=', '>', '<', '!'])

    # keyword counts (how many Python keywords appear)
    kwlist = set(keyword.kwlist)
    code_words = set(re.findall(r"\b\w+\b", code))
    features['n_keywords_present'] = sum(1 for w in code_words if w in kwlist)

    # AST features
    tree, err = _safe_parse(code)
    features['ast_parse_success'] = 1 if tree is not None else 0
    features['ast_syntax_error'] = 0 if tree is not None else 1
    n_funcs = 0
    n_assign_targets = 0
    n_names = 0
    var_names = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                n_funcs += 1
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        var_names.add(t.id)
                        n_assign_targets += 1
            if isinstance(node, ast.Name):
                n_names += 1
                if isinstance(node.ctx, ast.Store):
                    var_names.add(node.id)
    features['n_functions'] = n_funcs
    features['n_assignments'] = n_assign_targets
    features['n_names_total'] = n_names
    features['n_unique_var_names'] = len(var_names)

    # indentation characteristics: average leading spaces of non-blank lines
    indents = []
    for ln in code.splitlines():
        if ln.strip():
            leading = len(ln) - len(ln.lstrip(' '))
            indents.append(leading)
    features['avg_leading_spaces'] = sum(indents) / len(indents) if indents else 0
    features['has_tab_indentation'] = 1 if '\t' in code else 0

    # optional error_message feature
    if error_message:
        features['error_msg_len'] = len(str(error_message))
    else:
        features['error_msg_len'] = 0

    return features
