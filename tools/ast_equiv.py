"""Prove an edit touched only comments and docstrings.

Compares each tracked .py file against a git ref: parses both, strips every
docstring, and dumps the ASTs. Identical dumps mean nothing executable moved.
Files that differ are listed for manual review (a deliberate code deletion will
show up here, which is the point).

Usage: python3 tools/ast_equiv.py [ref]        default ref: HEAD
"""

import ast
import subprocess
import sys


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def dump(src):
    return ast.dump(strip_docstrings(ast.parse(src)), include_attributes=False)


def main(ref="HEAD"):
    files = subprocess.run(["git", "ls-files", "*.py"], capture_output=True,
                           text=True, check=True).stdout.split()
    same, diff, err = [], [], []
    for f in files:
        try:
            old = subprocess.run(["git", "show", f"{ref}:{f}"],
                                 capture_output=True, text=True)
            if old.returncode != 0:
                err.append((f, "not in ref"))
                continue
            (same if dump(old.stdout) == dump(open(f).read()) else diff).append(f)
        except SyntaxError as e:
            err.append((f, f"SyntaxError: {e}"))

    print(f"identical AST : {len(same)}")
    print(f"AST CHANGED   : {len(diff)}")
    for f in diff:
        print(f"    {f}")
    if err:
        print(f"errors        : {len(err)}")
        for f, e in err:
            print(f"    {f}: {e}")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
