"""Mechanical guard for the review-validation marker boundary."""

import ast
from pathlib import Path


class _ModuleSlowMarkerVisitor(ast.NodeVisitor):
    """Find slow markers assigned to the module-level pytestmark name."""

    def __init__(self, pytest_names: set[str], mark_names: set[str], filename: str):
        self._pytest_names = pytest_names
        self._mark_names = mark_names
        self._filename = filename
        self._scope_depth = 0
        self._slow_aliases: set[str] = set()
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_nested_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_nested_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_nested_scope(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._scope_depth == 0:
            if any(self._is_pytestmark_target(target) for target in node.targets):
                self._record_if_slow(node.value, node.lineno)
            self._update_slow_aliases(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._scope_depth == 0:
            if self._is_pytestmark_target(node.target) and node.value is not None:
                self._record_if_slow(node.value, node.lineno)
            if node.value is not None:
                self._update_slow_aliases((node.target,), node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._scope_depth == 0:
            if self._is_pytestmark_target(node.target):
                self._record_if_slow(node.value, node.lineno)
            self._discard_slow_alias(node.target)
        self.generic_visit(node)

    def _visit_nested_scope(self, node: ast.AST) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    @staticmethod
    def _is_pytestmark_target(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "pytestmark"
        if isinstance(node, (ast.List, ast.Tuple)):
            return any(
                _ModuleSlowMarkerVisitor._is_pytestmark_target(item) for item in node.elts
            )
        return False

    def _record_if_slow(self, value: ast.AST, lineno: int) -> None:
        if self._contains_slow_marker(value):
            self.violations.append(f"{self._filename}:{lineno}")

    def _contains_slow_marker(self, value: ast.AST) -> bool:
        return any(self._is_slow_marker(node) for node in ast.walk(value))

    def _update_slow_aliases(
        self, targets: tuple[ast.expr, ...] | list[ast.expr], value: ast.AST
    ) -> None:
        for target in targets:
            if isinstance(target, ast.Name) and target.id != "pytestmark":
                if self._contains_slow_marker(value):
                    self._slow_aliases.add(target.id)
                else:
                    self._slow_aliases.discard(target.id)

    def _discard_slow_alias(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._slow_aliases.discard(target.id)

    def _is_slow_marker(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self._slow_aliases
        if not isinstance(node, ast.Attribute) or node.attr != "slow":
            return False
        marker = node.value
        if isinstance(marker, ast.Attribute) and marker.attr == "mark":
            return isinstance(marker.value, ast.Name) and marker.value.id in self._pytest_names
        return isinstance(marker, ast.Name) and marker.id in self._mark_names


def _pytest_marker_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    pytest_names = {"pytest"}
    mark_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    pytest_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name == "mark":
                    mark_names.add(alias.asname or alias.name)
    return pytest_names, mark_names


def test_modules_do_not_assign_slow_to_module_level_pytestmark():
    tests_dir = Path(__file__).parent
    violations = []
    test_paths = {
        path
        for pattern in ("test_*.py", "*_test.py")
        for path in tests_dir.rglob(pattern)
    }
    for path in sorted(test_paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        pytest_names, mark_names = _pytest_marker_names(tree)
        visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, path.name)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    assert violations == []


def test_module_slow_marker_alias_is_detected():
    source = "slow = pytest.mark.slow\npytestmark = slow\n"
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:2"]
