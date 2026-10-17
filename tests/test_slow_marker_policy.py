"""Mechanical guard for the review-validation marker boundary."""

import ast
from pathlib import Path


class _ClassPytestmarkBindingVisitor(ast.NodeVisitor):
    """Find pytestmark bindings in a class body, excluding nested scopes."""

    def __init__(self) -> None:
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "pytestmark" and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.found = True


class _ModuleSlowMarkerVisitor(ast.NodeVisitor):
    """Find slow markers assigned to the module-level pytestmark name."""

    def __init__(self, pytest_names: set[str], mark_names: set[str], filename: str):
        self._pytest_names = pytest_names
        self._mark_names = mark_names
        self._filename = filename
        self._scope_depth = 0
        self._module_visible_class_depth = 0
        self._list_aliases: dict[str, set[str]] = {}
        self._slow_aliases: set[str] = set()
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_nested_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_nested_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._scope_depth == 0 and not self._class_has_pytestmark_binding(node):
            self._scope_depth += 1
            self._module_visible_class_depth += 1
            self.generic_visit(node)
            self._module_visible_class_depth -= 1
            self._scope_depth -= 1
        else:
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
                if isinstance(node.target, ast.Name):
                    self._mark_list_alias_slow(node.target.id)
            elif isinstance(node.target, ast.Name) and node.target.id in self._list_aliases:
                alias = node.target.id
                if self._contains_slow_marker(node.value):
                    self._mark_list_alias_slow(alias)
                    if self._is_current_pytestmark_list(alias):
                        self._record_if_slow(node.value, node.lineno)
            else:
                self._discard_slow_alias(node.target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        is_module_scope = self._scope_depth == 0
        is_visible_class_scope = (
            self._scope_depth == 1 and self._module_visible_class_depth > 0
        )
        if (
            (is_module_scope or is_visible_class_scope)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "extend", "insert"}
            and isinstance(node.func.value, ast.Name)
        ):
            alias = node.func.value.id
            if alias == "pytestmark":
                self._record_if_slow(node, node.lineno)
                if self._call_receives_slow_marker(node):
                    self._mark_list_alias_slow(alias)
            elif is_module_scope and alias in self._list_aliases:
                if self._call_receives_slow_marker(node):
                    self._mark_list_alias_slow(alias)
                    if self._is_current_pytestmark_list(alias):
                        self._record_if_slow(node, node.lineno)
        self.generic_visit(node)

    def _visit_nested_scope(self, node: ast.AST) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    @staticmethod
    def _class_has_pytestmark_binding(node: ast.ClassDef) -> bool:
        visitor = _ClassPytestmarkBindingVisitor()
        for statement in node.body:
            visitor.visit(statement)
        return visitor.found

    def _remove_list_alias(self, alias: str) -> set[str] | None:
        group = self._list_aliases.pop(alias, None)
        if group is not None:
            group.discard(alias)
        return group

    def _mark_list_alias_slow(self, alias: str) -> None:
        group = self._list_aliases.get(alias)
        if group is None:
            self._slow_aliases.add(alias)
        else:
            self._slow_aliases.update(group)

    def _is_current_pytestmark_list(self, alias: str) -> bool:
        group = self._list_aliases.get(alias)
        return group is not None and "pytestmark" in group

    @staticmethod
    def _is_pytestmark_target(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "pytestmark"
        if isinstance(node, ast.Subscript):
            return isinstance(node.value, ast.Name) and node.value.id == "pytestmark"
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

    def _call_receives_slow_marker(self, node: ast.Call) -> bool:
        return any(self._contains_slow_marker(argument) for argument in node.args) or any(
            self._contains_slow_marker(keyword.value) for keyword in node.keywords
        )

    def _update_slow_aliases(
        self, targets: tuple[ast.expr, ...] | list[ast.expr], value: ast.AST
    ) -> None:
        source_group = (
            self._list_aliases.get(value.id)
            if isinstance(value, ast.Name)
            else None
        )
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            target_group = self._remove_list_alias(target.id)
            if isinstance(value, ast.List):
                target_group = {target.id}
                self._list_aliases[target.id] = target_group
            elif source_group is not None:
                source_group.add(target.id)
                target_group = source_group
                self._list_aliases[target.id] = target_group

            if target_group is not None and self._contains_slow_marker(value):
                self._slow_aliases.update(target_group)
            elif self._contains_slow_marker(value):
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


def test_module_slow_marker_list_alias_mutation_is_detected():
    source = (
        "marks = []\n"
        "marks.append(pytest.mark.slow)\n"
        "pytestmark = marks\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3"]


def test_module_slow_marker_list_alias_chain_mutation_is_detected():
    source = (
        "marks = []\n"
        "alias = marks\n"
        "alias.append(pytest.mark.slow)\n"
        "pytestmark = marks\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:4"]


def test_module_slow_marker_list_alias_augassign_is_detected():
    source = (
        "marks = []\n"
        "marks += [pytest.mark.slow]\n"
        "pytestmark = marks\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3"]


def test_module_slow_marker_pytestmark_alias_mutation_is_detected():
    source = (
        "pytestmark = []\n"
        "marks = pytestmark\n"
        "marks.append(pytest.mark.slow)\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3"]


def test_module_slow_marker_subscript_mutation_is_detected():
    source = (
        "pytestmark = []\n"
        "pytestmark[:] = [pytest.mark.slow]\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:2"]


def test_module_slow_marker_mutation_is_detected():
    source = (
        "pytestmark = []\n"
        "pytestmark.append(pytest.mark.slow)\n"
        "pytestmark.extend([pytest.mark.slow])\n"
        "pytestmark.insert(0, pytest.mark.slow)\n"
        "def add_marker():\n"
        "    pytestmark.append(pytest.mark.slow)\n"
        "class Marker:\n"
        "    pytestmark = []\n"
        "    pytestmark.extend([pytest.mark.slow])\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:2", "synthetic.py:3", "synthetic.py:4"]


def test_module_slow_marker_class_body_mutation_is_detected():
    source = (
        "pytestmark = []\n"
        "class Marker:\n"
        "    pytestmark.extend([pytest.mark.slow])\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3"]
