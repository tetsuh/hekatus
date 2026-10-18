"""Mechanical guard for the review-validation marker boundary."""

import ast
from copy import deepcopy
from pathlib import Path


class _ModuleSlowMarkerVisitor(ast.NodeVisitor):
    """Find slow markers assigned at module or top-level class scope."""

    def __init__(self, pytest_names: set[str], mark_names: set[str], filename: str):
        self._pytest_names = set(pytest_names)
        self._mark_names = set(mark_names)
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

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_nested_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._scope_depth == 0:
            for decorator in node.decorator_list:
                self._record_if_slow(decorator, decorator.lineno)

            module_state = (
                self._pytest_names,
                self._mark_names,
                self._list_aliases,
                self._slow_aliases,
            )
            # Keep class-local aliases isolated while retaining shared module-list mutations.
            module_list_groups: dict[int, set[str]] = {}
            for group in self._list_aliases.values():
                module_list_groups.setdefault(id(group), set(group))
            self._pytest_names = set(self._pytest_names)
            self._mark_names = set(self._mark_names)
            self._list_aliases = deepcopy(self._list_aliases)
            class_group_origins = {
                id(self._list_aliases[alias]): group_id
                for group_id, module_aliases in module_list_groups.items()
                for alias in module_aliases
                if alias in self._list_aliases
            }
            self._slow_aliases = set(self._slow_aliases)
            self._scope_depth += 1
            self._module_visible_class_depth += 1
            try:
                for statement in node.body:
                    self.visit(statement)
            finally:
                class_list_aliases = self._list_aliases
                class_slow_aliases = self._slow_aliases
                self._module_visible_class_depth -= 1
                self._scope_depth -= 1
                (
                    self._pytest_names,
                    self._mark_names,
                    self._list_aliases,
                    self._slow_aliases,
                ) = module_state
                for class_group in class_list_aliases.values():
                    group_id = class_group_origins.get(id(class_group))
                    if group_id is not None and class_group.intersection(class_slow_aliases):
                        self._slow_aliases.update(module_list_groups[group_id])
        else:
            self._visit_nested_scope(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_tracked_scope():
            if any(
                self._is_pytestmark_target(target)
                or self._is_pytestmark_list_subscript_target(target)
                for target in node.targets
            ):
                self._record_if_slow(node.value, node.lineno)
            for target in node.targets:
                if self._is_tracked_list_subscript_target(target):
                    self._mark_slow_subscript_target(target, node.value)
            self._update_slow_aliases(node.targets, node.value)
            self._update_mark_aliases(node.targets, node.value)
            self._update_pytest_aliases(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._is_tracked_scope():
            if self._is_pytestmark_target(node.target) and node.value is not None:
                self._record_if_slow(node.value, node.lineno)
            if node.value is not None:
                self._update_slow_aliases((node.target,), node.value)
                self._update_mark_aliases((node.target,), node.value)
                self._update_pytest_aliases((node.target,), node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._is_tracked_scope():
            if self._is_pytestmark_target(node.target):
                self._record_if_slow(node.value, node.lineno)
                if self._contains_slow_marker(node.value):
                    if isinstance(node.target, ast.Name):
                        self._mark_list_alias_slow(node.target.id)
                    else:
                        self._mark_slow_subscript_target(node.target, node.value)
            elif isinstance(node.target, ast.Name) and node.target.id in self._list_aliases:
                alias = node.target.id
                if self._contains_slow_marker(node.value):
                    self._mark_list_alias_slow(alias)
                    if self._is_current_pytestmark_list(alias):
                        self._record_if_slow(node.value, node.lineno)
            elif self._is_tracked_list_subscript_target(node.target):
                if self._is_pytestmark_list_subscript_target(node.target):
                    self._record_if_slow(node.value, node.lineno)
                self._mark_slow_subscript_target(node.target, node.value)
            else:
                self._discard_slow_alias(node.target)
            self._discard_mark_alias(node.target)
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
            elif (is_module_scope or is_visible_class_scope) and alias in self._list_aliases:
                if self._call_receives_slow_marker(node):
                    self._mark_list_alias_slow(alias)
                    if self._is_current_pytestmark_list(alias):
                        self._record_if_slow(node, node.lineno)
        self.generic_visit(node)

    def _visit_nested_scope(self, node: ast.AST) -> None:
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def _is_tracked_scope(self) -> bool:
        return self._scope_depth == 0 or (
            self._scope_depth == 1 and self._module_visible_class_depth > 0
        )

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

    def _is_tracked_list_subscript_target(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Subscript):
            return False
        root = self._subscript_root_name(node)
        return root == "pytestmark" or (
            root is not None and root in self._list_aliases
        )

    def _is_pytestmark_list_subscript_target(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Subscript):
            return False
        root = self._subscript_root_name(node)
        return root == "pytestmark" or (
            root is not None and self._is_current_pytestmark_list(root)
        )

    @staticmethod
    def _subscript_root_name(node: ast.AST) -> str | None:
        while isinstance(node, ast.Subscript):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def _mark_slow_subscript_target(self, target: ast.AST, value: ast.AST) -> None:
        if self._contains_slow_marker(value):
            root = self._subscript_root_name(target)
            if root is not None:
                self._mark_list_alias_slow(root)

    def _record_if_slow(self, value: ast.AST, lineno: int) -> None:
        if self._contains_slow_marker(value):
            self.violations.append(f"{self._filename}:{lineno}")

    def _contains_slow_marker(self, value: ast.AST) -> bool:
        return any(self._is_slow_marker(node) for node in ast.walk(value))

    def _call_receives_slow_marker(self, node: ast.Call) -> bool:
        return any(self._contains_slow_marker(argument) for argument in node.args) or any(
            self._contains_slow_marker(keyword.value) for keyword in node.keywords
        )

    def _update_mark_aliases(
        self, targets: tuple[ast.expr, ...] | list[ast.expr], value: ast.AST
    ) -> None:
        for target in targets:
            pairs = self._matching_alias_pairs(target, value)
            if pairs is None:
                continue
            for alias_target, alias_value in pairs:
                if self._is_pytest_mark_namespace(alias_value):
                    self._mark_names.add(alias_target.id)
                else:
                    self._mark_names.discard(alias_target.id)

    def _is_pytest_mark_namespace(self, value: ast.AST) -> bool:
        return (
            isinstance(value, ast.Attribute)
            and value.attr == "mark"
            and isinstance(value.value, ast.Name)
            and value.value.id in self._pytest_names
        )

    def _update_pytest_aliases(
        self, targets: tuple[ast.expr, ...] | list[ast.expr], value: ast.AST
    ) -> None:
        if not isinstance(value, ast.Name) or value.id not in self._pytest_names:
            return
        for target in targets:
            if isinstance(target, ast.Name):
                self._pytest_names.add(target.id)

    def _update_slow_aliases(
        self, targets: tuple[ast.expr, ...] | list[ast.expr], value: ast.AST
    ) -> None:
        for target in targets:
            pairs = self._matching_alias_pairs(target, value)
            if pairs is None:
                continue
            for alias_target, alias_value in pairs:
                self._update_slow_alias(alias_target, alias_value)

    @classmethod
    def _matching_alias_pairs(
        cls, target: ast.AST, value: ast.AST
    ) -> list[tuple[ast.Name, ast.AST]] | None:
        if isinstance(target, ast.Name):
            return [(target, value)]
        if not isinstance(target, (ast.List, ast.Tuple)) or not isinstance(
            value, (ast.List, ast.Tuple)
        ):
            return None
        if len(target.elts) != len(value.elts):
            return None
        pairs: list[tuple[ast.Name, ast.AST]] = []
        for target_item, value_item in zip(target.elts, value.elts):
            item_pairs = cls._matching_alias_pairs(target_item, value_item)
            if item_pairs is None:
                return None
            pairs.extend(item_pairs)
        return pairs

    def _update_slow_alias(self, target: ast.Name, value: ast.AST) -> None:
        source_group = (
            self._list_aliases.get(value.id)
            if isinstance(value, ast.Name)
            else None
        )
        target_group = self._remove_list_alias(target.id)
        if isinstance(value, ast.List):
            target_group = {target.id}
            self._list_aliases[target.id] = target_group
        elif source_group is not None:
            source_group.add(target.id)
            target_group = source_group
            self._list_aliases[target.id] = target_group

        contains_slow = self._contains_slow_marker(value)
        if target_group is not None and contains_slow:
            self._slow_aliases.update(target_group)
        elif contains_slow:
            self._slow_aliases.add(target.id)
        else:
            self._slow_aliases.discard(target.id)

    def _discard_slow_alias(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._slow_aliases.discard(target.id)

    def _discard_mark_alias(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._mark_names.discard(target.id)

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


def test_modules_do_not_assign_slow_to_module_or_class_pytestmark():
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


def test_module_slow_marker_pytest_module_alias_is_detected():
    source = (
        "import pytest as pt\n"
        "pytest_api = pt\n"
        "pytestmark = pytest_api.mark.slow\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3"]


def test_module_slow_marker_pytest_mark_alias_is_detected():
    source = (
        "import pytest\n"
        "m = pytest.mark\n"
        "pytestmark = m.slow\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3"]


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


def test_module_slow_marker_pytestmark_alias_subscript_mutation_is_detected():
    source = (
        "pytestmark = []\n"
        "marks = pytestmark\n"
        "marks[:] = [pytest.mark.slow]\n"
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

    assert visitor.violations == [
        "synthetic.py:2",
        "synthetic.py:3",
        "synthetic.py:4",
        "synthetic.py:9",
    ]


def test_module_slow_marker_class_decorator_is_detected():
    source = (
        "import pytest as pt\n"
        "from pytest import mark as pm\n"
        "pytest_api = pt\n"
        "slow = pytest.mark.slow\n"
        "@pytest.mark.slow\n"
        "class Direct:\n"
        "    pass\n"
        "@pytest_api.mark.slow\n"
        "class PytestAlias:\n"
        "    pass\n"
        "@pm.slow\n"
        "class MarkAlias:\n"
        "    pass\n"
        "@slow\n"
        "class MarkerAlias:\n"
        "    pass\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == [
        "synthetic.py:5",
        "synthetic.py:8",
        "synthetic.py:11",
        "synthetic.py:14",
    ]


def test_module_slow_marker_class_pytestmark_assignment_is_detected():
    source = (
        "slow = pytest.mark.slow\n"
        "class Direct:\n"
        "    pytestmark = pytest.mark.slow\n"
        "class Alias:\n"
        "    pytestmark = slow\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3", "synthetic.py:5"]


def test_module_slow_marker_class_list_assignment_is_detected():
    source = (
        "class Direct:\n"
        "    pytestmark = [pytest.mark.slow]\n"
        "class Alias:\n"
        "    slow = pytest.mark.slow\n"
        "    pytestmark = [slow]\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:2", "synthetic.py:5"]


def test_module_slow_marker_class_pytestmark_augassign_after_initialization_is_detected():
    source = (
        "class Direct:\n"
        "    pytestmark = []\n"
        "    pytestmark += [pytest.mark.slow]\n"
        "class Annotated:\n"
        "    pytestmark: list = []\n"
        "    pytestmark += [pytest.mark.slow]\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3", "synthetic.py:6"]


def test_module_slow_marker_class_pytestmark_mutations_after_initialization_are_detected():
    source = (
        "slow = pytest.mark.slow\n"
        "class Append:\n"
        "    pytestmark = []\n"
        "    pytestmark.append(slow)\n"
        "class Extend:\n"
        "    pytestmark = []\n"
        "    pytestmark.extend([slow])\n"
        "class Insert:\n"
        "    pytestmark = []\n"
        "    pytestmark.insert(0, slow)\n"
        "class Slice:\n"
        "    pytestmark = []\n"
        "    pytestmark[:] = [slow]\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == [
        "synthetic.py:4",
        "synthetic.py:7",
        "synthetic.py:10",
        "synthetic.py:13",
    ]


def test_module_slow_marker_class_alias_list_mutations_are_detected():
    source = (
        "slow = pytest.mark.slow\n"
        "class AliasAugAssign:\n"
        "    marks = []\n"
        "    marks += [slow]\n"
        "    pytestmark = marks\n"
        "class AliasAppend:\n"
        "    pytestmark = []\n"
        "    marks = pytestmark\n"
        "    marks.append(slow)\n"
        "class AliasSlice:\n"
        "    marks = []\n"
        "    marks[:] = [slow]\n"
        "    pytestmark = marks\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == [
        "synthetic.py:5",
        "synthetic.py:9",
        "synthetic.py:13",
    ]


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


def test_module_slow_marker_class_body_alias_mutation_is_detected():
    source = (
        "pytestmark = []\n"
        "marks = pytestmark\n"
        "class Marker:\n"
        "    marks.append(pytest.mark.slow)\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:4"]


def test_module_slow_marker_class_body_tracks_local_binding():
    source = (
        "pytestmark = []\n"
        "class Marker:\n"
        "    pytestmark.append(pytest.mark.slow)\n"
        "    def add_marker():\n"
        "        pytestmark.append(pytest.mark.slow)\n"
        "    class Nested:\n"
        "        pytestmark.append(pytest.mark.slow)\n"
        "    pytestmark = []\n"
        "    pytestmark.append(pytest.mark.slow)\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3", "synthetic.py:9"]


def test_module_slow_marker_excludes_nested_and_non_slow_class_scopes():
    source = (
        "marks = []\n"
        "@pytest.mark.slow\n"
        "def function_scope():\n"
        "    pytestmark = pytest.mark.slow\n"
        "    pytestmark.append(pytest.mark.slow)\n"
        "class Outer:\n"
        "    pytestmark = []\n"
        "    pytestmark += [pytest.mark.integration]\n"
        "    @pytest.mark.slow\n"
        "    class Nested:\n"
        "        pytestmark = pytest.mark.slow\n"
        "    @pytest.mark.slow\n"
        "    def method(self):\n"
        "        pytestmark = pytest.mark.slow\n"
        "class LocalAliases:\n"
        "    slow = pytest.mark.slow\n"
        "    marks = []\n"
        "    marks.append(slow)\n"
        "class UsesOnlyModuleNames:\n"
        "    pytestmark = marks\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == []


def test_module_slow_marker_destructured_alias_is_detected():
    source = (
        "tuple_slow, = (pytest.mark.slow,)\n"
        "list_slow, = [pytest.mark.slow]\n"
        "pytestmark = tuple_slow\n"
        "pytestmark = list_slow\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == ["synthetic.py:3", "synthetic.py:4"]


def test_module_slow_marker_non_slow_pytestmark_augassign_is_ignored():
    source = (
        "pytestmark = []\n"
        "pytestmark += [pytest.mark.integration]\n"
        "alias = pytestmark\n"
        "pytestmark = alias\n"
    )
    tree = ast.parse(source, filename="synthetic.py")
    pytest_names, mark_names = _pytest_marker_names(tree)
    visitor = _ModuleSlowMarkerVisitor(pytest_names, mark_names, "synthetic.py")
    visitor.visit(tree)

    assert visitor.violations == []
