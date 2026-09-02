"""Regression checks for the shipped RoboTwin inference configuration."""

import ast
from pathlib import Path


def _constant_assignment(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == name
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, int):
                return value
    raise AssertionError(f"Could not find integer assignment for {name}")


def test_robotwin_chunk_includes_wan_temporal_context() -> None:
    config_path = (
        Path(__file__).parents[1] / "wan_va" / "configs" / "va_robotwin_cfg.py"
    )
    # Wan2.2's causal temporal convolution has a three-frame kernel.
    assert _constant_assignment(config_path, "frame_chunk_size") >= 3
