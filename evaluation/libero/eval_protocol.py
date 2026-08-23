"""Protocol planning and result helpers for LIBERO evaluation.

This module intentionally depends only on the Python standard library so that
the protocol invariants can be tested without installing MuJoCo, LIBERO, or the
LingBot-VA runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOLS = ("original", "plus")
ORIGINAL_SUITE_TASK_COUNT = 10
LIBERO_PLUS_SUITE_TASK_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
# Canonical JSON hash of task_classification.json at LIBERO-Plus commit
# 4976dc30028e805ff8094b55501d532c48fec182. Canonicalization makes this
# independent of whitespace and line-ending conversion.
LIBERO_PLUS_CLASSIFICATION_SEMANTIC_SHA256 = (
    "84b63b9d836146286d62f6d2aafea15a8a68fd197bec1cba36930f22da9143ce"
)


class EvaluationProtocolError(ValueError):
    """Raised when an evaluation request mixes incompatible protocols."""


@dataclass(frozen=True)
class EvaluationPlan:
    """Resolved, validated task and trial selection."""

    protocol: str
    benchmark_total_tasks: int
    task_start: int
    task_end: int
    trials_per_task: int

    @property
    def selected_task_count(self) -> int:
        return self.task_end - self.task_start

    @property
    def denominator(self) -> int:
        return self.selected_task_count * self.trials_per_task

    @property
    def selection_unit(self) -> str:
        return "task" if self.protocol == "original" else "variant"

    @property
    def task_indices(self) -> range:
        return range(self.task_start, self.task_end)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "protocol": self.protocol,
            "benchmark_total_tasks": self.benchmark_total_tasks,
            "task_range": [self.task_start, self.task_end],
            "selection_unit": self.selection_unit,
            "selected_entries": self.selected_task_count,
            "trials_per_entry": self.trials_per_task,
            "denominator": self.denominator,
        }
        result[f"selected_{self.selection_unit}s"] = self.selected_task_count
        result[f"trials_per_{self.selection_unit}"] = self.trials_per_task
        return result


@dataclass(frozen=True)
class TaskMetadata:
    """Stable task metadata written next to each evaluation result."""

    task_id: int
    name: str
    category: str
    difficulty_level: int | None


@dataclass(frozen=True)
class ClassificationBundle:
    """Validated LIBERO-Plus classification data and its provenance."""

    tasks: tuple[TaskMetadata, ...]
    source: str
    sha256: str
    semantic_sha256: str


def add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    """Add protocol arguments shared by the command line client and tests."""

    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default="original",
        help=(
            "Evaluation protocol. 'original' uses 50 trials per task; 'plus' "
            "uses exactly one trial per perturbation variant."
        ),
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help=(
            "Optional half-open task/variant slice [START, END). By default, "
            "all tasks or variants in the selected suite are evaluated."
        ),
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=None,
        help=(
            "Trials per task. Defaults to 50 for original LIBERO and 1 for "
            "LIBERO-Plus. Values other than 1 are rejected in plus mode."
        ),
    )


def normalize_checkpoint_id(protocol: str, checkpoint_id: str | None) -> str | None:
    """Normalize declared checkpoint identity and require it for Plus resume."""

    if protocol not in PROTOCOLS:
        raise EvaluationProtocolError(
            f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}."
        )
    if checkpoint_id is None:
        if protocol == "plus":
            raise EvaluationProtocolError(
                "LIBERO-Plus requires --checkpoint-id NAME@REVISION so an "
                "interrupted run cannot resume results from an unidentified "
                "checkpoint."
            )
        return None
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise EvaluationProtocolError(
            "--checkpoint-id must be non-empty when it is provided."
        )
    return checkpoint_id.strip()


def resolve_evaluation_plan(
    protocol: str,
    benchmark_total_tasks: int,
    task_range: Sequence[int] | None = None,
    test_num: int | None = None,
) -> EvaluationPlan:
    """Resolve defaults and reject protocol/task/trial mismatches."""

    if protocol not in PROTOCOLS:
        raise EvaluationProtocolError(
            f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}."
        )
    if benchmark_total_tasks <= 0:
        raise EvaluationProtocolError(
            f"Benchmark must contain tasks, got {benchmark_total_tasks}."
        )

    trials_per_task = (
        (50 if protocol == "original" else 1) if test_num is None else test_num
    )
    if trials_per_task <= 0:
        raise EvaluationProtocolError(
            f"--test-num must be positive, got {trials_per_task}."
        )
    if protocol == "plus" and trials_per_task != 1:
        raise EvaluationProtocolError(
            "LIBERO-Plus requires exactly one trial per variant. Omit "
            "--test-num or set --test-num 1."
        )

    if task_range is None:
        task_start, task_end = 0, benchmark_total_tasks
    else:
        if len(task_range) != 2:
            raise EvaluationProtocolError(
                "--task-range must contain exactly START END for [START, END)."
            )
        task_start, task_end = task_range

    if task_start < 0 or task_start >= task_end or task_end > benchmark_total_tasks:
        raise EvaluationProtocolError(
            "Invalid --task-range "
            f"[{task_start}, {task_end}) for a benchmark with "
            f"{benchmark_total_tasks} tasks."
        )

    return EvaluationPlan(
        protocol=protocol,
        benchmark_total_tasks=benchmark_total_tasks,
        task_start=task_start,
        task_end=task_end,
        trials_per_task=trials_per_task,
    )


def validate_benchmark_shape(
    protocol: str,
    benchmark_total_tasks: int,
    suite: str | None = None,
) -> None:
    """Fail closed when the installed LIBERO package is for another protocol."""

    if protocol == "original" and benchmark_total_tasks != ORIGINAL_SUITE_TASK_COUNT:
        raise EvaluationProtocolError(
            "The original LIBERO protocol expects a 10-task suite, but the "
            f"installed benchmark exposes {benchmark_total_tasks} tasks. This "
            "looks like LIBERO-Plus; rerun with --protocol plus or install the "
            "original LIBERO package."
        )
    if protocol == "plus":
        if suite is None:
            raise EvaluationProtocolError(
                "LIBERO-Plus validation requires the benchmark suite name so "
                "the exact official variant count can be checked."
            )
        expected_count = LIBERO_PLUS_SUITE_TASK_COUNTS.get(suite)
        if expected_count is None:
            raise EvaluationProtocolError(
                f"No canonical LIBERO-Plus task count is registered for {suite!r}."
            )
        if benchmark_total_tasks == expected_count:
            return
        raise EvaluationProtocolError(
            f"Canonical LIBERO-Plus {suite} contains exactly {expected_count} "
            f"variants, but the installed benchmark exposes {benchmark_total_tasks}. "
            "Refusing to label a partial or modified task set as LIBERO-Plus."
        )


def sha256_file(path: str | Path) -> str:
    """Return a content hash suitable for benchmark provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash JSON data independently of whitespace, indentation, and key order."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_plus_task_metadata(
    classification_path: str | Path,
    suite: str,
    task_names: Sequence[str],
    expected_semantic_sha256: str | None = None,
) -> ClassificationBundle:
    """Load and exactly align LIBERO-Plus task classification metadata.

    The exact length, 1-based IDs, and task names are checked before any
    rollout. This prevents a stale or mismatched classification file from
    silently assigning the wrong perturbation categories.
    """

    path = Path(classification_path).resolve()
    if not path.is_file():
        raise EvaluationProtocolError(
            "LIBERO-Plus task classification file was not found at "
            f"{path}. Install the official LIBERO-Plus benchmark or pass the "
            "matching --task-classification file."
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationProtocolError(
            f"Could not read LIBERO-Plus classification file {path}: {exc}"
        ) from exc

    semantic_sha256 = canonical_json_sha256(payload)
    if (
        expected_semantic_sha256 is not None
        and semantic_sha256 != expected_semantic_sha256
    ):
        raise EvaluationProtocolError(
            "LIBERO-Plus task metadata does not match the pinned official "
            "classification. "
            f"Expected semantic SHA-256 {expected_semantic_sha256}, got "
            f"{semantic_sha256} from {path}."
        )

    if not isinstance(payload, dict) or suite not in payload:
        raise EvaluationProtocolError(
            f"Classification file {path} has no entry for suite {suite!r}."
        )
    raw_tasks = payload[suite]
    if not isinstance(raw_tasks, list):
        raise EvaluationProtocolError(
            f"Classification entry for {suite!r} must be a list."
        )
    if len(raw_tasks) != len(task_names):
        raise EvaluationProtocolError(
            "LIBERO-Plus classification/benchmark size mismatch for "
            f"{suite}: {len(raw_tasks)} metadata entries versus "
            f"{len(task_names)} benchmark variants."
        )

    tasks: list[TaskMetadata] = []
    for task_index, (raw_task, benchmark_name) in enumerate(
        zip(raw_tasks, task_names, strict=True)
    ):
        if not isinstance(raw_task, dict):
            raise EvaluationProtocolError(
                f"Classification entry {task_index} for {suite} is not an object."
            )

        expected_id = task_index + 1
        task_id = raw_task.get("id")
        name = raw_task.get("name")
        category = raw_task.get("category")
        difficulty_level = raw_task.get("difficulty_level")

        if task_id != expected_id:
            raise EvaluationProtocolError(
                f"Classification entry {task_index} has id {task_id!r}; "
                f"expected the 1-based id {expected_id}."
            )
        if name != benchmark_name:
            raise EvaluationProtocolError(
                "LIBERO-Plus classification/benchmark order mismatch at index "
                f"{task_index}: metadata={name!r}, benchmark={benchmark_name!r}."
            )
        if not isinstance(category, str) or not category.strip():
            raise EvaluationProtocolError(
                f"Classification entry {task_index} has no valid category."
            )
        if difficulty_level is not None and (
            not isinstance(difficulty_level, int) or isinstance(difficulty_level, bool)
        ):
            raise EvaluationProtocolError(
                "Classification entry "
                f"{task_index} has invalid difficulty_level "
                f"{difficulty_level!r}."
            )

        tasks.append(
            TaskMetadata(
                task_id=task_id,
                name=name,
                category=category,
                difficulty_level=difficulty_level,
            )
        )

    return ClassificationBundle(
        tasks=tuple(tasks),
        source=str(path),
        sha256=sha256_file(path),
        semantic_sha256=semantic_sha256,
    )


def resolve_prompt(
    protocol: str,
    benchmark_prompt: str,
    environment_prompt: str | None,
) -> str:
    """Use BDDL language in Plus mode without changing original behavior.

    Some LIBERO-Plus task filenames encode perturbation metadata. The BDDL
    ``:language`` field is the task instruction and is exposed by the created
    environment, so Plus evaluation uses it rather than filename-derived text.
    """

    if protocol == "original":
        return benchmark_prompt
    if not isinstance(environment_prompt, str) or not environment_prompt.strip():
        raise EvaluationProtocolError(
            "LIBERO-Plus environment did not expose a non-empty BDDL language "
            "instruction; refusing to evaluate with filename-derived text."
        )
    return environment_prompt.strip()


def aggregate_task_results(
    task_results: Iterable[Mapping[str, Any]],
    selection_unit: str = "variant",
) -> dict[str, Any]:
    """Aggregate explicit success numerators and rollout denominators."""

    if selection_unit not in {"task", "variant"}:
        raise EvaluationProtocolError(
            f"Unknown selection unit {selection_unit!r}; expected task or variant."
        )

    successes = 0
    denominator = 0
    completed_entries = 0
    by_category: dict[str, dict[str, int]] = {}

    for result in task_results:
        task_successes = result.get("successes")
        task_denominator = result.get("denominator")
        if (
            not isinstance(task_successes, int)
            or isinstance(task_successes, bool)
            or not isinstance(task_denominator, int)
            or isinstance(task_denominator, bool)
            or task_successes < 0
            or task_denominator <= 0
            or task_successes > task_denominator
        ):
            raise EvaluationProtocolError(
                "Each task result must have integer 0 <= successes <= "
                "denominator and denominator > 0."
            )

        category_value = result.get("category", "Unclassified")
        category = (
            category_value
            if isinstance(category_value, str) and category_value
            else "Unclassified"
        )
        category_totals = by_category.setdefault(
            category, {"completed_entries": 0, "successes": 0, "denominator": 0}
        )
        category_totals["completed_entries"] += 1
        category_totals["successes"] += task_successes
        category_totals["denominator"] += task_denominator

        completed_entries += 1
        successes += task_successes
        denominator += task_denominator

    category_results: dict[str, dict[str, int | float | None]] = {}
    for category, totals in sorted(by_category.items()):
        category_denominator = totals["denominator"]
        category_results[category] = {
            **totals,
            f"completed_{selection_unit}s": totals["completed_entries"],
            "observed_success_rate": totals["successes"] / category_denominator,
        }

    return {
        "completed_entries": completed_entries,
        f"completed_{selection_unit}s": completed_entries,
        "successes": successes,
        "denominator": denominator,
        "observed_success_rate": successes / denominator if denominator else None,
        "by_category": category_results,
    }


def describe_evaluation_status(
    plan: EvaluationPlan,
    completed: Mapping[str, Any],
) -> dict[str, Any]:
    """Distinguish diagnostics for prefixes/shards from a full-suite score."""

    completed_entries = completed.get("completed_entries")
    completed_denominator = completed.get("denominator")
    if (
        not isinstance(completed_entries, int)
        or isinstance(completed_entries, bool)
        or completed_entries < 0
        or completed_entries > plan.selected_task_count
        or not isinstance(completed_denominator, int)
        or isinstance(completed_denominator, bool)
        or completed_denominator < 0
        or completed_denominator > plan.denominator
    ):
        raise EvaluationProtocolError(
            "Completed entries/denominator must stay within the selected plan."
        )

    is_full_suite = plan.task_start == 0 and plan.task_end == plan.benchmark_total_tasks
    is_plan_complete = (
        completed_entries == plan.selected_task_count
        and completed_denominator == plan.denominator
    )
    canonical_trials_per_entry = 50 if plan.protocol == "original" else 1
    uses_canonical_trial_count = plan.trials_per_task == canonical_trials_per_entry
    reportable = is_full_suite and is_plan_complete and uses_canonical_trial_count
    official_full_suite_denominator = (
        plan.benchmark_total_tasks * canonical_trials_per_entry
    )

    if reportable:
        status = "complete"
        score_reason = "complete_full_suite"
    elif is_full_suite and is_plan_complete:
        status = "complete_noncanonical"
        score_reason = "noncanonical_trial_count"
    elif is_plan_complete:
        status = "complete_shard"
        score_reason = "intentional_shard"
    else:
        status = "in_progress"
        score_reason = "incomplete_run"

    return {
        "status": status,
        "scope": "full_suite" if is_full_suite else "shard",
        "is_plan_complete": is_plan_complete,
        "is_full_suite": is_full_suite,
        "uses_canonical_trial_count": uses_canonical_trial_count,
        "canonical_trials_per_entry": canonical_trials_per_entry,
        "reportable_as_full_suite_score": reportable,
        "score_reason": score_reason,
        "official_full_suite_denominator": official_full_suite_denominator,
        "planned_coverage_fraction": plan.denominator / official_full_suite_denominator,
        "completed_coverage_fraction": (
            completed_denominator / official_full_suite_denominator
        ),
    }


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object and convert corruption into a protocol error."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationProtocolError(
            f"Could not read JSON file {input_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvaluationProtocolError(f"JSON file {input_path} must contain an object.")
    return payload


def load_or_create_manifest(
    payload: Mapping[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    """Create a manifest once, or verify an existing resume manifest exactly."""

    output_path = Path(path)
    expected = dict(payload)
    if output_path.exists():
        existing = read_json_object(output_path)
        existing_stable = {
            key: value for key, value in existing.items() if key != "created_at_utc"
        }
        expected_stable = {
            key: value for key, value in expected.items() if key != "created_at_utc"
        }
        if existing_stable != expected_stable:
            raise EvaluationProtocolError(
                f"Existing run manifest {output_path} does not match this run; "
                "refusing to mix results."
            )
        return existing

    write_json_atomic(expected, output_path)
    return expected


def load_resumable_task_result(
    path: str | Path,
    *,
    run_fingerprint: str,
    manifest_reference: str,
    manifest_sha256: str,
    expected_task_fields: Mapping[str, Any],
    trials_per_task: int,
) -> dict[str, Any] | None:
    """Load a matching task result; ignore older runs and fail on corruption."""

    input_path = Path(path)
    if not input_path.is_file():
        return None

    result = read_json_object(input_path)
    if result.get("run_fingerprint") != run_fingerprint:
        return None

    for field, expected_value in expected_task_fields.items():
        if result.get(field) != expected_value:
            raise EvaluationProtocolError(
                f"Resumable result {input_path} has {field}={result.get(field)!r}; "
                f"expected {expected_value!r}."
            )

    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise EvaluationProtocolError(
            f"Resumable result {input_path} has no valid provenance object."
        )
    if (
        provenance.get("manifest") != manifest_reference
        or provenance.get("manifest_sha256") != manifest_sha256
    ):
        raise EvaluationProtocolError(
            f"Resumable result {input_path} does not reference the immutable "
            "manifest for this run."
        )

    successes = result.get("successes")
    denominator = result.get("denominator")
    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or successes < 0
        or denominator <= 0
        or denominator > trials_per_task
        or successes > denominator
    ):
        raise EvaluationProtocolError(
            f"Resumable result {input_path} has an invalid success denominator."
        )

    expected_rate = successes / denominator
    reported_rate = result.get("success_rate")
    if (
        not isinstance(reported_rate, (int, float))
        or isinstance(reported_rate, bool)
        or abs(reported_rate - expected_rate) > 1e-12
    ):
        raise EvaluationProtocolError(
            f"Resumable result {input_path} has an inconsistent success rate."
        )
    prompt = result.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise EvaluationProtocolError(
            f"Resumable result {input_path} has no recorded task instruction."
        )

    return result


def write_json_atomic(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write JSON without leaving a partially written result file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline="\n"
        ) as file_obj:
            file_obj.write(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
