import argparse
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "libero"))

from eval_protocol import (  # noqa: E402
    EvaluationProtocolError,
    LIBERO_PLUS_CLASSIFICATION_SEMANTIC_SHA256,
    LIBERO_PLUS_SUITE_TASK_COUNTS,
    add_protocol_arguments,
    aggregate_task_results,
    describe_evaluation_status,
    load_plus_task_metadata,
    load_or_create_manifest,
    load_resumable_task_result,
    normalize_checkpoint_id,
    resolve_evaluation_plan,
    resolve_prompt,
    sha256_file,
    validate_benchmark_shape,
    write_json_atomic,
)


class EvaluationPlanTest(unittest.TestCase):
    def test_plus_requires_declared_checkpoint_for_safe_resume(self):
        self.assertEqual(
            normalize_checkpoint_id("plus", "  model@revision  "),
            "model@revision",
        )
        self.assertIsNone(normalize_checkpoint_id("original", None))
        with self.assertRaisesRegex(
            EvaluationProtocolError, "requires --checkpoint-id"
        ):
            normalize_checkpoint_id("plus", None)

    def test_cli_defaults_preserve_original_protocol(self):
        parser = argparse.ArgumentParser()
        add_protocol_arguments(parser)

        args = parser.parse_args([])
        plan = resolve_evaluation_plan(
            args.protocol,
            benchmark_total_tasks=10,
            task_range=args.task_range,
            test_num=args.test_num,
        )

        self.assertEqual(args.protocol, "original")
        self.assertEqual(list(plan.task_indices), list(range(10)))
        self.assertEqual(plan.trials_per_task, 50)
        self.assertEqual(plan.denominator, 500)
        self.assertEqual(plan.to_dict()["selected_tasks"], 10)
        self.assertNotIn("selected_variants", plan.to_dict())

    def test_plus_defaults_to_every_variant_once(self):
        plan = resolve_evaluation_plan("plus", benchmark_total_tasks=2519)

        self.assertEqual(plan.task_start, 0)
        self.assertEqual(plan.task_end, 2519)
        self.assertEqual(plan.selected_task_count, 2519)
        self.assertEqual(plan.trials_per_task, 1)
        self.assertEqual(plan.denominator, 2519)
        self.assertEqual(plan.to_dict()["selected_variants"], 2519)
        self.assertEqual(plan.to_dict()["trials_per_variant"], 1)

    def test_plus_rejects_more_than_one_trial(self):
        with self.assertRaisesRegex(
            EvaluationProtocolError, "exactly one trial per variant"
        ):
            resolve_evaluation_plan("plus", benchmark_total_tasks=2519, test_num=50)

    def test_plus_task_slice_has_an_explicit_slice_denominator(self):
        plan = resolve_evaluation_plan(
            "plus", benchmark_total_tasks=2519, task_range=[10, 20]
        )

        self.assertEqual(list(plan.task_indices), list(range(10, 20)))
        self.assertEqual(plan.selected_task_count, 10)
        self.assertEqual(plan.denominator, 10)

    def test_invalid_task_slices_are_rejected(self):
        invalid_ranges = ([-1, 1], [3, 3], [5, 4], [0, 11], [0])
        for task_range in invalid_ranges:
            with self.subTest(task_range=task_range):
                with self.assertRaises(EvaluationProtocolError):
                    resolve_evaluation_plan(
                        "original",
                        benchmark_total_tasks=10,
                        task_range=task_range,
                    )

    def test_installed_benchmark_shape_cannot_cross_protocols(self):
        validate_benchmark_shape("original", 10)
        for suite, task_count in LIBERO_PLUS_SUITE_TASK_COUNTS.items():
            with self.subTest(suite=suite):
                validate_benchmark_shape("plus", task_count, suite=suite)

        with self.assertRaisesRegex(EvaluationProtocolError, "looks like LIBERO-Plus"):
            validate_benchmark_shape("original", 2519)
        with self.assertRaisesRegex(EvaluationProtocolError, "exactly 2519"):
            validate_benchmark_shape("plus", 11, suite="libero_10")
        with self.assertRaisesRegex(EvaluationProtocolError, "suite name"):
            validate_benchmark_shape("plus", 2519)


class ClassificationTest(unittest.TestCase):
    def _write_classification(self, directory: str, tasks: list[dict]) -> Path:
        path = Path(directory) / "task_classification.json"
        path.write_text(
            json.dumps({"libero_10": tasks}),
            encoding="utf-8",
        )
        return path

    def test_classification_is_loaded_with_content_provenance(self):
        tasks = [
            {
                "id": 1,
                "name": "variant_a",
                "category": "Camera Viewpoints",
                "difficulty_level": 2,
            },
            {
                "id": 2,
                "name": "variant_b",
                "category": "Light Conditions",
                "difficulty_level": None,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_classification(directory, tasks)
            bundle = load_plus_task_metadata(
                path, "libero_10", ["variant_a", "variant_b"]
            )

        self.assertEqual(len(bundle.tasks), 2)
        self.assertEqual(bundle.tasks[0].category, "Camera Viewpoints")
        self.assertIsNone(bundle.tasks[1].difficulty_level)
        self.assertEqual(len(bundle.sha256), 64)
        self.assertEqual(len(bundle.semantic_sha256), 64)

    def test_noncanonical_plus_metadata_fails_closed(self):
        tasks = [
            {
                "id": 1,
                "name": "variant_a",
                "category": "Camera Viewpoints",
                "difficulty_level": 2,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_classification(directory, tasks)
            with self.assertRaisesRegex(
                EvaluationProtocolError, "pinned official classification"
            ):
                load_plus_task_metadata(
                    path,
                    "libero_10",
                    ["variant_a"],
                    expected_semantic_sha256=(
                        LIBERO_PLUS_CLASSIFICATION_SEMANTIC_SHA256
                    ),
                )

    def test_classification_order_mismatch_fails_closed(self):
        tasks = [
            {
                "id": 1,
                "name": "variant_b",
                "category": "Camera Viewpoints",
                "difficulty_level": 2,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_classification(directory, tasks)
            with self.assertRaisesRegex(EvaluationProtocolError, "order mismatch"):
                load_plus_task_metadata(path, "libero_10", ["variant_a"])

    def test_classification_size_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_classification(directory, [])
            with self.assertRaisesRegex(EvaluationProtocolError, "size mismatch"):
                load_plus_task_metadata(path, "libero_10", ["variant_a"])


class ReportingTest(unittest.TestCase):
    def test_aggregation_preserves_category_numerators_and_denominators(self):
        summary = aggregate_task_results(
            [
                {"category": "Camera", "successes": 1, "denominator": 1},
                {"category": "Camera", "successes": 0, "denominator": 1},
                {"category": "Language", "successes": 1, "denominator": 1},
            ],
            selection_unit="variant",
        )

        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["denominator"], 3)
        self.assertEqual(summary["completed_variants"], 3)
        self.assertAlmostEqual(summary["observed_success_rate"], 2 / 3)
        self.assertEqual(summary["by_category"]["Camera"]["successes"], 1)
        self.assertEqual(summary["by_category"]["Camera"]["denominator"], 2)

    def test_plus_prompt_comes_from_bddl_environment(self):
        filename_prompt = "task view 0 0 100 initstate 0"

        self.assertEqual(
            resolve_prompt("plus", filename_prompt, "Turn on the stove"),
            "Turn on the stove",
        )
        self.assertEqual(
            resolve_prompt("original", "turn on the stove", "Turn on the stove"),
            "turn on the stove",
        )
        with self.assertRaisesRegex(EvaluationProtocolError, "BDDL language"):
            resolve_prompt("plus", filename_prompt, None)


class ScoreEligibilityTest(unittest.TestCase):
    @staticmethod
    def _results(count: int) -> list[dict]:
        return [
            {"category": "Background", "successes": index % 2, "denominator": 1}
            for index in range(count)
        ]

    def test_incomplete_full_suite_is_not_a_plus_score(self):
        plan = resolve_evaluation_plan("plus", benchmark_total_tasks=2519)
        completed = aggregate_task_results(self._results(10))

        status = describe_evaluation_status(plan, completed)

        self.assertEqual(status["status"], "in_progress")
        self.assertEqual(status["scope"], "full_suite")
        self.assertFalse(status["reportable_as_full_suite_score"])
        self.assertEqual(status["score_reason"], "incomplete_run")
        self.assertAlmostEqual(status["completed_coverage_fraction"], 10 / 2519)

    def test_completed_shard_is_not_a_plus_score(self):
        plan = resolve_evaluation_plan(
            "plus", benchmark_total_tasks=2519, task_range=[0, 10]
        )
        completed = aggregate_task_results(self._results(10))

        status = describe_evaluation_status(plan, completed)

        self.assertEqual(status["status"], "complete_shard")
        self.assertEqual(status["scope"], "shard")
        self.assertTrue(status["is_plan_complete"])
        self.assertFalse(status["reportable_as_full_suite_score"])

    def test_only_completed_full_suite_is_reportable(self):
        plan = resolve_evaluation_plan("plus", benchmark_total_tasks=2)
        completed = aggregate_task_results(self._results(2))

        status = describe_evaluation_status(plan, completed)

        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["reportable_as_full_suite_score"])

    def test_original_noncanonical_trial_count_is_not_reportable(self):
        plan = resolve_evaluation_plan("original", benchmark_total_tasks=10, test_num=1)
        completed = aggregate_task_results(self._results(10), selection_unit="task")

        status = describe_evaluation_status(plan, completed)

        self.assertEqual(status["status"], "complete_noncanonical")
        self.assertFalse(status["uses_canonical_trial_count"])
        self.assertEqual(status["canonical_trials_per_entry"], 50)
        self.assertEqual(status["official_full_suite_denominator"], 500)
        self.assertFalse(status["reportable_as_full_suite_score"])
        self.assertEqual(status["score_reason"], "noncanonical_trial_count")

    def test_original_ten_by_fifty_is_reportable(self):
        plan = resolve_evaluation_plan("original", benchmark_total_tasks=10)
        completed = aggregate_task_results(
            [
                {
                    "category": "Original LIBERO",
                    "successes": 25,
                    "denominator": 50,
                }
                for _ in range(10)
            ],
            selection_unit="task",
        )

        status = describe_evaluation_status(plan, completed)

        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["uses_canonical_trial_count"])
        self.assertEqual(status["official_full_suite_denominator"], 500)
        self.assertTrue(status["reportable_as_full_suite_score"])


class ResultLifecycleTest(unittest.TestCase):
    def test_manifest_is_immutable_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = {
                "schema_version": 1,
                "created_at_utc": "first",
                "run_fingerprint": "abc",
                "run_identity": {"protocol": "plus"},
            }
            first = load_or_create_manifest(manifest, path)
            resumed = load_or_create_manifest(
                {**manifest, "created_at_utc": "second"}, path
            )

            self.assertEqual(first, resumed)
            self.assertEqual(resumed["created_at_utc"], "first")
            with self.assertRaisesRegex(EvaluationProtocolError, "refusing to mix"):
                load_or_create_manifest(
                    {
                        **manifest,
                        "created_at_utc": "third",
                        "run_fingerprint": "different",
                    },
                    path,
                )

    def test_matching_task_result_resumes_and_other_run_is_ignored(self):
        expected_fields = {
            "protocol": "plus",
            "suite": "libero_10",
            "task_index": 0,
            "task_id": 1,
            "task_name": "variant_a",
            "category": "Background Textures",
            "difficulty_level": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            result = {
                **expected_fields,
                "run_fingerprint": "run-a",
                "prompt": "Turn on the stove",
                "successes": 1,
                "denominator": 1,
                "success_rate": 1.0,
                "provenance": {
                    "manifest": "plus/runs/run-a/manifest.json",
                    "manifest_sha256": "manifest-hash",
                },
            }
            write_json_atomic(result, path)

            resumed = load_resumable_task_result(
                path,
                run_fingerprint="run-a",
                manifest_reference="plus/runs/run-a/manifest.json",
                manifest_sha256="manifest-hash",
                expected_task_fields=expected_fields,
                trials_per_task=1,
            )
            ignored = load_resumable_task_result(
                path,
                run_fingerprint="run-b",
                manifest_reference="plus/runs/run-b/manifest.json",
                manifest_sha256="other-hash",
                expected_task_fields=expected_fields,
                trials_per_task=1,
            )

            self.assertEqual(resumed, result)
            self.assertIsNone(ignored)

    def test_same_run_with_wrong_manifest_fails_closed(self):
        expected_fields = {
            "protocol": "plus",
            "suite": "libero_10",
            "task_index": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            write_json_atomic(
                {
                    **expected_fields,
                    "run_fingerprint": "run-a",
                    "prompt": "Turn on the stove",
                    "successes": 0,
                    "denominator": 1,
                    "success_rate": 0.0,
                    "provenance": {
                        "manifest": "manifest.json",
                        "manifest_sha256": "stale",
                    },
                },
                path,
            )

            with self.assertRaisesRegex(EvaluationProtocolError, "immutable manifest"):
                load_resumable_task_result(
                    path,
                    run_fingerprint="run-a",
                    manifest_reference="manifest.json",
                    manifest_sha256="current",
                    expected_task_fields=expected_fields,
                    trials_per_task=1,
                )

    def test_atomic_writer_leaves_no_fixed_or_unique_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json_atomic({"value": 1}, path)
            write_json_atomic({"value": 2}, path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
            self.assertEqual(len(sha256_file(path)), 64)


class CompatibilityTest(unittest.TestCase):
    def test_old_run_positional_arguments_stay_in_the_same_order(self):
        client_path = REPO_ROOT / "evaluation" / "libero" / "client.py"
        tree = ast.parse(client_path.read_text(encoding="utf-8"))
        run_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )

        self.assertEqual(
            [argument.arg for argument in run_function.args.args],
            ["libero_benchmark", "port", "out_dir", "test_num", "task_range"],
        )
        self.assertEqual(
            [argument.arg for argument in run_function.args.kwonlyargs],
            ["protocol", "task_classification", "checkpoint_id"],
        )

    def test_plus_launcher_forwards_cli_overrides(self):
        launcher = (
            REPO_ROOT / "evaluation" / "libero" / "launch_client_plus.sh"
        ).read_text(encoding="utf-8")
        original_launcher = (
            REPO_ROOT / "evaluation" / "libero" / "launch_client.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('"$@"', launcher)
        self.assertIn('"$@"', original_launcher)


if __name__ == "__main__":
    unittest.main()
