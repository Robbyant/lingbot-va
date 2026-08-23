import argparse
import importlib.metadata
import secrets
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
    WebsocketClientPolicy,
)

from eval_protocol import (
    ClassificationBundle,
    EvaluationProtocolError,
    LIBERO_PLUS_CLASSIFICATION_SEMANTIC_SHA256,
    TaskMetadata,
    add_protocol_arguments,
    aggregate_task_results,
    canonical_json_sha256,
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


def save_video(
    real_obs_list,
    save_path,
    fps=15,
    video_names=(
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    ),
):
    if not real_obs_list:
        print("No real observation frames; skipping video")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)

    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(
            np.uint8
        )
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as exc:
            print(f"Error: constructing environment failed: {exc}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """Extract and vertically flip the two uint8 camera observations."""

    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {
        "observation.images.agentview_rgb": agentview,
        "observation.images.eye_in_hand_rgb": eye_in_hand,
    }


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.0] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(
    model,
    benchmark_instance,
    protocol,
    task_idx,
    video_root,
    episode_idx,
):
    num_tasks = benchmark_instance.get_num_tasks()
    if task_idx >= num_tasks:
        raise EvaluationProtocolError(
            f"Task index {task_idx} must be smaller than {num_tasks}."
        )

    task = benchmark_instance.get_task(task_idx)
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    if cur_env is None:
        raise RuntimeError(
            f"Could not construct environment for task index {task_idx} after 5 tries."
        )

    try:
        prompt = resolve_prompt(
            protocol,
            benchmark_prompt=task.language,
            environment_prompt=getattr(cur_env, "language_instruction", None),
        )
        first_obs = init_single_env(
            cur_env, init_states[episode_idx % init_states.shape[0]]
        )

        model.infer(dict(reset=True, prompt=prompt))

        full_obs_list = []
        done = False
        first = True
        while cur_env.env.timestep < 800:
            ret = model.infer(dict(obs=first_obs, prompt=prompt))
            action = ret["action"]

            key_frame_list = []
            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4
            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    ee_action = action[:, i, j]
                    observes, done = env_one_step(cur_env, ee_action)
                    if done:
                        break
                    if (j + 1) % action_per_frame == 0:
                        full_obs_list.append(observes)
                        key_frame_list.append(observes)

                if done:
                    break

            first = False

            if done:
                break
            model.infer(
                dict(
                    obs=key_frame_list,
                    compute_kv_cache=True,
                    imagine=False,
                    state=action,
                )
            )
    finally:
        cur_env.close()

    artifact_name = task.name if protocol == "plus" else prompt.replace(" ", "_")
    out_file = (
        Path(video_root)
        / f"{task_idx}_{artifact_name}"
        / f"{episode_idx}_{bool(done)}.mp4"
    )
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=(
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ),
    )

    return bool(done), prompt


def _git_revision(path):
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _installed_versions():
    versions = {}
    for distribution in ("libero", "robosuite", "mujoco", "numpy", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _build_provenance(classification, checkpoint_id):
    client_path = Path(__file__).resolve()
    repo_root = client_path.parents[2]
    benchmark_path = Path(benchmark.__file__).resolve()
    protocol_path = client_path.with_name("eval_protocol.py")

    classification_provenance = None
    if classification is not None:
        classification_provenance = {
            "path": classification.source,
            "sha256": classification.sha256,
            "semantic_sha256": classification.semantic_sha256,
        }

    return {
        "lingbot_va_git_commit": _git_revision(repo_root),
        "client_path": str(client_path),
        "client_sha256": sha256_file(client_path),
        "protocol_sha256": sha256_file(protocol_path),
        "libero_benchmark_module": str(benchmark_path),
        "libero_benchmark_module_sha256": sha256_file(benchmark_path),
        "libero_benchmark_git_commit": _git_revision(benchmark_path.parent),
        "task_classification": classification_provenance,
        "checkpoint_id_declared_by_user": checkpoint_id,
        "package_versions": _installed_versions(),
    }


def _planned_category_counts(plan, task_metadata):
    counts = Counter(task_metadata[task_idx].category for task_idx in plan.task_indices)
    return {
        category: {
            f"{plan.selection_unit}s": count,
            "denominator": count * plan.trials_per_task,
        }
        for category, count in sorted(counts.items())
    }


def _result_path(out_dir, protocol, suite, task_idx, run_root=None):
    out_dir = Path(out_dir)
    if protocol == "original":
        # Preserve the existing result path for downstream original-LIBERO users.
        return out_dir / f"{suite}_{task_idx}.json"
    if run_root is None:
        raise ValueError("run_root is required for collision-free LIBERO-Plus output.")
    return Path(run_root) / "tasks" / f"{suite}_{task_idx}.json"


def run(
    libero_benchmark,
    port,
    out_dir,
    test_num=None,
    task_range=None,
    *,
    protocol="original",
    task_classification=None,
    checkpoint_id=None,
):
    """Run one explicitly selected LIBERO evaluation protocol."""

    checkpoint_id = normalize_checkpoint_id(protocol, checkpoint_id)

    # Preserve the original CLI's optional checkpoint metadata while preventing
    # two unidentified original-LIBERO invocations from sharing a resume key.
    unidentified_session_nonce = (
        secrets.token_hex(16) if checkpoint_id is None else None
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    benchmark_total_tasks = benchmark_instance.get_num_tasks()

    validate_benchmark_shape(
        protocol,
        benchmark_total_tasks,
        suite=libero_benchmark,
    )
    plan = resolve_evaluation_plan(
        protocol=protocol,
        benchmark_total_tasks=benchmark_total_tasks,
        task_range=task_range,
        test_num=test_num,
    )

    task_names = [
        benchmark_instance.get_task(task_idx).name
        for task_idx in range(benchmark_total_tasks)
    ]
    classification: ClassificationBundle | None = None
    if protocol == "plus":
        classification_path = (
            Path(task_classification)
            if task_classification is not None
            else Path(benchmark.__file__)
            .resolve()
            .with_name("task_classification.json")
        )
        classification = load_plus_task_metadata(
            classification_path,
            suite=libero_benchmark,
            task_names=task_names,
            expected_semantic_sha256=(LIBERO_PLUS_CLASSIFICATION_SEMANTIC_SHA256),
        )
        task_metadata = classification.tasks
    else:
        task_metadata = tuple(
            TaskMetadata(
                task_id=task_idx + 1,
                name=task_name,
                category="Original LIBERO",
                difficulty_level=None,
            )
            for task_idx, task_name in enumerate(task_names)
        )

    output_root = Path(out_dir)
    output_root.mkdir(exist_ok=True, parents=True)

    provenance = _build_provenance(classification, checkpoint_id)
    classification_semantic_sha256 = (
        classification.semantic_sha256 if classification is not None else None
    )
    run_identity = {
        "schema_version": 1,
        "benchmark_suite": libero_benchmark,
        "evaluation_plan": plan.to_dict(),
        "checkpoint_id_declared_by_user": checkpoint_id,
        "unidentified_session_nonce": unidentified_session_nonce,
        "code": {
            "lingbot_va_git_commit": provenance["lingbot_va_git_commit"],
            "client_sha256": provenance["client_sha256"],
            "protocol_sha256": provenance["protocol_sha256"],
        },
        "benchmark": {
            "git_commit": provenance["libero_benchmark_git_commit"],
            "module_sha256": provenance["libero_benchmark_module_sha256"],
            "task_order_sha256": canonical_json_sha256(task_names),
            "task_classification_semantic_sha256": (classification_semantic_sha256),
            "package_versions": provenance["package_versions"],
        },
    }
    run_fingerprint = canonical_json_sha256(run_identity)
    run_stem = (
        f"{libero_benchmark}_{protocol}_{plan.task_start}-{plan.task_end}_"
        f"{run_fingerprint[:12]}"
    )
    if protocol == "plus":
        run_root = output_root / "plus" / "runs" / run_stem
        video_root = run_root / "videos" / libero_benchmark
    else:
        run_root = output_root / "runs" / run_stem
        # Preserve the existing original-LIBERO video location.
        video_root = output_root / libero_benchmark

    manifest_path = run_root / "manifest.json"
    summary_path = run_root / "summary.json"
    manifest_reference = manifest_path.relative_to(output_root).as_posix()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_suite": libero_benchmark,
        "evaluation_plan": plan.to_dict(),
        "planned_by_category": _planned_category_counts(plan, task_metadata),
        "run_fingerprint": run_fingerprint,
        "run_identity": run_identity,
        "prompt_source": (
            "benchmark task language (original behavior)"
            if protocol == "original"
            else "BDDL :language via OffScreenRenderEnv.language_instruction"
        ),
        "provenance": provenance,
    }
    manifest = load_or_create_manifest(manifest, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)

    print(
        "Evaluation plan: "
        f"protocol={protocol}, suite={libero_benchmark}, "
        f"selected_{plan.selection_unit}s={plan.selected_task_count}, "
        f"trials_per_{plan.selection_unit}={plan.trials_per_task}, "
        f"denominator={plan.denominator}"
    )
    print(f"Run manifest: {manifest_path}")

    task_results = []
    model = None

    def write_summary():
        completed = aggregate_task_results(
            task_results,
            selection_unit=plan.selection_unit,
        )
        evaluation_status = describe_evaluation_status(plan, completed)
        observed_success_rate = completed["observed_success_rate"]
        full_suite_success_rate = (
            observed_success_rate
            if evaluation_status["reportable_as_full_suite_score"]
            else None
        )
        summary = {
            "schema_version": 1,
            "benchmark_suite": libero_benchmark,
            "protocol": protocol,
            "run_fingerprint": run_fingerprint,
            "planned": plan.to_dict(),
            "completed": completed,
            "evaluation_status": evaluation_status,
            "score": {
                "metric": "micro_success_rate",
                "value": full_suite_success_rate,
                "reportable": evaluation_status["reportable_as_full_suite_score"],
                "reason": evaluation_status["score_reason"],
            },
            "planned_by_category": manifest["planned_by_category"],
            "provenance": {
                "manifest": manifest_reference,
                "manifest_sha256": manifest_sha256,
                "task_classification_sha256": (
                    classification.sha256 if classification is not None else None
                ),
                "task_classification_semantic_sha256": (classification_semantic_sha256),
            },
        }
        write_json_atomic(summary, summary_path)
        return summary

    summary = write_summary()
    progress_bar = tqdm(
        plan.task_indices,
        total=plan.selected_task_count,
    )

    for task_idx in progress_bar:
        metadata = task_metadata[task_idx]
        result_path = _result_path(
            out_dir=output_root,
            protocol=protocol,
            suite=libero_benchmark,
            task_idx=task_idx,
            run_root=run_root,
        )
        expected_task_fields = {
            "protocol": protocol,
            "suite": libero_benchmark,
            "task_index": task_idx,
            "task_id": metadata.task_id,
            "task_name": metadata.name,
            "category": metadata.category,
            "difficulty_level": metadata.difficulty_level,
        }
        task_result = load_resumable_task_result(
            result_path,
            run_fingerprint=run_fingerprint,
            manifest_reference=manifest_reference,
            manifest_sha256=manifest_sha256,
            expected_task_fields=expected_task_fields,
            trials_per_task=plan.trials_per_task,
        )
        if task_result is None:
            successes = 0
            completed_trials = 0
        else:
            successes = task_result["successes"]
            completed_trials = task_result["denominator"]
            print(
                f"Resuming task {task_idx} from "
                f"{completed_trials}/{plan.trials_per_task} completed trials."
            )

        for episode_idx in tqdm(
            range(completed_trials, plan.trials_per_task),
            total=plan.trials_per_task - completed_trials,
        ):
            if model is None:
                model = WebsocketClientPolicy(port=port)
            succeeded, prompt = run_one(
                model=model,
                benchmark_instance=benchmark_instance,
                protocol=protocol,
                task_idx=task_idx,
                video_root=video_root,
                episode_idx=episode_idx,
            )
            successes += int(succeeded)
            completed_trials = episode_idx + 1
            success_rate = successes / completed_trials
            task_result = {
                # Legacy keys remain for original-LIBERO result consumers.
                "succ_num": float(successes),
                "total_num": float(completed_trials),
                "succ_rate": success_rate,
                # Protocol-explicit result schema.
                "protocol": protocol,
                "suite": libero_benchmark,
                "task_index": task_idx,
                "task_id": metadata.task_id,
                "task_name": metadata.name,
                "category": metadata.category,
                "difficulty_level": metadata.difficulty_level,
                "prompt": prompt,
                "successes": successes,
                "denominator": completed_trials,
                "success_rate": success_rate,
                "run_fingerprint": run_fingerprint,
                "provenance": {
                    "manifest": manifest_reference,
                    "manifest_sha256": manifest_sha256,
                    "task_classification_sha256": (
                        classification.sha256 if classification is not None else None
                    ),
                    "task_classification_semantic_sha256": (
                        classification_semantic_sha256
                    ),
                },
            }
            write_json_atomic(task_result, result_path)
            print(
                f"Task {task_idx} [{metadata.category}]: "
                f"{successes}/{completed_trials} "
                f"({success_rate:.4f})"
            )

        if task_result is None:
            raise AssertionError("Validated evaluation plan produced no trials.")
        task_results.append(task_result)
        summary = write_summary()
        completed = summary["completed"]
        observed_rate = completed["observed_success_rate"]
        print(
            "Observed completed entries: "
            f"{completed['successes']}/{completed['denominator']} "
            f"({observed_rate:.4f}); planned denominator={plan.denominator}; "
            f"score reportable={summary['score']['reportable']}"
        )

    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark suite name",
    )
    add_protocol_arguments(parser)
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for videos, manifests, and results",
    )
    parser.add_argument(
        "--task-classification",
        type=str,
        default=None,
        help=(
            "Optional LIBERO-Plus task_classification.json override. By default, "
            "the file next to the installed benchmark module is used."
        ),
    )
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help=(
            "Checkpoint name/revision recorded as user-declared provenance. "
            "Required for LIBERO-Plus; the separate inference server cannot "
            "verify it automatically."
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(**vars(args))
    except EvaluationProtocolError as exc:
        parser.error(str(exc))
    print("Finished all evaluation tasks.")


if __name__ == "__main__":
    main()
