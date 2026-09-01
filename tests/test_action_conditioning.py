import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "wan_va"))

# The model imports FlashAttention eagerly even though these CPU tests only
# exercise the server's action-conditioning helpers.
if importlib.util.find_spec("flash_attn_interface") is None:
    flash_attn_interface = types.ModuleType("flash_attn_interface")
    flash_attn_interface.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn_interface", loader=None
    )
    flash_attn_interface.flash_attn_func = lambda query, key, value: query
    sys.modules["flash_attn_interface"] = flash_attn_interface

from wan_va_server import VA_Server


class _TransformerStub:
    def __init__(self):
        self.cache_batch_size = None

    def clear_cache(self, cache_name):
        pass

    def create_empty_cache(self, *args, **kwargs):
        self.cache_batch_size = kwargs["batch_size"]


class _StreamingVAEStub:
    def clear_cache(self):
        pass


def _make_server(tmp_path, video_guidance, action_guidance):
    server = VA_Server.__new__(VA_Server)
    server.cache_name = "test"
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.save_root = str(tmp_path)
    server.env_type = "none"
    server.streaming_vae = _StreamingVAEStub()
    server.streaming_vae_half = None
    server.transformer = _TransformerStub()
    server.job_config = SimpleNamespace(
        guidance_scale=video_guidance,
        action_guidance_scale=action_guidance,
        action_per_frame=2,
        height=32,
        width=32,
        obs_cam_keys=["camera"],
        patch_size=(1, 2, 2),
        frame_chunk_size=2,
        attn_window=4,
        action_dim=3,
        used_action_channel_ids=[0, 1],
        norm_stat={"q01": [-1.0, 2.0, 0.0], "q99": [1.0, 4.0, 0.0]},
        action_norm_method="quantiles",
    )

    calls = []

    def encode_prompt(**kwargs):
        calls.append(kwargs)
        positive = torch.ones(1, 2, 3)
        negative = (
            torch.zeros(1, 2, 3) if kwargs["do_classifier_free_guidance"] else None
        )
        return positive, negative

    server.encode_prompt = encode_prompt
    return server, calls


@pytest.mark.parametrize(
    ("video_guidance", "action_guidance", "uses_cfg"),
    [(1.0, 1.0, False), (5.0, 1.0, True), (1.0, 2.0, True), (5.0, 2.0, True)],
)
def test_reset_builds_unconditional_prompt_for_every_cfg_mode(
    tmp_path, video_guidance, action_guidance, uses_cfg
):
    server, calls = _make_server(tmp_path, video_guidance, action_guidance)

    server._reset(prompt="move the gripper")

    assert server.use_cfg is uses_cfg
    assert calls[0]["do_classifier_free_guidance"] is uses_cfg
    assert calls[0]["negative_prompt"] is None
    assert (server.negative_prompt_embeds is not None) is uses_cfg
    assert server.transformer.cache_batch_size == (2 if uses_cfg else 1)

    inputs = {
        "noisy_latents": torch.zeros(1, 3, 1, 1, 1),
        "text_emb": server.prompt_embeds.clone(),
        "grid_id": torch.zeros(3, 1),
        "timesteps": torch.zeros(1),
    }
    repeated = server._repeat_input_for_cfg(inputs)
    expected_batch = 2 if uses_cfg else 1
    assert repeated["noisy_latents"].shape[0] == expected_batch
    assert repeated["grid_id"].shape[0] == expected_batch
    assert repeated["timesteps"].shape[0] == expected_batch
    assert repeated["text_emb"].shape[0] == expected_batch
    if uses_cfg:
        torch.testing.assert_close(repeated["text_emb"][0], server.prompt_embeds[0])
        torch.testing.assert_close(
            repeated["text_emb"][1], server.negative_prompt_embeds[0]
        )


def test_encode_prompt_uses_empty_text_as_default_unconditional_branch():
    server = VA_Server.__new__(VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    requested_prompts = []

    def get_prompt_embeds(prompt, **kwargs):
        requested_prompts.append(prompt)
        return torch.zeros(len(prompt), 2, 3)

    server._get_t5_prompt_embeds = get_prompt_embeds
    _, negative = server.encode_prompt(
        prompt="move the gripper",
        negative_prompt=None,
        do_classifier_free_guidance=True,
    )

    assert requested_prompts == [["move the gripper"], [""]]
    assert negative is not None


def test_preprocess_action_matches_training_quantile_clamp():
    server = VA_Server.__new__(VA_Server)
    server.job_config = SimpleNamespace(inverse_used_action_channel_ids=[0, 1, 2])
    server.action_norm_method = "quantiles"
    server.actions_q01 = torch.tensor([-1.0, 2.0, 0.0]).reshape(-1, 1, 1)
    server.actions_q99 = torch.tensor([1.0, 4.0, 0.0]).reshape(-1, 1, 1)
    action = np.array([[[-3.0]], [[6.0]]], dtype=np.float32)

    actual = server.preprocess_action(action)[0, ..., 0]

    padded = np.pad(action, ((0, 1), (0, 0), (0, 0)))
    q01 = server.actions_q01.numpy()
    q99 = server.actions_q99.numpy()
    expected = np.clip((padded - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0, -1.5, 1.5)
    torch.testing.assert_close(actual, torch.from_numpy(expected))
    assert actual[0].item() == -1.5
    assert actual[1].item() == 1.5
