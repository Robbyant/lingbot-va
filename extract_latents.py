"""
Extract VAE latents from LIBERO lerobot dataset for LingBot-VA training.

Usage:
    python extract_latents.py \
        --dataset_path /hpc2hdd/home/mpeng885/Datasets_raw/libero/libero_10_no_noops_1.0.0_lerobot \
        --model_path /hpc2hdd/home/mpeng885/chenyizi/lingbot-va-main/model \
        --obs_cam_keys observation.images.image observation.images.wrist_image \
        --target_fps 5 --height 128 --width 128 --device cuda:0
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from einops import rearrange
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel


# ---------------------------------------------------------------------------
# Video reading
# ---------------------------------------------------------------------------

def read_video_frames_av(video_path: Path, frame_ids: list, gpu_id: int = 0) -> list:
    """Read specific frame indices from a video file using PyAV.
    Tries NVDEC hardware decoding first, falls back to CPU on failure."""
    frame_set = set(frame_ids)
    max_id = max(frame_ids)
    buf = {}

    def _decode(container):
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = 'DEFAULT'
        for cur_idx, frame in enumerate(container.decode(stream)):
            if cur_idx in frame_set:
                buf[cur_idx] = frame.to_ndarray(format='rgb24')
            if cur_idx > max_id:
                break

    try:
        with av.open(str(video_path), options={
            'hwaccel': 'cuda',
            'hwaccel_device': str(gpu_id),
        }) as container:
            _decode(container)
    except Exception:
        buf.clear()
        with av.open(str(video_path)) as container:
            _decode(container)

    return [buf[fid] for fid in frame_ids if fid in buf]


def sample_frame_ids(start: int, end: int, ori_fps: int, target_fps: int) -> list:
    """
    Sample frame indices at target_fps from [start, end).
    Trims to largest (4n+1) count for VAE temporal alignment.
    """
    stride = max(1, round(ori_fps / target_fps))
    ids = list(range(start, end, stride))
    n = (len(ids) - 1) // 4
    ids = ids[:4 * n + 1]
    return ids if len(ids) >= 1 else [start]


def frames_to_tensor(frames: list, height: int, width: int) -> torch.Tensor:
    """Convert list of HxWxC uint8 frames to [3, F, H, W] float32 in [-1, 1]."""
    resized = []
    for f in frames:
        t = torch.from_numpy(f).float()
        t = t.permute(2, 0, 1).unsqueeze(0)                          # 1 C H W
        t = F.interpolate(t, size=(height, width), mode='bilinear', align_corners=False)
        resized.append(t.squeeze(0))                                   # C H W
    tensor = torch.stack(resized, dim=1)                               # C F H W
    return tensor / 127.5 - 1.0


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_video(vae: AutoencoderKLWan,
                 video_tensor: torch.Tensor,
                 device: str):
    """
    Encode [3, F, H, W] tensor -> (latent_flat [F'*H'*W', 48], F', H', W').
    Latent is normalized (z-scored) and stored as bfloat16.
    Uses vae.encode() which handles full sequences correctly.
    """
    x = video_tensor.unsqueeze(0).to(device).to(torch.bfloat16)   # 1 3 F H W
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
    mu = posterior.mean                                             # 1 48 F' H' W'

    mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=torch.float32)
    std  = torch.tensor(vae.config.latents_std,  device=mu.device, dtype=torch.float32)
    mu_norm = ((mu.float() - mean.view(1, -1, 1, 1, 1)) /
               std.view(1, -1, 1, 1, 1)).to(torch.bfloat16)

    mu_norm = mu_norm.squeeze(0).cpu()                             # 48 F' H' W'
    _, Fl, Hl, Wl = mu_norm.shape
    latent_flat = rearrange(mu_norm, 'c f h w -> (f h w) c')
    return latent_flat, Fl, Hl, Wl


def encode_text(text_encoder, tokenizer, text: str, device: str,  # noqa: E501
                max_length: int = 512) -> torch.Tensor:
    """Encode text -> [L, D] bfloat16 tensor."""
    inputs = tokenizer(text, return_tensors='pt', padding='max_length',
                       max_length=max_length, truncation=True).to(device)
    with torch.no_grad():
        emb = text_encoder(**inputs).last_hidden_state[0]          # L D
    return emb.cpu().to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--model_path',   required=True)
    parser.add_argument('--obs_cam_keys', nargs='+',
                        default=['observation.images.image',
                                 'observation.images.wrist_image'])
    parser.add_argument('--target_fps', type=int, default=5)
    parser.add_argument('--height',     type=int, default=128)
    parser.add_argument('--width',      type=int, default=128)
    parser.add_argument('--device',     default='cuda:0')
    parser.add_argument('--ep_start',   type=int, default=0)
    parser.add_argument('--ep_end',     type=int, default=-1)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    model_path   = Path(args.model_path)
    device       = args.device
    gpu_id       = int(device.split(':')[1]) if ':' in device else 0

    # ---- Load models --------------------------------------------------------
    print('Loading VAE...')
    vae = AutoencoderKLWan.from_pretrained(
        str(model_path / 'vae'), torch_dtype=torch.bfloat16).to(device)

    print('Loading text encoder...')
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(model_path / 'text_encoder'), torch_dtype=torch.bfloat16).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(str(model_path / 'tokenizer'))

    # ---- Empty embedding ----------------------------------------------------
    empty_emb_path = dataset_path / 'empty_emb.pt'
    if not empty_emb_path.exists():
        print('Generating empty text embedding...')
        empty_emb = encode_text(text_encoder, tokenizer, '', device)
        torch.save(empty_emb, empty_emb_path)
        print(f'Saved empty_emb.pt  shape={empty_emb.shape}')
    else:
        print('empty_emb.pt already exists.')

    # ---- Load / update episodes.jsonl ---------------------------------------
    episodes_file = dataset_path / 'meta' / 'episodes.jsonl'
    episodes = []
    with open(episodes_file) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    changed = False
    for ep in episodes:
        if 'action_config' not in ep:
            task_text = ep['tasks'][0] if ep['tasks'] else 'robot manipulation'
            ep['action_config'] = [{
                'start_frame': 0,
                'end_frame':   ep['length'],
                'action_text': task_text,
            }]
            changed = True
    if changed:
        with open(episodes_file, 'w') as f:
            for ep in episodes:
                f.write(json.dumps(ep) + '\n')
        print(f'Updated episodes.jsonl with action_config for {len(episodes)} episodes')
    else:
        print('episodes.jsonl already has action_config.')

    # ---- Dataset info -------------------------------------------------------
    with open(dataset_path / 'meta' / 'info.json') as f:
        info = json.load(f)
    ori_fps     = info.get('fps', 20)
    chunks_size = info.get('chunks_size', 1000)
    print(f'Dataset: {len(episodes)} eps, ori_fps={ori_fps}, chunks_size={chunks_size}')

    # ---- Pre-encode text prompts -------------------------------------------
    all_texts = {acfg['action_text']
                 for ep in episodes
                 for acfg in ep.get('action_config', [])}
    print(f'Pre-encoding {len(all_texts)} unique prompts...')
    text_cache = {}
    for text in tqdm(sorted(all_texts), desc='text enc'):
        text_cache[text] = encode_text(text_encoder, tokenizer, text, device)

    # ---- Extract latents ----------------------------------------------------
    print('Extracting VAE latents...')
    skipped = processed = errors = 0

    ep_end = args.ep_end if args.ep_end >= 0 else len(episodes)
    episodes = episodes[args.ep_start:ep_end]

    for ep in tqdm(episodes, desc='episodes'):
        ep_idx   = ep['episode_index']
        ep_chunk = ep_idx // chunks_size

        for acfg in ep.get('action_config', []):
            start_frame = acfg['start_frame']
            end_frame   = acfg['end_frame']
            action_text = acfg['action_text']
            text_emb    = text_cache[action_text]

            frame_ids = sample_frame_ids(start_frame, end_frame, ori_fps, args.target_fps)
            if len(frame_ids) < 5:
                print(f'  Skip ep {ep_idx}: too few frames ({len(frame_ids)})')
                continue

            # Check if all views already done
            if all(
                (dataset_path / 'latents' / f'chunk-{ep_chunk:03d}' / ck /
                 f'episode_{ep_idx:06d}_{start_frame}_{end_frame}.pth').exists()
                for ck in args.obs_cam_keys
            ):
                skipped += 1
                continue

            # Encode each camera view
            for cam_key in args.obs_cam_keys:
                out_dir  = (dataset_path / 'latents' / f'chunk-{ep_chunk:03d}' / cam_key)
                out_file = out_dir / f'episode_{ep_idx:06d}_{start_frame}_{end_frame}.pth'
                if out_file.exists():
                    continue

                video_path = (dataset_path / 'videos' / f'chunk-{ep_chunk:03d}' /
                              cam_key / f'episode_{ep_idx:06d}.mp4')
                if not video_path.exists():
                    print(f'  Warning: missing {video_path}')
                    errors += 1
                    continue

                try:
                    frames = read_video_frames_av(video_path, frame_ids, gpu_id)
                except Exception as e:
                    print(f'  Error reading ep {ep_idx} {cam_key}: {e}')
                    errors += 1
                    continue

                if len(frames) < 5:
                    print(f'  Skip ep {ep_idx} {cam_key}: decoded only {len(frames)} frames')
                    continue

                # Trim to (4n+1)
                n      = (len(frames) - 1) // 4
                frames = frames[:4 * n + 1]
                used_ids = frame_ids[:len(frames)]

                video_tensor = frames_to_tensor(frames, args.height, args.width)
                latent_flat, Fl, Hl, Wl = encode_video(vae, video_tensor, device)

                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'latent':            latent_flat,      # [F'*H'*W', 48] bfloat16
                    'latent_num_frames': Fl,
                    'latent_height':     Hl,
                    'latent_width':      Wl,
                    'video_num_frames':  len(frames),
                    'video_height':      args.height,
                    'video_width':       args.width,
                    'text_emb':          text_emb,         # [L, D] bfloat16
                    'text':              action_text,
                    'frame_ids':         used_ids,
                    'start_frame':       start_frame,
                    'end_frame':         end_frame,
                    'fps':               args.target_fps,
                    'ori_fps':           ori_fps,
                }, out_file)

            processed += 1

    print(f'\nDone!  processed={processed}  skipped={skipped}  errors={errors}')


if __name__ == '__main__':
    main()
