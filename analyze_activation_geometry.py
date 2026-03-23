from __future__ import annotations

import argparse
import importlib.util
import math
import os
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze layerwise activation geometry for a Parameter Golf checkpoint."
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"),
        help="Compatible baseline train_gpt.py containing GPT and data helpers.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to raw .pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use. Default: cuda.",
    )
    parser.add_argument(
        "--data-path",
        default="./data/datasets/fineweb10B_sp1024",
        help="Dataset directory containing fineweb_val_*.bin shards.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="./data/tokenizers/fineweb_1024_bpe.model",
        help="SentencePiece tokenizer path.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="Tokenizer vocab size. Default: 1024.",
    )
    parser.add_argument(
        "--train-seq-len",
        type=int,
        default=1024,
        help="Sequence length. Default: 1024.",
    )
    parser.add_argument(
        "--batch-size-tokens",
        type=int,
        default=524_288,
        help="Validation batch tokens. Default: 524288.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=8,
        help="Maximum number of validation batches to analyze. Default: 8.",
    )
    parser.add_argument(
        "--capture",
        nargs="+",
        default=("block_out", "attn_in", "mlp_in"),
        choices=("embed", "block_out", "attn_in", "mlp_in"),
        help="Activation families to capture.",
    )
    parser.add_argument(
        "--token-subsample",
        type=int,
        default=8192,
        help="Maximum number of token vectors to accumulate per capture site. Default: 8192.",
    )
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_train_gpt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActivationAccumulator:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.samples: list[torch.Tensor] = []
        self.total_seen = 0

    def add(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32, device="cpu")
        if flat.numel() == 0:
            return
        remaining = self.max_samples - self.total_seen
        if remaining <= 0:
            return
        if flat.shape[0] > remaining:
            stride = max(flat.shape[0] // remaining, 1)
            flat = flat[::stride][:remaining]
        self.samples.append(flat.contiguous())
        self.total_seen += flat.shape[0]

    def matrix(self) -> torch.Tensor:
        if not self.samples:
            raise ValueError("No samples accumulated.")
        return torch.cat(self.samples, dim=0)


def summarize_matrix(name: str, x: torch.Tensor) -> dict[str, float | int | str]:
    centered = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)
    e = s.square()
    total = float(e.sum().item())
    probs = e / max(total, 1e-12)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())

    def energy_rank(frac: float) -> int:
        if total <= 0.0:
            return 0
        cutoff = total * frac
        cum = torch.cumsum(e, dim=0)
        return int(torch.searchsorted(cum, cutoff).item()) + 1

    r90 = energy_rank(0.90)
    r95 = energy_rank(0.95)
    r99 = energy_rank(0.99)
    dim = x.shape[1]
    return {
        "name": name,
        "num_samples": x.shape[0],
        "dim": dim,
        "r90": r90,
        "r95": r95,
        "r99": r99,
        "r90_frac": r90 / dim,
        "r95_frac": r95 / dim,
        "r99_frac": r99 / dim,
        "effective_rank": math.exp(entropy),
        "top1_energy": float((e[0] / total).item()) if total > 0 else 0.0,
    }


def build_args(module, cli: argparse.Namespace):
    args = module.Hyperparameters()
    args.data_path = cli.data_path
    args.train_files = os.path.join(cli.data_path, "fineweb_train_*.bin")
    args.val_files = os.path.join(cli.data_path, "fineweb_val_*.bin")
    args.tokenizer_path = cli.tokenizer_path
    args.vocab_size = cli.vocab_size
    args.val_batch_size = cli.batch_size_tokens
    args.train_seq_len = cli.train_seq_len
    return args


def main() -> None:
    cli = parse_args()
    module = load_module(cli.train_script)
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)

    args = build_args(module, cli)
    state_dict = torch.load(cli.checkpoint, map_location="cpu")

    model = module.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for submodule in model.modules():
        if isinstance(submodule, module.CastedLinear):
            submodule.float()
    module.restore_low_dim_params_to_fp32(model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    val_tokens = module.load_validation_tokens(args.val_files, args.train_seq_len)
    grad_accum_steps = 8
    local_batch_tokens = args.val_batch_size // grad_accum_steps
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len

    accumulators: dict[str, ActivationAccumulator] = defaultdict(lambda: ActivationAccumulator(cli.token_subsample))
    hooks = []

    def maybe_add(name: str, tensor: torch.Tensor) -> None:
        accumulators[name].add(tensor)

    if "embed" in cli.capture:
        def embed_hook(_module, _inputs, output):
            maybe_add("embed", output)
        hooks.append(model.tok_emb.register_forward_hook(embed_hook))

    if "attn_in" in cli.capture:
        for i, block in enumerate(model.blocks):
            def make_hook(idx: int):
                def hook(_module, inputs, _output):
                    maybe_add(f"blocks.{idx}.attn_in", inputs[0])
                return hook
            hooks.append(block.attn.register_forward_hook(make_hook(i)))

    if "mlp_in" in cli.capture:
        for i, block in enumerate(model.blocks):
            def make_hook(idx: int):
                def hook(_module, inputs, _output):
                    maybe_add(f"blocks.{idx}.mlp_in", inputs[0])
                return hook
            hooks.append(block.mlp.register_forward_hook(make_hook(i)))

    if "block_out" in cli.capture:
        for i, block in enumerate(model.blocks):
            def make_hook(idx: int):
                def hook(_module, _inputs, output):
                    maybe_add(f"blocks.{idx}.block_out", output)
                return hook
            hooks.append(block.register_forward_hook(make_hook(i)))

    with torch.inference_mode():
        batches_done = 0
        for batch_seq_start in range(0, total_seqs, local_batch_seqs):
            if batches_done >= cli.max_batches:
                break
            batch_seq_end = min(batch_seq_start + local_batch_seqs, total_seqs)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _ = model(x, y)
            batches_done += 1

    for hook in hooks:
        hook.remove()

    rows = []
    for name, acc in accumulators.items():
        rows.append(summarize_matrix(name, acc.matrix()))
    rows.sort(key=lambda row: (row["name"]))

    print(f"Analyzed checkpoint: {cli.checkpoint}")
    print(f"Batches analyzed: {cli.max_batches}")
    print(
        f"{'name':26} {'samples':>8} {'dim':>6} {'r90%':>6} {'r95%':>6} {'r99%':>6} {'eff_rk':>8} {'top1E':>7}"
    )
    for row in rows:
        print(
            f"{row['name'][:26]:26} {int(row['num_samples']):8d} {int(row['dim']):6d} "
            f"{float(row['r90_frac']):6.3f} {float(row['r95_frac']):6.3f} {float(row['r99_frac']):6.3f} "
            f"{float(row['effective_rank']):8.1f} {float(row['top1_energy']):7.3f}"
        )


if __name__ == "__main__":
    main()
