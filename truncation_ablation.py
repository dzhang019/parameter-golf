from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import io
import math
import os
import zlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply post-hoc low-rank truncation to a checkpoint and measure BPB degradation."
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"),
        help="Path to a compatible train_gpt.py containing GPT/eval helpers.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a raw .pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use for eval and SVD. Default: cuda.",
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
        "--val-batch-size",
        type=int,
        default=524_288,
        help="Validation batch tokens. Default: 524288.",
    )
    parser.add_argument(
        "--train-seq-len",
        type=int,
        default=1024,
        help="Sequence length to use for validation. Default: 1024.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Target truncation rank.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help=(
            "State-dict glob patterns to truncate, e.g. "
            "'blocks.*.attn.c_q.weight' or 'blocks.[3-7].attn.c_q.weight' after shell expansion is disabled."
        ),
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help=(
            "Ranked truncation group in the form "
            "\"RANK:pattern1,pattern2,...\". Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=(),
        help="Optional state-dict glob patterns to skip after include matching.",
    )
    parser.add_argument(
        "--quantized-roundtrip",
        action="store_true",
        help="Also quantize/dequantize the truncated model and evaluate the roundtrip BPB.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print the tensors that would be truncated, without evaluating.",
    )
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_train_gpt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def truncated_svd_matrix(weight: torch.Tensor, rank: int, device: torch.device) -> torch.Tensor:
    matrix = weight.detach().to(device=device, dtype=torch.float32)
    max_rank = min(matrix.shape)
    r = min(rank, max_rank)
    if r <= 0 or r >= max_rank:
        return weight.detach().clone()
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    approx = (u[:, :r] * s[:r]) @ vh[:r, :]
    return approx.to(dtype=weight.dtype, device=weight.device).contiguous()


def build_args(module, cli: argparse.Namespace):
    args = module.Hyperparameters()
    args.data_path = cli.data_path
    args.train_files = os.path.join(cli.data_path, "fineweb_train_*.bin")
    args.val_files = os.path.join(cli.data_path, "fineweb_val_*.bin")
    args.tokenizer_path = cli.tokenizer_path
    args.vocab_size = cli.vocab_size
    args.val_batch_size = cli.val_batch_size
    args.train_seq_len = cli.train_seq_len
    return args


def parse_groups(cli: argparse.Namespace) -> list[tuple[int, tuple[str, ...]]]:
    groups: list[tuple[int, tuple[str, ...]]] = []
    if cli.group:
        for item in cli.group:
            try:
                rank_text, pattern_text = item.split(":", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid --group value {item!r}. Expected RANK:pattern1,pattern2,...") from exc
            rank = int(rank_text)
            patterns = tuple(pattern.strip() for pattern in pattern_text.split(",") if pattern.strip())
            if not patterns:
                raise ValueError(f"Group {item!r} did not include any patterns.")
            groups.append((rank, patterns))
        return groups

    if cli.rank is None or not cli.include:
        raise ValueError("Pass either --group ... (one or more times), or --rank with --include.")
    groups.append((cli.rank, tuple(cli.include)))
    return groups


def main() -> None:
    cli = parse_args()
    module = load_module(cli.train_script)
    exclude = tuple(cli.exclude)
    groups = parse_groups(cli)
    state_dict = torch.load(cli.checkpoint, map_location="cpu")
    selected_ranks: dict[str, int] = {}
    for rank_value, include_patterns in groups:
        matched = [
            name
            for name, tensor in state_dict.items()
            if isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and matches_any(name, include_patterns)
            and not matches_any(name, exclude)
        ]
        if not matched:
            raise ValueError(f"No tensors matched group rank={rank_value} include={include_patterns}")
        for name in matched:
            if name in selected_ranks and selected_ranks[name] != rank_value:
                raise ValueError(f"Tensor {name} matched multiple groups with different ranks.")
            selected_ranks[name] = rank_value

    if not selected_ranks:
        raise ValueError("No tensors matched the provided include/exclude patterns.")

    print("Matched tensors:")
    for name in sorted(selected_ranks):
        print(f"  rank={selected_ranks[name]} {name} {tuple(state_dict[name].shape)}")
    if cli.list_only:
        return

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    truncated_state = {name: tensor.clone() for name, tensor in state_dict.items()}
    for name, rank_value in selected_ranks.items():
        truncated_state[name] = truncated_svd_matrix(state_dict[name], rank_value, device)

    args = build_args(module, cli)
    world_size = 1
    grad_accum_steps = 8 // world_size
    rank = 0

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Expected SentencePiece .model tokenizer: {args.tokenizer_path}")
    sp = module.spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")

    val_tokens = module.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = module.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    base_model = module.GPT(
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
    for submodule in base_model.modules():
        if isinstance(submodule, module.CastedLinear):
            submodule.float()
    module.restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    base_model.load_state_dict(truncated_state, strict=True)
    val_loss, val_bpb = module.eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    print(f"\ntruncated_prequant val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")

    if cli.quantized_roundtrip:
        quant_obj, quant_stats = module.quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")
        base_model.load_state_dict(module.dequantize_state_dict_int8(quant_state), strict=True)
        q_val_loss, q_val_bpb = module.eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        print(
            "truncated_quantized_roundtrip "
            f"val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f} "
            f"artifact_bytes:{len(quant_blob)} payload_bytes:{quant_stats['int8_payload_bytes']}"
        )


if __name__ == "__main__":
    main()
