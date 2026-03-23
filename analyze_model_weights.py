from __future__ import annotations

import argparse
import io
import json
import math
import re
import zlib
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze weight spectra for a Parameter Golf checkpoint or compressed "
            "submission artifact."
        )
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Record directory containing final_model.pt and/or final_model.int8.ptz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a raw .pt checkpoint or compressed .ptz/.zst artifact.",
    )
    parser.add_argument(
        "--prefer",
        choices=("raw", "compressed"),
        default="raw",
        help="When --record-dir is used, prefer the raw or compressed artifact.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for linear algebra. Default: cpu.",
    )
    parser.add_argument(
        "--rank-fracs",
        type=float,
        nargs="+",
        default=(0.90, 0.95, 0.99),
        help="Energy fractions for cumulative spectral rank cutoffs.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to save the full analysis as JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many matrices to print in each compact summary table.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.record_dir is None:
        raise ValueError("Pass either --checkpoint or --record-dir.")
    raw_path = args.record_dir / "final_model.pt"
    compressed_candidates = [
        args.record_dir / "final_model.int8.ptz",
        args.record_dir / "final_model.ptz",
        args.record_dir / "final_model.zst",
        args.record_dir / "final_model.zstd",
    ]
    if args.prefer == "raw" and raw_path.exists():
        return raw_path
    for candidate in compressed_candidates:
        if candidate.exists():
            return candidate
    if raw_path.exists():
        return raw_path
    raise FileNotFoundError(f"No supported checkpoint found under {args.record_dir}")


def torch_load_bytes(blob: bytes) -> Any:
    return torch.load(io.BytesIO(blob), map_location="cpu")


def maybe_decompress(blob: bytes, suffix: str) -> bytes:
    if suffix == ".ptz":
        return zlib.decompress(blob)
    if suffix in {".zst", ".zstd"}:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "zstandard is required to read .zst/.zstd artifacts. "
                "Install it in the active environment or analyze the raw checkpoint."
            ) from exc
        return zstd.ZstdDecompressor().decompress(blob)
    return blob


def dequantize_clean_per_row(obj: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        dtype_name = obj["dtypes"][name]
        dtype = getattr(torch, dtype_name)
        scales = obj["scales"][name]
        is_per_row = qmeta.get(name, {}).get("scheme") == "per_row" or scales.ndim > 0
        if is_per_row:
            scales = scales.to(dtype=torch.float32)
            view_shape = (q.shape[0],) + (1,) * (q.ndim - 1)
            out[name] = (q.float() * scales.view(*view_shape)).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(scales.item())).to(dtype=dtype).contiguous()

    for name, tensor in obj["passthrough"].items():
        restored = tensor.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            restored = restored.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = restored
    return out


def load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    suffix = path.suffix.lower()
    meta: dict[str, Any] = {"source_path": str(path), "format": suffix or "unknown"}

    if suffix in {".pt", ".pth", ".bin"}:
        obj = torch.load(path, map_location="cpu")
    else:
        raw_bytes = path.read_bytes()
        payload = maybe_decompress(raw_bytes, suffix)
        obj = torch_load_bytes(payload)
        meta["compressed_bytes"] = len(raw_bytes)
        meta["decompressed_bytes"] = len(payload)

    if isinstance(obj, dict) and "__quant_format__" in obj:
        meta["quant_format"] = obj["__quant_format__"]
        state_dict = dequantize_clean_per_row(obj)
        meta["dequantized_from_quantized_artifact"] = True
    elif isinstance(obj, dict):
        state_dict = obj
        meta["dequantized_from_quantized_artifact"] = False
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")

    return state_dict, meta


LAYER_PATTERN = re.compile(r"blocks\.(\d+)\.")


def tensor_group(name: str) -> str:
    if name == "tok_emb.weight":
        return "embedding"
    if name == "skip_weights":
        return "skip"
    if ".attn.c_q.weight" in name:
        return "attn_q"
    if ".attn.c_k.weight" in name:
        return "attn_k"
    if ".attn.c_v.weight" in name:
        return "attn_v"
    if ".attn.proj.weight" in name:
        return "attn_out"
    if ".mlp.fc.weight" in name:
        return "mlp_in"
    if ".mlp.proj.weight" in name:
        return "mlp_out"
    if ".resid_mix" in name:
        return "resid_mix"
    return "other"


def extract_layer(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))


def energy_rank(singular_values: torch.Tensor, frac: float) -> int:
    energy = singular_values.square()
    total = float(energy.sum().item())
    if total <= 0.0:
        return 0
    cutoff = total * frac
    cumulative = torch.cumsum(energy, dim=0)
    return int(torch.searchsorted(cumulative, cutoff).item()) + 1


def summarize_matrix(
    name: str,
    tensor: torch.Tensor,
    rank_fracs: list[float],
    device: torch.device,
) -> dict[str, Any]:
    matrix = tensor.detach().to(device=device, dtype=torch.float32)
    singular_values = torch.linalg.svdvals(matrix).to("cpu")
    energy = singular_values.square()
    energy_total = float(energy.sum().item())
    probs = energy / max(energy_total, 1e-12)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
    min_dim = min(matrix.shape)

    ranks = {f"rank_{int(frac * 100)}": energy_rank(singular_values, frac) for frac in rank_fracs}
    rank_fracs_out = {
        f"rank_{int(frac * 100)}_frac": ranks[f"rank_{int(frac * 100)}"] / min_dim
        for frac in rank_fracs
    }

    top1_energy = float((energy[0] / energy_total).item()) if energy_total > 0 else 0.0
    top8_energy = (
        float((energy[: min(8, energy.numel())].sum() / energy_total).item())
        if energy_total > 0
        else 0.0
    )

    return {
        "name": name,
        "group": tensor_group(name),
        "layer": extract_layer(name),
        "shape": list(matrix.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "max_singular": float(singular_values[0].item()),
        "min_singular": float(singular_values[-1].item()),
        "condition_number": float(
            singular_values[0].item() / max(singular_values[-1].item(), 1e-12)
        ),
        "stable_rank": float(energy_total / max(singular_values[0].item() ** 2, 1e-12)),
        "effective_rank": float(math.exp(entropy)),
        "top1_energy_frac": top1_energy,
        "top8_energy_frac": top8_energy,
        "spectral_tail_ratio": float(
            singular_values[-1].item() / max(singular_values[0].item(), 1e-12)
        ),
        "spectral_decay_10": float(
            singular_values[min(9, singular_values.numel() - 1)].item()
            / max(singular_values[0].item(), 1e-12)
        ),
        **ranks,
        **rank_fracs_out,
    }


def analyze_state_dict(
    state_dict: dict[str, torch.Tensor],
    rank_fracs: list[float],
    device: torch.device,
) -> dict[str, Any]:
    matrices = []
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            matrices.append(summarize_matrix(name, tensor, rank_fracs, device))

    layer_summary: dict[int, dict[str, list[dict[str, Any]]]] = {}
    group_summary: dict[str, list[dict[str, Any]]] = {}
    for entry in matrices:
        group_summary.setdefault(entry["group"], []).append(entry)
        if entry["layer"] is not None:
            layer_summary.setdefault(entry["layer"], {}).setdefault(entry["group"], []).append(entry)

    def avg_entries(entries: list[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        numeric_keys = [
            "stable_rank",
            "effective_rank",
            "top1_energy_frac",
            "top8_energy_frac",
            "rank_90_frac",
            "rank_95_frac",
            "rank_99_frac",
            "spectral_decay_10",
        ]
        for key in numeric_keys:
            vals = [float(entry[key]) for entry in entries if key in entry]
            if vals:
                out[key] = sum(vals) / len(vals)
        return out

    return {
        "num_tensors": len(state_dict),
        "num_matrices": len(matrices),
        "matrices": matrices,
        "group_summary": {group: avg_entries(entries) for group, entries in group_summary.items()},
        "layer_summary": {
            str(layer): {group: avg_entries(entries) for group, entries in grouped.items()}
            for layer, grouped in sorted(layer_summary.items())
        },
    }


def print_table(title: str, rows: list[dict[str, Any]], top_k: int) -> None:
    print(f"\n== {title} ==")
    if not rows:
        print("(none)")
        return
    print(
        f"{'name':42} {'shape':>14} {'r90%':>6} {'r95%':>6} {'r99%':>6} "
        f"{'eff_rk':>8} {'stab_rk':>8} {'top1E':>7}"
    )
    for row in rows[:top_k]:
        shape = f"{row['shape'][0]}x{row['shape'][1]}"
        print(
            f"{row['name'][:42]:42} {shape:>14} "
            f"{row['rank_90_frac']:6.3f} {row['rank_95_frac']:6.3f} {row['rank_99_frac']:6.3f} "
            f"{row['effective_rank']:8.1f} {row['stable_rank']:8.1f} {row['top1_energy_frac']:7.3f}"
        )


def print_group_summary(group_summary: dict[str, dict[str, float]]) -> None:
    print("\n== Group Summary ==")
    print(
        f"{'group':14} {'r90%':>6} {'r95%':>6} {'r99%':>6} "
        f"{'eff_rk':>8} {'stab_rk':>8} {'top1E':>7}"
    )
    for group, stats in sorted(group_summary.items()):
        print(
            f"{group:14} {stats.get('rank_90_frac', 0.0):6.3f} "
            f"{stats.get('rank_95_frac', 0.0):6.3f} {stats.get('rank_99_frac', 0.0):6.3f} "
            f"{stats.get('effective_rank', 0.0):8.1f} {stats.get('stable_rank', 0.0):8.1f} "
            f"{stats.get('top1_energy_frac', 0.0):7.3f}"
        )


def print_layer_summary(layer_summary: dict[str, dict[str, dict[str, float]]]) -> None:
    print("\n== Layer Summary (rank_95_frac) ==")
    header = ["layer", "attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out"]
    print(" ".join(f"{item:>10}" for item in header))
    for layer, stats in layer_summary.items():
        row = [layer]
        for key in header[1:]:
            row.append(f"{stats.get(key, {}).get('rank_95_frac', float('nan')):0.3f}")
        print(" ".join(f"{item:>10}" for item in row))


def main() -> None:
    args = parse_args()
    path = resolve_checkpoint_path(args)
    rank_fracs = sorted(set(float(x) for x in args.rank_fracs))
    device = torch.device(args.device)

    state_dict, load_meta = load_state_dict(path)
    analysis = analyze_state_dict(state_dict, rank_fracs, device)
    analysis["load_meta"] = load_meta

    matrices = analysis["matrices"]
    by_rank95 = sorted(matrices, key=lambda row: row["rank_95_frac"])
    by_effective_rank = sorted(matrices, key=lambda row: row["effective_rank"])
    by_top1_energy = sorted(matrices, key=lambda row: row["top1_energy_frac"], reverse=True)

    print(f"Analyzed: {path}")
    print(json.dumps(load_meta, indent=2, sort_keys=True))
    print(f"Matrices analyzed: {analysis['num_matrices']}")

    print_group_summary(analysis["group_summary"])
    print_layer_summary(analysis["layer_summary"])
    print_table("Most Compressible By rank_95_frac", by_rank95, args.top_k)
    print_table("Most Compressible By effective_rank", by_effective_rank, args.top_k)
    print_table("Most Dominated By Top Singular Direction", by_top1_energy, args.top_k)

    if args.json_output is not None:
        args.json_output.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved JSON to {args.json_output}")


if __name__ == "__main__":
    main()
