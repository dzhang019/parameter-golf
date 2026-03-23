from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+"
    r"(?:(?:train_loss:(?P<train_loss>[-0-9.]+))|(?:val_loss:(?P<val_loss>[-0-9.]+)\s+val_bpb:(?P<val_bpb>[-0-9.]+)))"
    r".*?train_time:(?P<train_time>\d+)ms\s+step_avg:(?P<step_avg>[-0-9.]+)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-0-9.]+)\s+val_bpb:(?P<val_bpb>[-0-9.]+)"
)
STOP_RE = re.compile(r"stopping_early: wallclock_cap train_time:(?P<train_time>\d+)ms step:(?P<step>\d+)/(?P<iterations>\d+)")
PARAMS_RE = re.compile(r"model_params:(?P<params>\d+)")
MEM_RE = re.compile(r"peak memory allocated:\s+(?P<alloc>\d+)\s+MiB\s+reserved:\s+(?P<reserved>\d+)\s+MiB")
SIZE_RE = re.compile(r"Serialized model int8\+zlib:\s+(?P<bytes>\d+)\s+bytes")
TOTAL_SIZE_RE = re.compile(r"Total submission size int8\+zlib:\s+(?P<bytes>\d+)\s+bytes")
BATCH_RE = re.compile(
    r"train_batch_tokens:(?P<batch>\d+)\s+train_seq_len:(?P<seq>\d+)\s+iterations:(?P<iterations>\d+)\s+warmup_steps:(?P<warmup>\d+)\s+max_wallclock_seconds:(?P<wallclock>[-0-9.]+)"
)
SEED_RE = re.compile(r"seed:(?P<seed>\d+)")
Q_RANK_RE = re.compile(r"(?:^|[_-])q(?P<qrank>\d+)(?:[_-]|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Parameter Golf training logs.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["records/track_10min_16mb/2026-03-22_danielsolutionv1/logs/*.txt"],
        help="Log files or globs to summarize.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("q_rank", "final_bpb", "prequant_bpb", "step_avg", "steps", "artifact_mb"),
        default="q_rank",
        help="Column to sort by.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if any(ch in pattern for ch in "*?[]"):
            paths.extend(sorted(Path().glob(pattern)))
        elif p.is_dir():
            paths.extend(sorted(p.glob("*.txt")))
        else:
            paths.append(p)
    return sorted(set(paths))


def q_rank_from_name(name: str) -> int | None:
    match = Q_RANK_RE.search(name)
    return int(match.group("qrank")) if match else None


def parse_log(path: Path) -> dict[str, object]:
    data: dict[str, object] = {
        "path": str(path),
        "run_id": path.stem,
        "q_rank": q_rank_from_name(path.stem),
    }
    last_step = None
    last_val = None
    best_val = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if (m := PARAMS_RE.search(line)):
            data["model_params"] = int(m.group("params"))
        if (m := BATCH_RE.search(line)):
            data["train_batch_tokens"] = int(m.group("batch"))
            data["train_seq_len"] = int(m.group("seq"))
            data["iterations"] = int(m.group("iterations"))
            data["warmup_steps"] = int(m.group("warmup"))
            data["max_wallclock_seconds"] = float(m.group("wallclock"))
        if (m := SEED_RE.search(line)):
            data["seed"] = int(m.group("seed"))
        if (m := MEM_RE.search(line)):
            data["peak_mem_alloc_mib"] = int(m.group("alloc"))
            data["peak_mem_reserved_mib"] = int(m.group("reserved"))
        if (m := SIZE_RE.search(line)):
            data["artifact_bytes"] = int(m.group("bytes"))
        if (m := TOTAL_SIZE_RE.search(line)):
            data["submission_bytes"] = int(m.group("bytes"))
        if (m := FINAL_RE.search(line)):
            data["final_val_loss"] = float(m.group("val_loss"))
            data["final_val_bpb"] = float(m.group("val_bpb"))
        if (m := STOP_RE.search(line)):
            data["stopped_early"] = True
            data["stop_train_time_ms"] = int(m.group("train_time"))
            data["stop_step"] = int(m.group("step"))
        if (m := STEP_RE.search(line)):
            step_info = {
                "step": int(m.group("step")),
                "train_time_ms": int(m.group("train_time")),
                "step_avg_ms": float(m.group("step_avg")),
            }
            if m.group("val_bpb") is not None:
                step_info["val_loss"] = float(m.group("val_loss"))
                step_info["val_bpb"] = float(m.group("val_bpb"))
                last_val = step_info
                if best_val is None or step_info["val_bpb"] < best_val["val_bpb"]:
                    best_val = step_info
            last_step = step_info

    if last_step is not None:
        data["last_step"] = last_step["step"]
        data["last_train_time_ms"] = last_step["train_time_ms"]
        data["last_step_avg_ms"] = last_step["step_avg_ms"]
        if last_step["train_time_ms"] > 0 and last_step["step"] > 0:
            data["steps_per_sec"] = 1000.0 / last_step["step_avg_ms"]
    if last_val is not None:
        data["last_prequant_val_bpb"] = last_val["val_bpb"]
        data["last_prequant_val_loss"] = last_val["val_loss"]
        data["last_val_step"] = last_val["step"]
    if best_val is not None:
        data["best_prequant_val_bpb"] = best_val["val_bpb"]
        data["best_val_step"] = best_val["step"]
    if "artifact_bytes" in data:
        data["artifact_mb"] = float(data["artifact_bytes"]) / 1_000_000.0
    return data


def sort_key(item: dict[str, object], field: str):
    value = item.get(field_map(field))
    if value is None:
        return math.inf
    return value


def field_map(field: str) -> str:
    return {
        "q_rank": "q_rank",
        "final_bpb": "final_val_bpb",
        "prequant_bpb": "last_prequant_val_bpb",
        "step_avg": "last_step_avg_ms",
        "steps": "last_step",
        "artifact_mb": "artifact_mb",
    }[field]


def fmt(value: object, spec: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, spec or ".4f")
    return str(value)


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.paths)
    rows = [parse_log(path) for path in paths if path.exists()]
    rows.sort(key=lambda row: sort_key(row, args.sort_by))

    if not rows:
        print("No log files found.")
        return

    header = [
        "run_id",
        "q_rank",
        "params",
        "steps",
        "step_avg_ms",
        "prequant_bpb",
        "final_bpb",
        "artifact_mb",
        "peak_mem_mib",
    ]
    print(" ".join(f"{col:>18}" for col in header))
    for row in rows:
        print(
            " ".join(
                [
                    f"{str(row.get('run_id', '-')):>18}",
                    f"{fmt(row.get('q_rank')):>18}",
                    f"{fmt(row.get('model_params')):>18}",
                    f"{fmt(row.get('last_step')):>18}",
                    f"{fmt(row.get('last_step_avg_ms'), '.2f'):>18}",
                    f"{fmt(row.get('last_prequant_val_bpb'), '.4f'):>18}",
                    f"{fmt(row.get('final_val_bpb'), '.4f'):>18}",
                    f"{fmt(row.get('artifact_mb'), '.3f'):>18}",
                    f"{fmt(row.get('peak_mem_alloc_mib')):>18}",
                ]
            )
        )


if __name__ == "__main__":
    main()
