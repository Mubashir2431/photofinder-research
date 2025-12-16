from __future__ import annotations
import json, os, platform, time
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class RunInfo:
    timestamp: float
    python: str
    platform: str
    processor: str | None

def get_run_info() -> Dict[str, Any]:
    info = RunInfo(
        timestamp=time.time(),
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        processor=platform.processor() or None,
    )
    return info.__dict__

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
