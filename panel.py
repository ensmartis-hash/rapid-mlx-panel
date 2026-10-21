#!/usr/bin/env python3
"""Local control panel for rapid-mlx serve slots.

Slots (name, port, API name, weights, and serve flags) persist in
~/.config/rapid-mlx-panel/slots.json. The panel binds to 127.0.0.1 only.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = 1
PANEL_PORT_DEFAULT = 8770
HOST = "127.0.0.1"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}\Z")
ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
API_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:/@-]{0,199}\Z")

VALUE_FLAGS = {
    "--host",
    "--port",
    "--api-key",
    "--tool-call-parser",
    "--reasoning-parser",
    "--log-level",
    "--mcp-config",
    "--cors-origins",
    "--trusted-hosts",
    "--served-model-name",
    "--max-tokens",
    "--gpu-memory-utilization",
    "--embedding-model",
    "--speculative-config",
    "--context-length",
    "--kv-cache-dtype",
    "--cache-memory-mb",
    "--cache-memory-percent",
    "--hybrid-cache-entries",
    "--rerank-model",
    "--default-temperature",
    "--default-top-p",
    "--default-top-k",
    "--name",
    "--drafter",
    "--context",
    "--learn-dir",
    "--learn-gib",
    "--learn-conversations",
    "--learn-daily-gib",
    "--slide-graph",
}

RESERVED_EXTRA = {
    "--port",
    "--host",
    "--served-model-name",
    "--kv-cache-dtype",
    "--kv-cache-turboquant",
    "--speculative-config",
    "--no-spec-decode",
    "--embedding-model",
    "--mllm",
    "--text-only",
    "--disable-prefix-cache",
    "--context-length",
    "--gpu-memory-utilization",
    "--cache-memory-mb",
    "--hybrid-cache-entries",
    "--pin-system-prompt",
    "--max-tokens",
    "--drafter",
    "--cortheon",
    "--learn",
    "--slide",
    "--name",
    "--context",
}

LOCK = threading.Lock()
_MODELS_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_CATALOG_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}

LANES = {"auto", "text", "vision"}
PREFIX_MODES = {"default", "off"}
KV_DTYPES = {"default", "bf16", "int8", "int4"}
KV_TURBOQUANT = {"default", "off", "v4", "k8v4"}
SPEC_MODES = {"default", "off", "mtp"}
ENGINES = {"rapid-mlx", "tensorfold"}

_CATALOG_SCRIPT = r"""
import json

from rapid_mlx.model_aliases import catalog_alias_for, list_profiles

try:
    from rapid_mlx.user_aliases import load_user_aliases
    user_aliases = load_user_aliases()
except Exception:
    user_aliases = {}

def pack(alias, profile):
    draft = (profile.mtp_draft_model or "").strip() or None
    native_draft = (getattr(profile, "native_mtp_draft_model", None) or "").strip() or None
    native = bool(getattr(profile, "supports_native_mtp", False))
    has_mtp = bool(draft or native)
    return {
        "alias": alias,
        "hf_path": profile.hf_path,
        "mtp_draft_model": draft,
        "native_mtp_draft_model": native_draft,
        "supports_native_mtp": native,
        "mtp_speculative_tokens": profile.mtp_speculative_tokens if has_mtp else None,
        "mtp_default_enabled": bool(getattr(profile, "mtp_default_enabled", True)),
        "mtp_continuous_tier": getattr(profile, "mtp_continuous_batching_tier", "unknown"),
        "turboquant_tier": getattr(profile, "turboquant_tier", "unknown"),
        "supports_image_input": bool(getattr(profile, "supports_image_input", False)),
        "is_text_only": bool(getattr(profile, "is_text_only", False)),
    }

index = {}
for alias, profile in sorted(list_profiles().items()):
    packed = pack(alias, profile)
    index[alias] = packed
    path = profile.hf_path
    if path and path not in index:
        index[path] = packed
for name, target in sorted(user_aliases.items()):
    if name in index:
        continue
    canonical = catalog_alias_for(target) or (target if target in index else None)
    if canonical in index:
        index[name] = index[canonical]
print(json.dumps(index, separators=(",", ":")))
"""


def state_dir() -> Path:
    override = os.environ.get("RAPID_MLX_PANEL_STATE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "rapid-mlx-panel"


def slots_path() -> Path:
    return state_dir() / "slots.json"


def runtime_path() -> Path:
    return state_dir() / "runtime.json"


def log_dir() -> Path:
    return state_dir() / "logs"


def rapid_mlx_bin() -> str:
    override = os.environ.get("RAPID_MLX_BIN", "").strip()
    if override:
        return override
    return "/Users/otto/.local/bin/rapid-mlx"


def rapid_mlx_python() -> str | None:
    """Interpreter that can import rapid_mlx. None for a stand-in binary."""
    try:
        resolved = Path(rapid_mlx_bin()).resolve()
    except OSError:
        return None
    candidate = resolved.parent / "python3"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def tensorfold_bin() -> str:
    override = os.environ.get("TENSORFOLD_BIN", "").strip()
    if override:
        return override
    return "/Users/otto/.local/bin/tensorfold"


def _blank_options() -> dict[str, Any]:
    return {
        "engine": "rapid-mlx",
        "drafter": "",
        "cortheon": False,
        "learn": False,
        "slide": False,
        "embedding_model": "",
        "lane": "auto",
        "prefix_cache": "default",
        "kv_cache_dtype": "default",
        "kv_cache_turboquant": "default",
        "spec_decode": "default",
        "mtp_tokens": 3,
        "mtp_tokens_auto": False,
        "mtp_model": "",
        "context_length": None,
        "gpu_memory_utilization": None,
        "cache_memory_mb": None,
        "hybrid_cache_entries": None,
        "pin_system_prompt": False,
        "max_tokens": None,
        "extra_args": "",
    }


def _seed_slots() -> list[dict[str, Any]]:
    """Match the serves already used on this Mac. Applied once, on first run."""
    base = _blank_options()
    smart = {
        **base,
        "id": "smart",
        "name": "smart",
        "port": 8000,
        "api_name": "",
        "model": "qwen3.8-27b-4bit",
    }
    fast = {
        **base,
        "id": "fast",
        "name": "fast",
        "port": 8004,
        "api_name": "",
        "model": "qwen3-4b-instruct-2507-4bit",
        "embedding_model": "mlx-community/embeddinggemma-300m-8bit",
    }
    uncensored = {
        **base,
        "id": "uncensored",
        "name": "uncensored",
        "port": 8005,
        "api_name": "",
        "model": "orcarouter/Qwen3.8-27B-Uncensored-MLX-4bit",
        "spec_decode": "mtp",
        "mtp_tokens": 3,
        "mtp_model": "orcarouter/Qwen3.8-27B-Uncensored-MLX",
        "gpu_memory_utilization": 1.0,
    }
    vision = {
        **base,
        "id": "vision",
        "name": "vision",
        "port": 8006,
        "api_name": "",
        "model": "qwen3-vl-4b-4bit",
        "lane": "vision",
    }
    return [smart, fast, uncensored, vision]


class PanelError(ValueError):
    pass


def _read_json(path: Path, default: Any) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PanelError(f"kan inte läsa {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def load_slots() -> list[dict[str, Any]]:
    path = slots_path()
    if not path.exists():
        slots = [] if os.environ.get("RAPID_MLX_PANEL_NO_SEED") == "1" else _seed_slots()
        _write_json(path, {"version": SCHEMA_VERSION, "slots": slots})
        return slots
    raw = _read_json(path, {"version": SCHEMA_VERSION, "slots": []})
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        raise PanelError(f"okänd version i {path}")
    slots = raw.get("slots")
    if not isinstance(slots, list):
        raise PanelError(f"'slots' i {path} måste vara en lista")
    return [_normalize_slot(item) for item in slots]


def load_runtime() -> dict[str, Any]:
    raw = _read_json(runtime_path(), {"version": SCHEMA_VERSION, "pids": {}})
    if not isinstance(raw, dict):
        return {"version": SCHEMA_VERSION, "pids": {}}
    pids = raw.get("pids")
    if not isinstance(pids, dict):
        pids = {}
    return {"version": SCHEMA_VERSION, "pids": pids}


def save_slots(slots: list[dict[str, Any]]) -> None:
    _write_json(slots_path(), {"version": SCHEMA_VERSION, "slots": slots})


def save_runtime(runtime: dict[str, Any]) -> None:
    _write_json(runtime_path(), runtime)


def _normalize_slot(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise PanelError("en plats måste vara ett objekt")
    slot = _blank_options()
    slot.update(
        {
            "id": str(item.get("id") or ""),
            "name": str(item.get("name") or "").strip(),
            "port": item.get("port"),
            "api_name": str(item.get("api_name") or "").strip(),
            "model": str(item.get("model") or "").strip(),
        }
    )
    for key in _blank_options():
        if key in item:
            slot[key] = item[key]
    slot["engine"] = str(slot.get("engine") or "rapid-mlx").strip()
    slot["embedding_model"] = str(slot.get("embedding_model") or "").strip()
    slot["drafter"] = str(slot.get("drafter") or "").strip()
    slot["mtp_model"] = str(slot.get("mtp_model") or "").strip()
    slot["extra_args"] = str(slot.get("extra_args") or "").strip()
    slot["pin_system_prompt"] = _as_bool(slot.get("pin_system_prompt"))
    slot["mtp_tokens_auto"] = _as_bool(slot.get("mtp_tokens_auto"))
    slot["cortheon"] = _as_bool(slot.get("cortheon"))
    slot["learn"] = _as_bool(slot.get("learn"))
    slot["slide"] = _as_bool(slot.get("slide"))
    return _validate_slot(slot)


def _optional_int(value: Any, field: str, *, minimum: int, maximum: int) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PanelError(f"{field} måste vara ett heltal") from exc
    if number < minimum or number > maximum:
        raise PanelError(f"{field} måste vara mellan {minimum} och {maximum}")
    return number


def _optional_float(value: Any, field: str, *, minimum: float, maximum: float) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PanelError(f"{field} måste vara ett tal") from exc
    if number < minimum or number > maximum:
        raise PanelError(f"{field} måste vara mellan {minimum} och {maximum}")
    return number


def _validate_slot(slot: dict[str, Any]) -> dict[str, Any]:
    slot_id = str(slot.get("id") or "").strip().lower()
    if not ID_RE.fullmatch(slot_id):
        raise PanelError("platsens id får bara innehålla små bokstäver, siffror och bindestreck")
    name = str(slot.get("name") or "").strip()
    if not NAME_RE.fullmatch(name):
        raise PanelError("namnet måste börja med en bokstav eller siffra och vara högst 64 tecken")
    port = _optional_int(slot.get("port"), "port", minimum=1, maximum=65535)
    if port is None:
        raise PanelError("port krävs")
    panel_port = int(os.environ.get("RAPID_MLX_PANEL_PORT", str(PANEL_PORT_DEFAULT)))
    if port == panel_port:
        raise PanelError(f"port {port} används av panelen")
    api_name = str(slot.get("api_name") or "").strip()
    if api_name and not API_NAME_RE.fullmatch(api_name):
        raise PanelError("API-namnet får inte innehålla mellanslag")
    engine = str(slot.get("engine") or "rapid-mlx").strip()
    if engine not in ENGINES:
        raise PanelError("motorn måste vara rapid-mlx eller tensorfold")
    model = str(slot.get("model") or "").strip()
    if any(ord(char) < 32 for char in model):
        raise PanelError("vikterna får inte innehålla styrtecken")
    if len(model) > 500:
        raise PanelError("vikterna är för långa")
    drafter = str(slot.get("drafter") or "").strip()
    if any(ord(char) < 32 for char in drafter) or len(drafter) > 500:
        raise PanelError("draftern är ogiltig")
    embedding = str(slot.get("embedding_model") or "").strip()
    if any(ord(char) < 32 for char in embedding) or len(embedding) > 500:
        raise PanelError("embedding-modellen är ogiltig")
    lane = str(slot.get("lane") or "auto")
    if lane not in LANES:
        raise PanelError("körfält måste vara auto, text eller vision")
    prefix = str(slot.get("prefix_cache") or "default")
    if prefix not in PREFIX_MODES:
        raise PanelError("prefix-cache måste vara default eller off")
    kv = str(slot.get("kv_cache_dtype") or "default")
    if kv not in KV_DTYPES:
        raise PanelError("kv-cache måste vara default, bf16, int8 eller int4")
    turboquant = str(slot.get("kv_cache_turboquant") or "default")
    if turboquant not in KV_TURBOQUANT:
        raise PanelError("TurboQuant måste vara default, off, v4 eller k8v4")
    if kv != "default" and turboquant != "default":
        raise PanelError("välj antingen en kv-dtype eller TurboQuant")
    spec = str(slot.get("spec_decode") or "default")
    if spec not in SPEC_MODES:
        raise PanelError("spekulativ avkodning måste vara default, off eller mtp")
    if lane == "vision" and spec == "mtp":
        raise PanelError("vision-körfältet kan inte kombineras med MTP")
    mtp_tokens_auto = _as_bool(slot.get("mtp_tokens_auto"))
    mtp_tokens = _optional_int(slot.get("mtp_tokens"), "mtp-token", minimum=1, maximum=16)
    if mtp_tokens is None and not mtp_tokens_auto:
        mtp_tokens = 3
    mtp_model = str(slot.get("mtp_model") or "").strip()
    if any(ord(char) < 32 for char in mtp_model) or len(mtp_model) > 500:
        raise PanelError("MTP-modellen är ogiltig")
    extra = str(slot.get("extra_args") or "")
    if extra:
        try:
            extra_tokens = shlex.split(extra)
        except ValueError as exc:
            raise PanelError(f"extra flaggor går inte att tolka: {exc}") from exc
        flag_names = {
            token.split("=", 1)[0] for token in extra_tokens if token.startswith("--")
        }
        blocked = RESERVED_EXTRA.intersection(flag_names)
        if blocked:
            raise PanelError(
                "de flaggorna har egna fält: " + ", ".join(sorted(blocked))
            )
    else:
        extra_tokens = []
    cleaned = _blank_options()
    cleaned.update(
        {
            "id": slot_id,
            "name": name,
            "port": port,
            "api_name": api_name,
            "model": model,
            "engine": engine,
            "drafter": drafter,
            "cortheon": _as_bool(slot.get("cortheon")),
            "learn": _as_bool(slot.get("learn")),
            "slide": _as_bool(slot.get("slide")),
            "embedding_model": embedding,
            "lane": lane,
            "prefix_cache": prefix,
            "kv_cache_dtype": kv,
            "kv_cache_turboquant": turboquant,
            "spec_decode": spec,
            "mtp_tokens": mtp_tokens,
            "mtp_tokens_auto": mtp_tokens_auto,
            "mtp_model": mtp_model,
            "context_length": _optional_int(
                slot.get("context_length"), "kontext", minimum=1, maximum=1_048_576
            ),
            "gpu_memory_utilization": _optional_float(
                slot.get("gpu_memory_utilization"),
                "gpu-minne",
                minimum=0.01,
                maximum=1.0,
            ),
            "cache_memory_mb": _optional_int(
                slot.get("cache_memory_mb"), "cache-mb", minimum=1, maximum=1_048_576
            ),
            "hybrid_cache_entries": _optional_int(
                slot.get("hybrid_cache_entries"),
                "hybrid-cache",
                minimum=0,
                maximum=10_000,
            ),
            "pin_system_prompt": bool(slot.get("pin_system_prompt")),
            "max_tokens": _optional_int(
                slot.get("max_tokens"), "max tokens", minimum=1, maximum=1_048_576
            ),
            "extra_args": " ".join(shlex.quote(token) for token in extra_tokens),
        }
    )
    return cleaned


def _reject_duplicates(slots: list[dict[str, Any]]) -> None:
    names: set[str] = set()
    ports: set[int] = set()
    ids: set[str] = set()
    for slot in slots:
        folded = slot["name"].casefold()
        if folded in names:
            raise PanelError(f"namnet {slot['name']!r} finns redan")
        if slot["port"] in ports:
            raise PanelError(f"port {slot['port']} används av en annan plats")
        if slot["id"] in ids:
            raise PanelError(f"id {slot['id']!r} finns redan")
        names.add(folded)
        ports.add(slot["port"])
        ids.add(slot["id"])


def slugify(name: str) -> str:
    lowered = name.casefold().strip()
    chars = [char if char.isalnum() else "-" for char in lowered]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug[:64].strip("-")
    if not slug or not ID_RE.fullmatch(slug):
        slug = "plats"
    return slug


def _unique_id(name: str, slots: list[dict[str, Any]]) -> str:
    taken = {slot["id"] for slot in slots}
    base = slugify(name)
    if base not in taken:
        return base
    for number in range(2, 100):
        candidate = f"{base}-{number}"[:64].strip("-")
        if candidate not in taken and ID_RE.fullmatch(candidate):
            return candidate
    return uuid.uuid4().hex[:12]


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _tensorfold_argv(slot: dict[str, Any]) -> list[str]:
    """Zig tensorfold-native. Drafter, Cortheon, learn and slide are opt-in."""
    argv = [
        tensorfold_bin(),
        "serve",
        slot["model"],
        "--host",
        HOST,
        "--port",
        str(slot["port"]),
    ]
    api_name = slot.get("api_name") or ""
    if api_name:
        argv.extend(["--name", api_name])
    if slot.get("drafter"):
        argv.extend(["--drafter", slot["drafter"]])
    if slot.get("cortheon"):
        argv.append("--cortheon")
    if slot.get("learn"):
        argv.append("--learn")
    if slot.get("slide"):
        argv.append("--slide")
    if slot.get("context_length"):
        argv.extend(["--context", str(slot["context_length"])])
    if slot.get("max_tokens"):
        argv.extend(["--max-tokens", str(slot["max_tokens"])])
    if slot.get("extra_args"):
        argv.extend(shlex.split(slot["extra_args"]))
    return argv


def build_argv(slot: dict[str, Any]) -> list[str]:
    if not slot.get("model"):
        raise PanelError("välj vikter innan platsen startas")
    if slot.get("engine") == "tensorfold":
        return _tensorfold_argv(slot)
    argv = [
        rapid_mlx_bin(),
        "serve",
        slot["model"],
        "--host",
        HOST,
        "--port",
        str(slot["port"]),
    ]
    api_name = slot.get("api_name") or ""
    if api_name and api_name != slot["model"]:
        argv.extend(["--served-model-name", api_name])
    if slot.get("embedding_model"):
        argv.extend(["--embedding-model", slot["embedding_model"]])
    if slot.get("lane") == "text":
        argv.append("--text-only")
    elif slot.get("lane") == "vision":
        argv.append("--mllm")
    if slot.get("prefix_cache") == "off":
        argv.append("--disable-prefix-cache")
    if slot.get("kv_cache_dtype") not in (None, "", "default"):
        argv.extend(["--kv-cache-dtype", slot["kv_cache_dtype"]])
    turboquant = slot.get("kv_cache_turboquant") or "default"
    if turboquant == "off":
        argv.extend(["--kv-cache-turboquant", "none"])
    elif turboquant in {"v4", "k8v4"}:
        argv.extend(["--kv-cache-turboquant", turboquant])
    if slot.get("spec_decode") == "off":
        argv.append("--no-spec-decode")
    elif slot.get("spec_decode") == "mtp":
        config: dict[str, Any] = {"method": "mtp"}
        if not slot.get("mtp_tokens_auto"):
            config["num_speculative_tokens"] = int(slot.get("mtp_tokens") or 3)
        if slot.get("mtp_model"):
            config["model"] = slot["mtp_model"]
        argv.extend(
            ["--speculative-config", json.dumps(config, separators=(",", ":"))]
        )
    if slot.get("context_length"):
        argv.extend(["--context-length", str(slot["context_length"])])
    if slot.get("gpu_memory_utilization") is not None:
        argv.extend(
            ["--gpu-memory-utilization", _format_number(slot["gpu_memory_utilization"])]
        )
    if slot.get("cache_memory_mb"):
        argv.extend(["--cache-memory-mb", str(slot["cache_memory_mb"])])
    if slot.get("hybrid_cache_entries") not in (None, ""):
        argv.extend(["--hybrid-cache-entries", str(slot["hybrid_cache_entries"])])
    if slot.get("pin_system_prompt"):
        argv.append("--pin-system-prompt")
    if slot.get("max_tokens"):
        argv.extend(["--max-tokens", str(slot["max_tokens"])])
    if slot.get("extra_args"):
        argv.extend(shlex.split(slot["extra_args"]))
    return argv


def command_preview(slot: dict[str, Any]) -> str:
    try:
        argv = build_argv(slot)
    except PanelError as exc:
        return str(exc)
    return " ".join(shlex.quote(part) for part in argv)


def parse_serve_command(command: str) -> dict[str, Any] | None:
    """Return a serve process description, or None when the line is not one."""
    tokens = command.split()
    if not tokens or os.path.basename(tokens[0]) == "caffeinate":
        return None
    serve_at = None
    engine = None
    for index, token in enumerate(tokens):
        base = os.path.basename(token)
        if "rapid-mlx" in token or "rapid_mlx" in token:
            engine = "rapid-mlx"
        elif "tensorfold" in base:
            engine = "tensorfold"
        if token == "serve" and engine:
            serve_at = index
            break
    if serve_at is None or engine is None:
        return None
    model = ""
    port = None
    served_name = None
    index = serve_at + 1
    model_seen = False
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            key, _, inline = token.partition("=")
            if inline:
                if key == "--port":
                    port = inline
                elif key in {"--served-model-name", "--name"}:
                    served_name = inline
                index += 1
                continue
            if key in VALUE_FLAGS and index + 1 < len(tokens):
                value = tokens[index + 1]
                if key == "--port":
                    port = value
                elif key in {"--served-model-name", "--name"}:
                    served_name = value
                index += 2
                continue
            index += 1
            continue
        if not model_seen:
            model = token
            model_seen = True
        index += 1
    return {
        "model": model or "(okänd)",
        "port": int(port) if port and port.isdigit() else None,
        "served_name": served_name,
        "engine": engine,
        "command": command,
    }


def scan_serve_processes() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["ps", "-axww", "-o", "pid=", "-o", "etime=", "-o", "command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    found: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, etime, command = parts
        if not pid_text.isdigit():
            continue
        parsed = parse_serve_command(command)
        if parsed is None:
            continue
        parsed["pid"] = int(pid_text)
        parsed["uptime"] = etime
        found.append(parsed)
    return found


def _reap(pid: int) -> bool:
    """Reap a child zombie. Returns True when this process has collected it."""
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return waited == pid


def _pid_alive(pid: int) -> bool:
    if _reap(pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((HOST, port)) == 0


def alias_enables_mtp(profile: dict[str, Any] | None, lane: str) -> bool:
    """Whether rapid-mlx turns MTP on when no speculative flag is passed."""
    if not profile or lane == "vision":
        return False
    if not profile.get("mtp_default_enabled", True):
        return False
    if profile.get("supports_native_mtp"):
        return True
    return profile.get("mtp_continuous_tier") == "verified" and bool(profile.get("mtp_draft_model"))


def _draft_phrase(profile: dict[str, Any]) -> str:
    draft = str(profile.get("mtp_draft_model") or "")
    depth = profile.get("mtp_speculative_tokens")
    if profile.get("supports_native_mtp") and not draft:
        return f"inbyggd MTP, djup {depth}" if depth else "inbyggd MTP"
    if draft and draft == profile.get("hf_path"):
        return f"samma checkpoint, djup {depth}"
    if draft:
        return f"{draft}, djup {depth}"
    return ""


def _tensorfold_advice(slot: dict[str, Any]) -> str:
    parts = ["Zig-motorn tensorfold."]
    drafter = str(slot.get("drafter") or "").strip()
    if drafter:
        parts.append(f"Drafter är {drafter}.")
        if drafter.casefold() == "z-lab/qwen3.8-27b-dflash2":
            parts.append(
                "tensorfold-native 1.1.1 tar inte det utkastet, dess model_type är qwen3."
            )
    else:
        parts.append("Ingen drafter.")
    features: list[str] = []
    if slot.get("cortheon"):
        features.append("Cortheon är på")
    if slot.get("learn"):
        features.append("learn sparar prompt-cachen")
    if slot.get("slide"):
        features.append("slide skriver om vikterna i checkpointen")
    if features:
        parts.append(", ".join(features) + ".")
    else:
        parts.append("Cortheon, learn och slide är av.")
    return " ".join(parts)


def describe_slot(slot: dict[str, Any], profile: dict[str, Any] | None) -> str:
    """One or two sentences about what the saved choices do for this weight."""
    if slot.get("engine") == "tensorfold":
        return _tensorfold_advice(slot)
    lane = str(slot.get("lane") or "auto")
    spec = str(slot.get("spec_decode") or "default")
    dtype = str(slot.get("kv_cache_dtype") or "default")
    turboquant = str(slot.get("kv_cache_turboquant") or "default")
    mtp_active = spec == "mtp" or (spec == "default" and alias_enables_mtp(profile, lane))
    parts: list[str] = []
    verified_mtp = bool(
        mtp_active and profile and profile.get("mtp_continuous_tier") == "verified"
    )
    if turboquant in {"v4", "k8v4"}:
        sentence = f"KV-cache blir TurboQuant {turboquant}."
        if verified_mtp:
            sentence += (
                " Verifierad MTP är kvalificerad med bf16."
                " Kontinuerlig MTP tillsammans med TurboQuant avvisas vid start."
            )
        elif profile is None:
            sentence += " Vikten har inget katalogkort."
        elif profile.get("turboquant_tier") != "k8v4_verified":
            sentence += " Aliaset är inte markerat som k8v4-verifierat."
        parts.append(sentence)
    elif turboquant == "off":
        parts.append("TurboQuant är uttryckligen av.")
    elif dtype in {"int8", "int4"}:
        sentence = f"KV-cache blir {dtype}."
        if verified_mtp:
            sentence += " Verifierad MTP är kvalificerad med bf16."
        parts.append(sentence)
    elif mtp_active:
        parts.append("KV-cache blir bf16 så länge MTP är igång.")
    elif profile and profile.get("turboquant_tier") == "k8v4_verified":
        parts.append("KV-cache blir TurboQuant k8v4, aliasets standard.")
    else:
        parts.append("KV-cache blir bf16.")
    parts.append(_mtp_sentence(slot, profile, lane, spec))
    return " ".join(part for part in parts if part)


def _mtp_sentence(
    slot: dict[str, Any],
    profile: dict[str, Any] | None,
    lane: str,
    spec: str,
) -> str:
    if lane == "vision":
        return "Vision-körfältet lämnar MTP av."
    if spec == "off":
        return "MTP är av."
    phrase = _draft_phrase(profile) if profile else ""
    if spec == "default":
        if alias_enables_mtp(profile, lane):
            return f"Standard slår på MTP med {phrase}."
        if phrase:
            return f"Standard lämnar MTP av. Katalogen har utkast {phrase}."
        if profile:
            return "Standard lämnar MTP av. Katalogen har inget MTP-utkast."
        return "Okatalogiserad vikt. Standard lämnar MTP av."
    chosen = str(slot.get("mtp_model") or "").strip()
    bits: list[str] = []
    if chosen:
        bits.append(f"Utkastet är {chosen}.")
    elif phrase:
        bits.append(f"Utkastet hämtas från katalogen: {phrase}.")
    elif profile and profile.get("supports_native_mtp"):
        bits.append("Inbyggd MTP använder målvikten.")
    else:
        bits.append("Katalogen har inget utkast. Ange ett, annars avbryts starten.")
    if slot.get("mtp_tokens_auto") and profile and profile.get("mtp_speculative_tokens"):
        bits.append(f"Djupet hämtas från katalogen ({profile['mtp_speculative_tokens']}).")
    elif not slot.get("mtp_tokens_auto"):
        bits.append(f"Djupet är {slot.get('mtp_tokens') or 3}.")
    return " ".join(bits)


def load_catalog() -> dict[str, Any]:
    """Alias and repo id → MTP and TurboQuant facts. Empty when rapid-mlx is absent."""
    now = time.time()
    cached = _CATALOG_CACHE["payload"]
    if isinstance(cached, dict) and now - _CATALOG_CACHE["at"] < 60:
        return cached
    python = rapid_mlx_python()
    if python is None:
        return {}
    try:
        completed = subprocess.run(
            [python, "-c", _CATALOG_SCRIPT],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return cached if isinstance(cached, dict) else {}
    start = completed.stdout.find("{")
    if completed.returncode != 0 or start < 0:
        return cached if isinstance(cached, dict) else {}
    try:
        payload = json.loads(completed.stdout[start:])
    except json.JSONDecodeError:
        return cached if isinstance(cached, dict) else {}
    if not isinstance(payload, dict):
        return cached if isinstance(cached, dict) else {}
    _CATALOG_CACHE["at"] = now
    _CATALOG_CACHE["payload"] = payload
    return payload


def classify(
    slots: list[dict[str, Any]],
    runtime: dict[str, Any],
    processes: list[dict[str, Any]],
    profiles: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    known = profiles or {}
    pids = runtime.get("pids", {})
    claimed: set[int] = set()
    enriched: list[dict[str, Any]] = []
    for slot in slots:
        view = dict(slot)
        view["command"] = command_preview(slot)
        profile = known.get(slot.get("model") or "")
        view["advice"] = describe_slot(slot, profile if isinstance(profile, dict) else None)
        on_port = [proc for proc in processes if proc.get("port") == slot["port"]]
        owned_pid = None
        record = pids.get(slot["id"])
        if isinstance(record, dict) and isinstance(record.get("pid"), int):
            owned_pid = record["pid"]
        owned = [proc for proc in on_port if proc["pid"] == owned_pid]
        chosen = owned[0] if owned else (on_port[0] if on_port else None)
        if chosen is not None:
            view["status"] = "running" if chosen["pid"] == owned_pid else "external"
            view["pid"] = chosen["pid"]
            view["uptime"] = chosen.get("uptime")
            view["live_command"] = chosen.get("command")
            view["live_model"] = chosen.get("model")
            claimed.add(chosen["pid"])
        elif owned_pid and _pid_alive(owned_pid):
            view["status"] = "detached"
            view["pid"] = owned_pid
            view["uptime"] = None
            view["live_command"] = None
            view["live_model"] = None
        else:
            view["status"] = "stopped"
            view["pid"] = None
            view["uptime"] = None
            view["live_command"] = None
            view["live_model"] = None
        enriched.append(view)
    unassigned = [proc for proc in processes if proc["pid"] not in claimed]
    return enriched, unassigned


def prune_runtime(runtime: dict[str, Any], processes: list[dict[str, Any]]) -> bool:
    del processes  # liveness, not the ps snapshot, decides whether a pid is stale
    pids = runtime.setdefault("pids", {})
    stale = [
        slot_id
        for slot_id, record in pids.items()
        if not isinstance(record, dict)
        or not isinstance(record.get("pid"), int)
        or not _pid_alive(record["pid"])
    ]
    for slot_id in stale:
        pids.pop(slot_id, None)
    return bool(stale)


def log_path_for(slot_id: str) -> Path:
    return log_dir() / f"{slot_id}.log"


def tail_log(slot_id: str, lines: int = 80) -> str:
    path = log_path_for(slot_id)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _append_log(slot_id: str, text: str) -> Path:
    path = log_path_for(slot_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
    return path


def start_slot(slot_id: str) -> dict[str, Any]:
    with LOCK:
        slots = load_slots()
        slot = next((item for item in slots if item["id"] == slot_id), None)
        if slot is None:
            raise PanelError("platsen finns inte")
        argv = build_argv(slot)
        processes = scan_serve_processes()
        if any(proc.get("port") == slot["port"] for proc in processes) or _port_open(slot["port"]):
            raise PanelError(f"port {slot['port']} är redan upptagen")
        binary = argv[0]
        if not os.path.exists(binary):
            raise PanelError(f"hittar inte motorn: {binary}")
        _append_log(slot_id, f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        _append_log(slot_id, command_preview(slot) + "\n")
        log_handle = log_path_for(slot_id).open("a", encoding="utf-8")
        try:
            try:
                process = subprocess.Popen(
                    argv,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=str(Path.home()),
                )
            except OSError as exc:
                raise PanelError(f"kunde inte starta motorn: {exc}") from exc
        finally:
            log_handle.close()
        pid = process.pid
        runtime = load_runtime()
        runtime.setdefault("pids", {})[slot_id] = {
            "pid": pid,
            "port": slot["port"],
            "started_at": int(time.time()),
        }
        save_runtime(runtime)
    deadline = time.time() + 0.8
    while time.time() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.05)
    exit_code = process.poll()
    if exit_code is None:
        # The panel tracks the pid itself. Mark the handle collected so
        # Popen's destructor does not warn while the server keeps running.
        process.returncode = 0
        return {"ok": True, "pid": pid, "status": "running"}
    with LOCK:
        runtime = load_runtime()
        record = runtime.get("pids", {}).get(slot_id)
        if isinstance(record, dict) and record.get("pid") == pid:
            runtime["pids"].pop(slot_id, None)
            save_runtime(runtime)
    raise PanelError(
        f"processen avslutades direkt (kod {exit_code}). " + tail_log(slot_id, 30)
    )


def _signal_pid(pid: int, sig: int, *, group: bool) -> None:
    try:
        if group:
            try:
                os.killpg(pid, sig)
                return
            except PermissionError:
                os.kill(pid, sig)
                return
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _terminate(pid: int, *, group: bool) -> None:
    if not _pid_alive(pid):
        return
    try:
        _signal_pid(pid, signal.SIGTERM, group=group)
    except PermissionError as exc:
        raise PanelError(f"får inte stoppa process {pid}: {exc}") from exc
    deadline = time.time() + 8
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)
    try:
        _signal_pid(pid, signal.SIGKILL, group=group)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        if not _pid_alive(pid):
            return
        raise PanelError(f"får inte stoppa process {pid}: {exc}") from exc


def stop_pid(pid: int, *, group: bool) -> None:
    processes = scan_serve_processes()
    if not any(proc["pid"] == pid for proc in processes):
        raise PanelError(f"process {pid} är inte en serve-process")
    _terminate(pid, group=group)
    with LOCK:
        runtime = load_runtime()
        pids = runtime.setdefault("pids", {})
        for slot_id, record in list(pids.items()):
            if isinstance(record, dict) and record.get("pid") == pid:
                pids.pop(slot_id, None)
        save_runtime(runtime)


def stop_slot(slot_id: str) -> dict[str, Any]:
    with LOCK:
        slots = load_slots()
        slot = next((item for item in slots if item["id"] == slot_id), None)
        if slot is None:
            raise PanelError("platsen finns inte")
        runtime = load_runtime()
        processes = scan_serve_processes()
        record = runtime.get("pids", {}).get(slot_id)
        owned = record.get("pid") if isinstance(record, dict) else None
        on_port = [proc for proc in processes if proc.get("port") == slot["port"]]
        target = None
        group = False
        if owned and any(proc["pid"] == owned for proc in on_port):
            target = owned
            group = True
        elif on_port:
            target = on_port[0]["pid"]
            group = False
        elif owned and _pid_alive(owned):
            target = owned
            group = True
    if target is None:
        return {"ok": True, "stopped": False}
    _terminate(target, group=group)
    with LOCK:
        runtime = load_runtime()
        pids = runtime.setdefault("pids", {})
        for key, record in list(pids.items()):
            if isinstance(record, dict) and record.get("pid") == target:
                pids.pop(key, None)
        save_runtime(runtime)
    return {"ok": True, "stopped": True, "pid": target}


def restart_slot(slot_id: str) -> dict[str, Any]:
    stop_slot(slot_id)
    deadline = time.time() + 10
    port = None
    with LOCK:
        slots = load_slots()
        slot = next((item for item in slots if item["id"] == slot_id), None)
        if slot is None:
            raise PanelError("platsen finns inte")
        port = slot["port"]
    while time.time() < deadline:
        still = any(proc.get("port") == port for proc in scan_serve_processes())
        if not still and not _port_open(port):
            break
        time.sleep(0.2)
    else:
        raise PanelError(f"port {port} släpptes inte efter stopp")
    return start_slot(slot_id)


def list_models() -> dict[str, Any]:
    now = time.time()
    if _MODELS_CACHE["payload"] is not None and now - _MODELS_CACHE["at"] < 20:
        return {**_MODELS_CACHE["payload"], "profiles": load_catalog()}
    binary = rapid_mlx_bin()
    try:
        completed = subprocess.run(
            [binary, "models", "--cached", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PanelError(f"kunde inte lista vikter: {exc}") from exc
    stdout = completed.stdout
    start = stdout.find("{")
    if completed.returncode != 0 or start < 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:500]
        raise PanelError(detail or "rapid-mlx models misslyckades")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as exc:
        raise PanelError(f"ogiltigt svar från rapid-mlx models: {exc}") from exc
    models = payload.get("cached") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise PanelError("rapid-mlx models saknar cached-listan")
    _MODELS_CACHE["at"] = now
    _MODELS_CACHE["payload"] = {"cached": models, "count": len(models)}
    return {**_MODELS_CACHE["payload"], "profiles": load_catalog()}


def preview_slot(body: dict[str, Any]) -> dict[str, str]:
    """Command and advice for an unsaved form. Nothing is written."""
    incoming = dict(body)
    incoming["id"] = "preview"
    if not str(incoming.get("name") or "").strip():
        incoming["name"] = "preview"
    if incoming.get("port") in (None, ""):
        incoming["port"] = 9
    try:
        slot = _normalize_slot(incoming)
    except PanelError as exc:
        return {"command": str(exc), "advice": ""}
    profile = load_catalog().get(slot["model"])
    return {
        "command": command_preview(slot),
        "advice": describe_slot(slot, profile if isinstance(profile, dict) else None),
    }


def snapshot() -> dict[str, Any]:
    profiles = load_catalog()
    with LOCK:
        slots = load_slots()
        runtime = load_runtime()
        processes = scan_serve_processes()
        if prune_runtime(runtime, processes):
            save_runtime(runtime)
        view, unassigned = classify(slots, runtime, processes, profiles)
    return {
        "slots": view,
        "unassigned": unassigned,
        "rapid_mlx": rapid_mlx_bin(),
        "state_dir": str(state_dir()),
    }


def create_slot(body: dict[str, Any]) -> dict[str, Any]:
    with LOCK:
        slots = load_slots()
        incoming = dict(body)
        incoming.pop("id", None)
        incoming["id"] = _unique_id(str(incoming.get("name") or "plats"), slots)
        slot = _normalize_slot(incoming)
        slots.append(slot)
        _reject_duplicates(slots)
        save_slots(slots)
    return slot


def update_slot(slot_id: str, body: dict[str, Any]) -> dict[str, Any]:
    with LOCK:
        slots = load_slots()
        index = next((i for i, item in enumerate(slots) if item["id"] == slot_id), None)
        if index is None:
            raise PanelError("platsen finns inte")
        incoming = dict(body)
        incoming["id"] = slot_id
        slot = _normalize_slot(incoming)
        slots[index] = slot
        _reject_duplicates(slots)
        save_slots(slots)
    return slot


def delete_slot(slot_id: str) -> None:
    with LOCK:
        slots = load_slots()
        slot = next((item for item in slots if item["id"] == slot_id), None)
        if slot is None:
            raise PanelError("platsen finns inte")
        processes = scan_serve_processes()
        if any(proc.get("port") == slot["port"] for proc in processes):
            raise PanelError("stoppa processen på porten innan platsen tas bort")
        slots = [item for item in slots if item["id"] != slot_id]
        save_slots(slots)
        runtime = load_runtime()
        runtime.get("pids", {}).pop(slot_id, None)
        save_runtime(runtime)


PAGE = (Path(__file__).resolve().parent / "panel.html").read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        host = self.headers.get("Host", "")
        return parsed.scheme in {"http", "https"} and parsed.netloc == host

    def _send(self, status: int, payload: Any, content_type: str = "application/json; charset=utf-8") -> None:
        if isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length > 1_000_000:
            raise PanelError("för stor förfrågan")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PanelError("ogiltig JSON") from exc
        if not isinstance(body, dict):
            raise PanelError("JSON-objektet måste vara ett objekt")
        return body

    def _guard(self) -> bool:
        if not self._origin_ok():
            self._send(403, {"error": "förfrågan kommer från ett annat ursprung"})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._send(200, snapshot())
                return
            if path == "/api/models":
                self._send(200, list_models())
                return
            if path.startswith("/api/slots/") and path.endswith("/log"):
                slot_id = path[len("/api/slots/") : -len("/log")].strip("/")
                if not ID_RE.fullmatch(slot_id):
                    raise PanelError("ogiltigt id")
                self._send(200, {"log": tail_log(slot_id)})
                return
            self._send(404, {"error": "saknas"})
        except PanelError as exc:
            self._send(400, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = urlparse(self.path).path.rstrip("/")
        try:
            if path == "/api/slots":
                self._send(201, create_slot(self._read_json()))
                return
            if path == "/api/preview":
                self._send(200, preview_slot(self._read_json()))
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 4 and parts[:2] == ["api", "slots"]:
                slot_id, action = parts[2], parts[3]
                if not ID_RE.fullmatch(slot_id):
                    raise PanelError("ogiltigt id")
                if action == "start":
                    self._send(200, start_slot(slot_id))
                    return
                if action == "stop":
                    self._send(200, stop_slot(slot_id))
                    return
                if action == "restart":
                    self._send(200, restart_slot(slot_id))
                    return
            if len(parts) == 4 and parts[:2] == ["api", "processes"] and parts[3] == "stop":
                if not parts[2].isdigit():
                    raise PanelError("ogiltigt pid")
                stop_pid(int(parts[2]), group=False)
                self._send(200, {"ok": True})
                return
            self._send(404, {"error": "saknas"})
        except PanelError as exc:
            self._send(400, {"error": str(exc)})

    def do_PUT(self) -> None:  # noqa: N802
        if not self._guard():
            return
        parts = [part for part in urlparse(self.path).path.split("/") if part]
        try:
            if len(parts) != 3 or parts[:2] != ["api", "slots"] or not ID_RE.fullmatch(parts[2]):
                self._send(404, {"error": "saknas"})
                return
            self._send(200, update_slot(parts[2], self._read_json()))
        except PanelError as exc:
            self._send(400, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._guard():
            return
        parts = [part for part in urlparse(self.path).path.split("/") if part]
        try:
            if len(parts) != 3 or parts[:2] != ["api", "slots"] or not ID_RE.fullmatch(parts[2]):
                self._send(404, {"error": "saknas"})
                return
            delete_slot(parts[2])
            self._send(200, {"ok": True})
        except PanelError as exc:
            self._send(400, {"error": str(exc)})


def main() -> None:
    port = int(os.environ.get("RAPID_MLX_PANEL_PORT", str(PANEL_PORT_DEFAULT)))
    os.environ["RAPID_MLX_PANEL_PORT"] = str(port)
    state_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, port), Handler)
    print(f"Rapid-MLX-panel på http://{HOST}:{port}", flush=True)
    print(f"Platser sparas i {slots_path()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStoppar panelen.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
