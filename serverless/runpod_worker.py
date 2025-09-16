# Copyright (c) 2025 Boson AI
"""RunPod Serverless worker entrypoint for Higgs Audio."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import runpod
import torch
import torchaudio

from boson_multimodal.data_types import AudioContent, ChatMLSample, Message, TextContent
from boson_multimodal.serve.serve_engine import HiggsAudioResponse, HiggsAudioServeEngine


LOG_LEVEL = os.environ.get("HIGGS_AUDIO_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
LOGGER = logging.getLogger("higgs_audio.runpod_worker")
LOGGER.setLevel(LOG_LEVEL)

DEFAULT_MODEL_ID = os.environ.get("HIGGS_AUDIO_MODEL_ID", "bosonai/higgs-audio-v2-generation-3B-base")
DEFAULT_AUDIO_TOKENIZER_ID = os.environ.get("HIGGS_AUDIO_AUDIO_TOKENIZER_ID", "bosonai/higgs-audio-v2-tokenizer")
DEFAULT_TOKENIZER_ID = os.environ.get("HIGGS_AUDIO_TOKENIZER_ID")
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "HIGGS_AUDIO_SYSTEM_PROMPT",
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>",
)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("HIGGS_AUDIO_MAX_NEW_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.environ.get("HIGGS_AUDIO_TEMPERATURE", "0.3"))
DEFAULT_TOP_P = float(os.environ.get("HIGGS_AUDIO_TOP_P", "0.95"))
_DEFAULT_TOP_K_ENV = os.environ.get("HIGGS_AUDIO_TOP_K", "50")
DEFAULT_TOP_K_VALUE = int(_DEFAULT_TOP_K_ENV) if _DEFAULT_TOP_K_ENV not in {"", None} else None
DEFAULT_RAS_WIN_LEN = os.environ.get("HIGGS_AUDIO_RAS_WIN_LEN", "7")
DEFAULT_RAS_WIN_MAX_REPEAT = int(os.environ.get("HIGGS_AUDIO_RAS_WIN_MAX_REPEAT", "2"))
DEFAULT_OUTPUT_FORMAT = os.environ.get("HIGGS_AUDIO_RESPONSE_FORMAT", "wav")
DEFAULT_DOWNLOAD_TIMEOUT = float(os.environ.get("HIGGS_AUDIO_DOWNLOAD_TIMEOUT", "30"))

ENGINE: HiggsAudioServeEngine | None = None
MODEL_LOCK = threading.Lock()


@dataclass
class ReferenceAudio:
    """Container describing a reference audio clip."""

    path: str
    transcript: str | None = None
    speaker_tag: str | None = None


def _ensure_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_reference_config(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_url": raw.get("audio_url") or raw.get("url"),
        "audio_base64": raw.get("audio_base64")
        or raw.get("base64")
        or raw.get("raw_audio"),
        "path": raw.get("path") or raw.get("audio_path"),
        "transcript": raw.get("transcript") or raw.get("text") or raw.get("prompt"),
        "speaker_tag": raw.get("speaker_tag") or raw.get("speaker"),
        "file_extension": raw.get("file_extension") or raw.get("extension"),
    }


def _parse_dtype(value: str | None):
    if value is None:
        return "auto"
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping.get(normalized)
    if dtype is None:
        LOGGER.warning("Unrecognized dtype '%s'. Falling back to auto.", value)
        return "auto"
    return dtype


def _get_engine() -> HiggsAudioServeEngine:
    global ENGINE
    if ENGINE is not None:
        return ENGINE

    with MODEL_LOCK:
        if ENGINE is None:
            device = os.environ.get("HIGGS_AUDIO_DEVICE")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = _parse_dtype(os.environ.get("HIGGS_AUDIO_TORCH_DTYPE"))

            LOGGER.info("Loading Higgs Audio model '%s' on device '%s'", DEFAULT_MODEL_ID, device)
            ENGINE = HiggsAudioServeEngine(
                model_name_or_path=DEFAULT_MODEL_ID,
                audio_tokenizer_name_or_path=DEFAULT_AUDIO_TOKENIZER_ID,
                tokenizer_name_or_path=DEFAULT_TOKENIZER_ID or DEFAULT_MODEL_ID,
                device=device,
                torch_dtype=torch_dtype,
            )
    return ENGINE


def _format_speaker_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    stripped = tag.strip()
    if not stripped:
        return None
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    return f"[{stripped}]"


def _decode_base64_audio(data: str) -> bytes:
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 audio payload.") from exc


def _download_audio(url: str, destination_dir: str, index: int, timeout: float) -> Path:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    suffix = Path(urlparse(url).path).suffix or ".bin"
    file_path = Path(destination_dir) / f"reference_{index}{suffix}"

    with open(file_path, "wb") as file_obj:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_obj.write(chunk)
    return file_path


def _prepare_reference_audios(job_input: dict[str, Any], temp_dir: str, timeout: float) -> list[ReferenceAudio]:
    references_cfg: list[dict[str, Any]] = []

    if "references" in job_input:
        raw_refs = job_input["references"]
        if isinstance(raw_refs, dict):
            references_cfg = [_normalize_reference_config(raw_refs)]
        else:
            references_cfg = [_normalize_reference_config(ref) for ref in raw_refs]
    else:
        urls = _ensure_iterable(job_input.get("reference_audio_urls"))
        if not urls:
            urls = _ensure_iterable(job_input.get("reference_audio_url"))
        if not urls:
            urls = _ensure_iterable(job_input.get("reference_audio"))

        base64_items = _ensure_iterable(
            job_input.get("reference_audio_base64") or job_input.get("reference_audio_b64")
        )
        paths = _ensure_iterable(job_input.get("reference_audio_paths") or job_input.get("reference_audio_path"))
        transcripts = _ensure_iterable(job_input.get("reference_transcripts") or job_input.get("reference_transcript"))
        speaker_tags = _ensure_iterable(job_input.get("reference_speaker_tags") or job_input.get("reference_speaker_tag"))
        file_extensions = _ensure_iterable(job_input.get("reference_file_extensions") or job_input.get("reference_file_extension"))

        max_len = max(
            len(urls),
            len(base64_items),
            len(paths),
            len(transcripts),
            len(speaker_tags),
            len(file_extensions),
            default=0,
        )

        for idx in range(max_len or 1):
            cfg = {
                "audio_url": urls[idx] if idx < len(urls) else None,
                "audio_base64": base64_items[idx] if idx < len(base64_items) else None,
                "path": paths[idx] if idx < len(paths) else None,
                "transcript": transcripts[idx] if idx < len(transcripts) else None,
                "speaker_tag": speaker_tags[idx] if idx < len(speaker_tags) else None,
                "file_extension": file_extensions[idx] if idx < len(file_extensions) else None,
            }
            if any([cfg["audio_url"], cfg["audio_base64"], cfg["path"]]):
                references_cfg.append(_normalize_reference_config(cfg))

    references: list[ReferenceAudio] = []
    for idx, cfg in enumerate(references_cfg):
        file_path: Path | None = None
        if cfg.get("audio_url"):
            try:
                file_path = _download_audio(cfg["audio_url"], temp_dir, idx, timeout)
            except requests.RequestException as exc:
                raise ValueError(f"Failed to download reference audio from '{cfg['audio_url']}': {exc}") from exc
        elif cfg.get("audio_base64"):
            suffix = cfg.get("file_extension") or "wav"
            if not str(suffix).startswith("."):
                suffix = f".{suffix}"
            file_path = Path(temp_dir) / f"reference_{idx}{suffix}"
            audio_bytes = _decode_base64_audio(cfg["audio_base64"])
            file_path.write_bytes(audio_bytes)
        elif cfg.get("path"):
            src = Path(str(cfg["path"]))
            if not src.exists():
                raise ValueError(f"Reference audio path '{src}' does not exist.")
            file_path = Path(temp_dir) / f"reference_{idx}{src.suffix or '.wav'}"
            shutil.copyfile(src, file_path)
        else:
            continue

        references.append(
            ReferenceAudio(
                path=str(file_path),
                transcript=cfg.get("transcript"),
                speaker_tag=cfg.get("speaker_tag"),
            )
        )

    return references


def _build_system_prompt(job_input: dict[str, Any]) -> str | None:
    system_prompt = job_input.get("system_prompt")
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    if system_prompt is not None:
        system_prompt = str(system_prompt)
    scene_prompt = job_input.get("scene_prompt")
    if scene_prompt:
        scene_prompt = str(scene_prompt)
        if "<|scene_desc_start|>" not in system_prompt:
            system_prompt = f"{system_prompt}\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
    return system_prompt


def _reference_messages(references: list[ReferenceAudio]) -> list[Message]:
    messages: list[Message] = []
    for idx, reference in enumerate(references):
        prefix = _format_speaker_tag(reference.speaker_tag)
        if prefix:
            transcript = reference.transcript or f"Reference sample {idx + 1}."
            prompt_text = f"{prefix} {transcript}".strip()
        else:
            prompt_text = reference.transcript or f"Reference sample {idx + 1}."
        messages.append(Message(role="user", content=prompt_text))
        messages.append(Message(role="assistant", content=AudioContent(audio_url=reference.path)))
    return messages


def _parse_messages(messages_payload: Sequence[dict[str, Any]]) -> list[Message]:
    parsed: list[Message] = []
    for idx, entry in enumerate(messages_payload):
        if "role" not in entry:
            raise ValueError(f"Message at index {idx} is missing 'role'.")
        role = entry["role"]
        content = entry.get("content", "")
        parsed.append(Message(role=role, content=_parse_message_content(content, idx)))
    return parsed


def _parse_message_content(content: Any, message_index: int):
    if isinstance(content, list):
        parts: list[Any] = []
        for idx, part in enumerate(content):
            parts.append(_parse_content_part(part, message_index, idx))
        return parts
    if isinstance(content, dict):
        return _parse_content_part(content, message_index, 0)
    return str(content)


def _parse_content_part(part: Any, message_index: int, part_index: int):
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        raise ValueError(f"Unsupported content type in message {message_index} part {part_index}.")
    content_type = part.get("type", "text")
    if content_type == "text":
        text = part.get("text")
        if text is None:
            raise ValueError(f"Missing 'text' value for text part in message {message_index}.")
        return TextContent(text=str(text))
    raise ValueError(
        "Audio content within custom messages is not supported. "
        "Please use the 'references' field to provide reference audio clips."
    )


def _build_script_message(script: str, speaker_tag: str | None) -> Message:
    prefix = _format_speaker_tag(speaker_tag)
    if prefix:
        script = f"{prefix} {script}".strip()
    return Message(role="user", content=script)


def _parse_stop_strings(raw_stop: Any) -> list[str] | None:
    if raw_stop is None:
        return None
    if isinstance(raw_stop, str):
        return [raw_stop]
    if isinstance(raw_stop, Sequence):
        stop_list = []
        for item in raw_stop:
            stop_list.append(str(item))
        return stop_list
    raise ValueError("stop_strings must be a string or a sequence of strings.")


def _extract_generation_kwargs(job_input: dict[str, Any]) -> dict[str, Any]:
    max_new_tokens = int(job_input.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    temperature = float(job_input.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(job_input.get("top_p", DEFAULT_TOP_P))
    top_k_value = job_input.get("top_k", DEFAULT_TOP_K_VALUE)
    top_k = int(top_k_value) if top_k_value not in {None, ""} else None

    ras_win_len_raw = job_input.get("ras_win_len", DEFAULT_RAS_WIN_LEN)
    ras_win_len = None
    if ras_win_len_raw not in {None, ""}:
        ras_win_len = int(ras_win_len_raw)
    if ras_win_len is not None and ras_win_len <= 0:
        ras_win_len = None

    ras_win_max = int(job_input.get("ras_win_max_num_repeat", DEFAULT_RAS_WIN_MAX_REPEAT))
    seed = job_input.get("seed")
    if seed not in {None, ""}:
        seed = int(seed)
    else:
        seed = None

    stop_strings = _parse_stop_strings(job_input.get("stop_strings"))
    force_audio_gen = job_input.get("force_audio_gen")
    if force_audio_gen is None:
        force_audio_gen = True

    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_strings": stop_strings,
        "force_audio_gen": force_audio_gen,
        "ras_win_len": ras_win_len,
        "ras_win_max_num_repeat": ras_win_max,
        "seed": seed,
    }


def _encode_audio_to_base64(audio: Any, sampling_rate: int, fmt: str) -> dict[str, str]:
    format_normalized = fmt.lower()
    if format_normalized not in {"wav", "flac"}:
        raise ValueError("Only 'wav' and 'flac' formats are supported for response audio.")

    buffer = io.BytesIO()
    tensor = torch.from_numpy(audio).unsqueeze(0)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    torchaudio.save(buffer, tensor, sampling_rate, format=format_normalized)
    audio_bytes = buffer.getvalue()
    mime_type = "audio/wav" if format_normalized == "wav" else "audio/flac"
    return {
        "base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "mime_type": mime_type,
        "format": format_normalized,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {})
    timeout = float(job_input.get("download_timeout", DEFAULT_DOWNLOAD_TIMEOUT))
    script = job_input.get("script") or job_input.get("prompt") or job_input.get("transcript")
    if isinstance(script, str):
        script = script.strip()
    if script == "":
        script = None

    messages_payload = job_input.get("messages")

    if script is None and not messages_payload:
        return {"error": "Request must include a 'script'/'prompt' or pre-built 'messages'."}

    try:
        with tempfile.TemporaryDirectory(prefix="higgs_audio_refs_") as temp_dir:
            references = _prepare_reference_audios(job_input, temp_dir, timeout)

            messages: list[Message] = []
            if messages_payload:
                if not isinstance(messages_payload, Sequence):
                    return {"error": "'messages' must be an array of message objects."}
                messages.extend(_parse_messages(messages_payload))
            else:
                system_prompt = _build_system_prompt(job_input)
                if system_prompt:
                    messages.append(Message(role="system", content=system_prompt))

            messages.extend(_reference_messages(references))

            if script:
                messages.append(_build_script_message(script, job_input.get("script_speaker_tag")))

            if not messages:
                return {"error": "No messages available to send to the model."}

            chat_sample = ChatMLSample(messages=messages)
            engine = _get_engine()
            generation_kwargs = _extract_generation_kwargs(job_input)

            with MODEL_LOCK:
                response: HiggsAudioResponse = engine.generate(
                    chat_ml_sample=chat_sample,
                    **generation_kwargs,
                )

            if response.audio is None:
                return {"error": "Model did not return audio tokens."}

            audio_format = job_input.get("response_format", DEFAULT_OUTPUT_FORMAT)
            audio_payload = _encode_audio_to_base64(response.audio, response.sampling_rate, audio_format)

            result: dict[str, Any] = {
                "audio_base64": audio_payload["base64"],
                "audio_format": audio_payload["format"],
                "mime_type": audio_payload["mime_type"],
                "sampling_rate": response.sampling_rate,
                "generated_text": response.generated_text.strip() if response.generated_text else "",
                "usage": response.usage,
                "model": engine.model_name_or_path,
            }

            if job_input.get("return_audio_tokens"):
                result["generated_audio_tokens"] = response.generated_audio_tokens.tolist()
            if job_input.get("return_text_tokens"):
                result["generated_text_tokens"] = response.generated_text_tokens.tolist()
            if script:
                result["prompt"] = script
            if references:
                result["reference_transcripts"] = [ref.transcript for ref in references]

            return result
    except ValueError as exc:
        LOGGER.error("Invalid request: %s", exc)
        return {"error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unhandled exception during inference.")
        return {"error": f"Unhandled exception: {exc}"}


runpod.serverless.start({"handler": handler})
