# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import logging
import os
import platform
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from ._version import __version__
from .generate import stream_generate
from .models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from .sample_utils import make_logits_processors, make_sampler
from .utils import common_prefix_len, load

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    from .instrumentation import (
        ErrorCategory,
        categorize_error,
        get_logger,
        get_metrics,
        get_tracer,
        init_telemetry,
        log_with_trace,
        timer,
        trace_method,
        trace_span,
    )

    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

    # Provide no-op implementations
    def init_telemetry(*args, **kwargs):
        pass

    def get_logger():
        return logging.getLogger(__name__)

    def get_metrics():
        return None

    def get_tracer():
        return None

    def log_with_trace(level, msg, **kwargs):
        logging.log(getattr(logging, level.upper()), msg)

    def timer(histogram, attributes=None):
        from contextlib import contextmanager

        @contextmanager
        def noop():
            yield

        return noop()

    def trace_method(span_name=None, attributes=None, record_exception=True):
        def decorator(func):
            return func

        return decorator

    def trace_span(name, attributes=None, kind=None):
        from contextlib import contextmanager

        @contextmanager
        def noop():
            yield None

        return noop()

    def categorize_error(e):
        return "unknown_error"

    class Status:
        def __init__(self, code, description=""):
            pass

    class StatusCode:
        OK = 0
        ERROR = 1

    trace = type(
        "trace",
        (),
        {
            "get_current_span": lambda: type(
                "span",
                (),
                {
                    "set_attributes": lambda self, attrs: None,
                    "set_attribute": lambda self, k, v: None,
                    "record_exception": lambda self, e: None,
                    "set_status": lambda self, s: None,
                    "is_recording": lambda self: False,
                },
            )()
        },
    )()
    TELEMETRY_AVAILABLE = False


def get_system_fingerprint():
    gpu_arch = mx.metal.device_info()["architecture"] if mx.metal.is_available() else ""
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    eos_token_id: Union[int, None],
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        eos_token_id (Union[int, None]): The token ID that represents the
          end-of-sequence. If the last token in `tokens` matches this, the
          generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`).
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def process_message_content(messages):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    The function operates on messages in place. It converts the 'content' field
    to a string instead of a list of text fragments.

    Args:
        message_list (list): A list of dictionaries, where each dictionary may
          have a 'content' key containing a list of dictionaries with 'type' and
          'text' keys.

    Raises:
        ValueError: If the 'content' type is not supported or if 'text' is missing.

    """
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""


@dataclass
class PromptCache:
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None, None)
    tokens: List[int] = field(default_factory=list)


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.logger = get_logger()
        self.metrics = get_metrics() if TELEMETRY_AVAILABLE else None

        # Preload the default model if it is provided
        if self.cli_args.model is not None:
            with trace_span(
                "model_provider.init", attributes={"model": self.cli_args.model}
            ):
                self.load("default_model", draft_model_path="default_model")

    def _validate_model_path(self, model_path: str):
        model_path = Path(model_path)
        if model_path.exists() and not model_path.is_relative_to(Path.cwd()):
            raise RuntimeError(
                "Local models must be relative to the current working dir."
            )

    # Added in adapter_path to load dynamically
    @trace_method(span_name="model_provider.load")
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        span = trace.get_current_span()
        span.set_attributes(
            {
                "model.path": str(model_path),
                "model.adapter_path": str(adapter_path) if adapter_path else "none",
                "model.draft_path": (
                    str(draft_model_path) if draft_model_path else "none"
                ),
            }
        )

        if self.model_key == (model_path, adapter_path, draft_model_path):
            if self.metrics:
                self.metrics.cache_hits.add(1, {"cache_type": "model"})
            log_with_trace("info", "Model cache hit", model_key=str(self.model_key))
            return self.model, self.tokenizer

        if self.metrics:
            self.metrics.cache_misses.add(1, {"cache_type": "model"})
        log_with_trace("info", "Loading new model", model_path=str(model_path))

        # Remove the old model if it exists.
        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        # Building tokenizer_config
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        try:
            with timer(
                self.metrics.model_load_duration if self.metrics else None,
                {"model": str(model_path)},
            ):
                if model_path == "default_model":
                    if self.cli_args.model is None:
                        raise ValueError(
                            "A model path has to be given as a CLI "
                            "argument or in the HTTP request"
                        )
                    model, tokenizer = load(
                        self.cli_args.model,
                        adapter_path=(
                            adapter_path if adapter_path else self.cli_args.adapter_path
                        ),  # if the user doesn't change the model but adds an adapter path
                        tokenizer_config=tokenizer_config,
                    )
                else:
                    self._validate_model_path(model_path)
                    model, tokenizer = load(
                        model_path,
                        adapter_path=adapter_path,
                        tokenizer_config=tokenizer_config,
                    )
        except Exception as e:
            error_type = categorize_error(e)
            if self.metrics:
                self.metrics.errors.add(
                    1, {"error_type": error_type, "operation": "model_load"}
                )
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            log_with_trace(
                "error",
                "Failed to load model",
                model_path=str(model_path),
                error=str(e),
                error_type=error_type,
            )
            raise

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer

        # Log successful model load
        log_with_trace(
            "info",
            "Model loaded successfully",
            model_path=str(model_path),
            tokenizer_vocab_size=tokenizer.vocab_size,
        )

        def validate_draft_tokenizer(draft_tokenizer):
            # Check if tokenizers are compatible
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. "
                    "Speculative decoding may not work as expected."
                )

        # Load draft model if specified
        if (
            draft_model_path == "default_model"
            and self.cli_args.draft_model is not None
        ):
            self.draft_model, draft_tokenizer = load(self.cli_args.draft_model)
            validate_draft_tokenizer(draft_tokenizer)

        elif draft_model_path is not None and draft_model_path != "default_model":
            self._validate_model_path(draft_model_path)
            self.draft_model, draft_tokenizer = load(draft_model_path)
            validate_draft_tokenizer(draft_tokenizer)
        return self.model, self.tokenizer


class APIHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        prompt_cache: Optional[PromptCache] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache or PromptCache()
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        self.logger = get_logger()
        self.metrics = get_metrics() if TELEMETRY_AVAILABLE else None
        self.tracer = get_tracer() if TELEMETRY_AVAILABLE else None
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        span_context = (
            self.tracer.start_as_current_span(
                "http.request",
                kind=(
                    trace.SpanKind.SERVER if hasattr(trace.SpanKind, "SERVER") else None
                ),
                attributes={
                    "http.method": "POST",
                    "http.url": self.path,
                    "http.scheme": "http",
                    "http.host": self.headers.get("Host", "unknown"),
                    "http.user_agent": self.headers.get("User-Agent", "unknown"),
                },
            )
            if self.tracer
            else trace_span("http.request")
        )

        with span_context as span:
            if self.metrics:
                self.metrics.active_requests.add(1)
            request_start = time.perf_counter()

            try:
                endpoints = {
                    "/v1/completions": self.handle_text_completions,
                    "/v1/chat/completions": self.handle_chat_completions,
                    "/chat/completions": self.handle_chat_completions,
                }

                if self.path not in endpoints:
                    if self.metrics:
                        self.metrics.request_counter.add(
                            1,
                            {"method": "POST", "endpoint": self.path, "status": "404"},
                        )
                    if span:
                        span.set_attribute("http.status_code", 404)
                    self._set_completion_headers(404)
                    self.end_headers()
                    self.wfile.write(b"Not Found")
                    return

                # Fetch and parse request body
                content_length = int(self.headers["Content-Length"])
                raw_body = self.rfile.read(content_length)
                if self.metrics:
                    self.metrics.request_size.record(
                        content_length, {"endpoint": self.path}
                    )

                self.body = json.loads(raw_body.decode())
                indent = "\t"  # Backslashes can't be inside of f-strings
                log_with_trace(
                    "debug",
                    "Incoming request",
                    endpoint=self.path,
                    content_length=content_length,
                    body_preview=str(self.body)[:200],
                )

                assert isinstance(
                    self.body, dict
                ), f"Request should be dict, but got {type(self.body)}"

                # Extract request parameters from the body
                self.stream = self.body.get("stream", False)
                self.stream_options = self.body.get("stream_options", None)
                self.requested_model = self.body.get("model", "default_model")
                self.requested_draft_model = self.body.get(
                    "draft_model", "default_model"
                )
                self.num_draft_tokens = self.body.get(
                    "num_draft_tokens", self.model_provider.cli_args.num_draft_tokens
                )
                self.adapter = self.body.get("adapters", None)
                self.max_tokens = self.body.get("max_completion_tokens", None)
                if self.max_tokens is None:
                    self.max_tokens = self.body.get(
                        "max_tokens", self.model_provider.cli_args.max_tokens
                    )
                self.temperature = self.body.get(
                    "temperature", self.model_provider.cli_args.temp
                )
                self.top_p = self.body.get("top_p", self.model_provider.cli_args.top_p)
                self.top_k = self.body.get("top_k", self.model_provider.cli_args.top_k)
                self.min_p = self.body.get("min_p", self.model_provider.cli_args.min_p)
                self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
                self.repetition_context_size = self.body.get(
                    "repetition_context_size", 20
                )
                self.xtc_probability = self.body.get("xtc_probability", 0.0)
                self.xtc_threshold = self.body.get("xtc_threshold", 0.0)
                self.logit_bias = self.body.get("logit_bias", None)
                self.logprobs = self.body.get("logprobs", -1)

                # Add request attributes to span
                if span:
                    span.set_attributes(
                        {
                            "mlx.model": self.requested_model,
                            "mlx.stream": self.stream,
                            "mlx.max_tokens": self.max_tokens,
                            "mlx.temperature": self.temperature,
                        }
                    )

                self.validate_model_parameters()

                # Load the model if needed
                try:
                    with trace_span(
                        "load_model", {"model": self.requested_model}
                    ) as load_span:
                        self.model, self.tokenizer = self.model_provider.load(
                            self.requested_model,
                            self.adapter,
                            self.requested_draft_model,
                        )
                except Exception as e:
                    error_type = categorize_error(e)
                    if self.metrics:
                        self.metrics.errors.add(
                            1,
                            {
                                "error_type": error_type,
                                "endpoint": self.path,
                                "operation": "model_load",
                            },
                        )
                    if span:
                        span.record_exception(e)
                        span.set_attribute("http.status_code", 404)
                    log_with_trace(
                        "error",
                        "Model load failed",
                        model=self.requested_model,
                        error=str(e),
                        error_type=error_type,
                    )
                    self._set_completion_headers(404)
                    self.end_headers()
                    self.wfile.write(b"Not Found")
                    return

                # Get stop id sequences, if provided
                stop_words = self.body.get("stop")
                stop_words = stop_words or []
                stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
                stop_id_sequences = [
                    self.tokenizer.encode(stop_word, add_special_tokens=False)
                    for stop_word in stop_words
                ]

                # Send header type
                (
                    self._set_stream_headers(200)
                    if self.stream
                    else self._set_completion_headers(200)
                )
                if span:
                    span.set_attribute("http.status_code", 200)

                # Call endpoint specific method
                with trace_span(
                    "prepare_prompt", {"endpoint": self.path}
                ) as prompt_span:
                    prompt = endpoints[self.path]()
                    if self.metrics:
                        self.metrics.prompt_tokens.record(
                            len(prompt),
                            {"endpoint": self.path, "model": self.requested_model},
                        )

                self.handle_completion(prompt, stop_id_sequences)

                # Record successful request
                request_duration = time.perf_counter() - request_start
                if self.metrics:
                    self.metrics.request_duration.record(
                        request_duration,
                        {
                            "method": "POST",
                            "endpoint": self.path,
                            "status": "200",
                            "model": self.requested_model,
                        },
                    )
                    self.metrics.request_counter.add(
                        1, {"method": "POST", "endpoint": self.path, "status": "200"}
                    )
                log_with_trace(
                    "info",
                    "Request completed",
                    endpoint=self.path,
                    duration_ms=request_duration * 1000,
                    model=self.requested_model,
                )

            except Exception as e:
                error_type = categorize_error(e)
                if self.metrics:
                    self.metrics.errors.add(
                        1,
                        {
                            "error_type": error_type,
                            "endpoint": self.path,
                            "operation": "request_handling",
                        },
                    )
                if span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                log_with_trace(
                    "error",
                    "Request failed",
                    endpoint=self.path,
                    error=str(e),
                    error_type=error_type,
                )
                raise
            finally:
                if self.metrics:
                    self.metrics.active_requests.add(-1)

    @trace_method(span_name="validate_parameters")
    def validate_model_parameters(self):
        """
        Validate the model parameters passed in the request for the correct types and values.
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")

        if not isinstance(self.min_p, (float, int)) or self.min_p < 0 or self.min_p > 1:
            raise ValueError("min_p must be a float between 0 and 1")

        if not isinstance(self.num_draft_tokens, int) or self.num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be a non-negative integer")

        if (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

        if self.logprobs != -1 and not (0 < self.logprobs <= 10):
            raise ValueError(
                f"logprobs must be between 1 and 10 but got {self.logprobs:,}"
            )

        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError("repetition_context_size must be a non-negative integer")

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")

            try:
                self.logit_bias = {int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")
        if not (
            isinstance(self.xtc_probability, float)
            and 0.00 <= self.xtc_probability <= 1.00
        ):
            raise ValueError(f"xtc_probability must be a float between 0.00 and 1.00")
        if not (
            isinstance(self.xtc_threshold, float) and 0.00 <= self.xtc_threshold <= 0.50
        ):
            raise ValueError(f"xtc_threshold must be a float between 0.00 and 0.5")
        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (Optional[int]): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (Optional[int]): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            token_logprobs (Optional[List[float]]): The log probabilities per token,
              in token order.
            top_tokens (Optional[List[Dict[int, float]]]): List of dictionaries mapping
              tokens to logprobs for the top N tokens at each token position.
            tokens (Optional[List[int]]): List of tokens to return with logprobs structure

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs if token_logprobs else []
        top_logprobs = top_tokens if top_tokens else []

        # Static response
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs,
                        "tokens": tokens,
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {"role": "assistant", "content": text}
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            ValueError(f"Unsupported response type: {self.object_type}")

        return response

    @trace_method(span_name="reset_prompt_cache")
    def reset_prompt_cache(self, prompt):
        """Resets the prompt cache and associated state.

        Args:
            prompt (List[int]): The tokenized new prompt which will populate the
                reset cache.
        """
        log_with_trace("debug", "Resetting prompt cache", prompt_length=len(prompt))
        self.prompt_cache.model_key = self.model_provider.model_key
        self.prompt_cache.cache = make_prompt_cache(self.model_provider.model)
        if self.model_provider.draft_model is not None:
            self.prompt_cache.cache += make_prompt_cache(
                self.model_provider.draft_model
            )
        self.prompt_cache.tokens = list(prompt)  # Cache the new prompt fully

    @trace_method(span_name="get_prompt_cache")
    def get_prompt_cache(self, prompt):
        """
        Determines the portion of the prompt that needs processing by comparing
        it to the cached prompt and attempting to reuse the common prefix.

        This function updates the internal prompt cache state (tokens and model cache)
        based on the comparison. If a common prefix exists, it attempts to trim
        the model cache (if supported) to match the common prefix length, avoiding
        recomputation.

        Args:
            prompt (List[int]): The tokenized new prompt.

        Returns:
            List[int]: The suffix of the prompt that actually needs to be processed
                       by the model. This will be the full prompt if the cache is
                       reset or cannot be effectively used.
        """
        span = trace.get_current_span()
        span.set_attribute("prompt.length", len(prompt))
        cache_len = len(self.prompt_cache.tokens)
        prompt_len = len(prompt)
        com_prefix_len = common_prefix_len(self.prompt_cache.tokens, prompt)

        # Leave at least one token in the prompt
        com_prefix_len = min(com_prefix_len, len(prompt) - 1)

        # Condition 1: Model changed or no common prefix at all. Reset cache.
        if (
            self.prompt_cache.model_key != self.model_provider.model_key
            or com_prefix_len == 0
        ):
            if self.metrics:
                self.metrics.cache_misses.add(
                    1, {"cache_type": "prompt", "reason": "no_prefix"}
                )
            span.set_attribute("cache.hit", False)
            span.set_attribute("cache.reason", "no_prefix")
            self.reset_prompt_cache(prompt)

        # Condition 2: Common prefix exists and matches cache length. Process suffix.
        elif com_prefix_len == cache_len:
            if self.metrics:
                self.metrics.cache_hits.add(1, {"cache_type": "prompt"})
            span.set_attribute("cache.hit", True)
            span.set_attribute("cache.common_prefix_len", com_prefix_len)
            log_with_trace(
                "debug",
                "Cache hit - processing suffix",
                cache_len=cache_len,
                prompt_len=prompt_len,
                suffix_len=prompt_len - com_prefix_len,
            )
            prompt = prompt[com_prefix_len:]
            self.prompt_cache.tokens.extend(prompt)

        # Condition 3: Common prefix exists but is shorter than cache length. Attempt trim.
        elif com_prefix_len < cache_len:
            log_with_trace(
                "debug",
                "Common prefix shorter than cache",
                common_prefix_len=com_prefix_len,
                cache_len=cache_len,
            )

            if can_trim_prompt_cache(self.prompt_cache.cache):
                num_to_trim = cache_len - com_prefix_len
                if self.metrics:
                    self.metrics.cache_hits.add(
                        1, {"cache_type": "prompt", "operation": "trim"}
                    )
                span.set_attribute("cache.hit", True)
                span.set_attribute("cache.trim_tokens", num_to_trim)
                log_with_trace("debug", "Trimming cache", tokens_to_trim=num_to_trim)
                trim_prompt_cache(self.prompt_cache.cache, num_to_trim)
                self.prompt_cache.tokens = self.prompt_cache.tokens[:com_prefix_len]
                prompt = prompt[com_prefix_len:]
                self.prompt_cache.tokens.extend(prompt)
            else:
                if self.metrics:
                    self.metrics.cache_misses.add(
                        1, {"cache_type": "prompt", "reason": "cannot_trim"}
                    )
                span.set_attribute("cache.hit", False)
                span.set_attribute("cache.reason", "cannot_trim")
                log_with_trace("debug", "Cache cannot be trimmed - resetting")
                self.reset_prompt_cache(prompt)

        # This case should logically not be reached if com_prefix_len <= cache_len
        else:
            log_with_trace(
                "error",
                "Unexpected cache state",
                common_prefix_len=com_prefix_len,
                cache_len=cache_len,
            )
            self.reset_prompt_cache(prompt)

        span.set_attribute("prompt.tokens_to_process", len(prompt))
        log_with_trace("debug", "Prompt cache processed", tokens_to_process=len(prompt))
        return prompt

    @trace_method(span_name="handle_completion")
    def handle_completion(
        self,
        prompt: List[int],
        stop_id_sequences: List[List[int]],
    ):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (List[int]): The tokenized prompt.
            stop_id_sequences (List[List[int]]): A list of stop words passed
              to the stopping_criteria function
        """
        span = trace.get_current_span()
        span.set_attributes(
            {
                "prompt.length": len(prompt),
                "stop_sequences.count": len(stop_id_sequences),
                "stream": self.stream,
            }
        )
        tokens = []
        finish_reason = "length"
        stop_sequence_suffix = None
        if self.stream:
            self.end_headers()
            log_with_trace("debug", "Starting stream generation")
        else:
            log_with_trace("debug", "Starting completion generation")
        token_logprobs = []
        top_tokens = []

        prompt = self.get_prompt_cache(prompt)

        text = ""
        tic = time.perf_counter()
        sampler = make_sampler(
            self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            xtc_probability=self.xtc_probability,
            xtc_threshold=self.xtc_threshold,
            xtc_special_tokens=[
                self.tokenizer.eos_token_id,
                self.tokenizer.encode("\n"),
            ],
        )
        logits_processors = make_logits_processors(
            self.logit_bias,
            self.repetition_penalty,
            self.repetition_context_size,
        )

        inference_timer = (
            timer(
                self.metrics.inference_duration,
                {"model": self.requested_model, "stream": self.stream},
            )
            if self.metrics
            else trace_span("inference")
        )

        with inference_timer:
            for gen_response in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=self.prompt_cache.cache,
                draft_model=self.model_provider.draft_model,
                num_draft_tokens=self.num_draft_tokens,
            ):
                segment = gen_response.text
                text += segment
                token = gen_response.token
                logprobs = gen_response.logprobs
                tokens.append(token)

            if self.logprobs > 0:
                sorted_indices = mx.argpartition(-logprobs, kth=self.logprobs - 1)
                top_indices = sorted_indices[: self.logprobs]
                top_logprobs = logprobs[top_indices]
                top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(tuple(top_token_info))

            token_logprobs.append(logprobs[token].item())

            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )
            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                    text = text[: -len(stop_sequence_suffix)]
                break

            if self.stream:
                # If the end of tokens overlaps with a stop sequence, generate new
                # tokens until we know if the stop sequence is hit or not
                if any(
                    (
                        sequence_overlap(tokens, sequence)
                        for sequence in stop_id_sequences
                    )
                ):
                    continue
                elif segment:
                    response = self.generate_response(segment, None)
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()

        self.prompt_cache.tokens.extend(tokens)

        # Record generation metrics
        if self.metrics:
            self.metrics.tokens_generated.record(
                len(tokens),
                {"model": self.requested_model, "finish_reason": finish_reason},
            )
            self.metrics.tokens_per_second.record(
                gen_response.generation_tps, {"model": self.requested_model}
            )
            self.metrics.peak_memory.record(
                gen_response.peak_memory * 1e9,
                {"model": self.requested_model},  # Convert GB to bytes
            )

        span.set_attributes(
            {
                "tokens.generated": len(tokens),
                "tokens.prompt_tps": gen_response.prompt_tps,
                "tokens.generation_tps": gen_response.generation_tps,
                "memory.peak_gb": gen_response.peak_memory,
                "finish_reason": finish_reason,
            }
        )

        log_with_trace(
            "info",
            "Generation completed",
            prompt_tps=gen_response.prompt_tps,
            generation_tps=gen_response.generation_tps,
            peak_memory_gb=gen_response.peak_memory,
            tokens_generated=len(tokens),
            finish_reason=finish_reason,
        )

        if self.stream:
            response = self.generate_response(segment, finish_reason)
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            if self.stream_options is not None and self.stream_options["include_usage"]:
                response = self.completion_usage_response(len(prompt), len(tokens))
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            response = self.generate_response(
                text,
                finish_reason,
                len(prompt),
                len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
            )
            response_json = json.dumps(response).encode()
            if self.metrics:
                self.metrics.response_size.record(
                    len(response_json),
                    {"endpoint": self.path, "model": self.requested_model},
                )

            log_with_trace(
                "debug",
                "Sending completion response",
                response_size=len(response_json),
                prompt_tokens=len(prompt),
                completion_tokens=len(tokens),
            )

            # Send an additional Content-Length header when it is known
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ):
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        return response

    @trace_method(span_name="handle_chat_completions")
    def handle_chat_completions(self) -> List[int]:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"
        if self.tokenizer.chat_template:
            messages = body["messages"]
            process_message_content(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                body.get("tools") or None,
                add_generation_prompt=True,
                **self.model_provider.cli_args.chat_template_args,
            )
        else:
            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        return prompt

    @trace_method(span_name="handle_text_completions")
    def handle_text_completions(self) -> List[int]:
        """
        Handle a text completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return self.tokenizer.encode(self.body["prompt"])

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        span_context = (
            self.tracer.start_as_current_span(
                "http.request",
                kind=(
                    trace.SpanKind.SERVER if hasattr(trace.SpanKind, "SERVER") else None
                ),
                attributes={
                    "http.method": "GET",
                    "http.url": self.path,
                    "http.scheme": "http",
                },
            )
            if self.tracer
            else trace_span("http.request")
        )

        with span_context as span:
            try:
                if self.path == "/v1/models":
                    self.handle_models_request()
                    if span:
                        span.set_attribute("http.status_code", 200)
                elif self.path == "/health":
                    self.handle_health_check()
                    if span:
                        span.set_attribute("http.status_code", 200)
                else:
                    if span:
                        span.set_attribute("http.status_code", 404)
                    self._set_completion_headers(404)
                    self.end_headers()
                    self.wfile.write(b"Not Found")

                if self.metrics:
                    self.metrics.request_counter.add(
                        1,
                        {
                            "method": "GET",
                            "endpoint": self.path,
                            "status": str(
                                span.attributes.get("http.status_code", "unknown")
                                if span
                                else "200"
                            ),
                        },
                    )
            except Exception as e:
                if span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def handle_health_check(self):
        """
        Handle a GET request for the /health endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        self.wfile.write('{"status": "ok"}'.encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        def probably_mlx_lm(repo):
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    logger = get_logger()

    with trace_span("server.startup", attributes={"host": host, "port": port}) as span:
        server_address = (host, port)
        prompt_cache = PromptCache()
        infos = socket.getaddrinfo(
            *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
        server_class.address_family, _, _, _, server_address = next(iter(infos))
        httpd = server_class(
            server_address,
            lambda *args, **kwargs: handler_class(
                model_provider,
                prompt_cache=prompt_cache,
                system_fingerprint=get_system_fingerprint(),
                *args,
                **kwargs,
            ),
        )
        warnings.warn(
            "mlx_lm.server is not recommended for production as "
            "it only implements basic security checks."
        )
        log_with_trace(
            "info",
            "MLX server started",
            host=host,
            port=port,
            system_fingerprint=get_system_fingerprint(),
        )

    httpd.serve_forever()


def main():
    # Initialize OpenTelemetry early
    if TELEMETRY_AVAILABLE:
        init_telemetry(
            service_name=os.getenv("OTEL_SERVICE_NAME", "mlx-service"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
        )

    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    args = parser.parse_args()

    # Configure standard logging to work with structlog
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Suppress verbose logging from some libraries
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)

    with trace_span("main", attributes={"args": str(args)}) as span:
        run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.server...` directly is deprecated."
        " Use `mlx_lm.server...` or `python -m mlx_lm server ...` instead."
    )
    main()
