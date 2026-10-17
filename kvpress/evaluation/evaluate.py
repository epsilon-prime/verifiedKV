# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.dataset_loaders import (
    AIME_CONFIGS,
    RULER_CONFIG_SPLIT,
    RULER_RAW_SPLIT,
    load_aime_dataset,
    load_loft_dataset,
    load_ruler_length_dataset,
)
from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from benchmarks.scbench.utils import flatten_multi_turns
from benchmarks.scbench.utils import normalize_task as normalize_scbench_task
from datasets import load_dataset, load_from_disk
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import AutoTokenizer, FineGrainedFP8Config, Pipeline, pipeline

from kvpress import (
    BernoulliPress,
    ComposedPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    ScorerPress,
    ThinKPress,
    VKvWRPress,
)
from kvpress.model_loading import add_chat_eos, load_text_only_causal_lm, needs_text_only_load, suppress_think

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None
    head_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None

    # Dataset and generation parameters
    fraction: float = 1.0
    n_samples: Optional[int] = None
    dataset_path: Optional[str] = None
    max_new_tokens: Optional[int] = None
    min_context_length: Optional[int] = None
    max_context_length: Optional[int] = None
    query_aware: bool = False
    needle_depth: Optional[int] = None

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"
    enable_micro_metrics: bool = False
    micro_metrics_max_records: Optional[int] = None
    micro_metrics_sampling_factor: float = 1.0
    micro_metrics: Optional[str] = None  # comma-separated subset; default is all of them
    # Per-head upstream error statistics -> head_stats.parquet (see kvpress/metric_logging/head_stats.py)
    enable_head_stats: bool = False
    head_stats_layer_stride: int = 1
    head_stats_step_stride: int = 1
    head_stats_question_stride: int = 1
    head_stats_max_rows: Optional[int] = None

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None
    trust_remote_code: bool = False

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    # Quantization
    fp8: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"
        if self.n_samples is not None:
            assert self.n_samples > 0, f"n_samples must be positive, got {self.n_samples}"
        if self.min_context_length is not None:
            assert self.min_context_length > 0, f"min_context_length must be positive, got {self.min_context_length}"
            if self.max_context_length is not None:
                assert (
                    self.min_context_length <= self.max_context_length
                ), "min_context_length must be <= max_context_length"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

        if self.dataset in (*RULER_CONFIG_SPLIT, *RULER_RAW_SPLIT, "loft"):
            assert self.data_dir is not None, f"data_dir must be set for {self.dataset}"

        if self.dataset == "scbench":
            # fails here rather than after the model is loaded, and normalizing the task once keeps its
            # spelling out of the results directory name, so "vt" and "scbench_vt" resolve to one run
            self.data_dir = normalize_scbench_task(self.data_dir)

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path

        Returns
        -------
        Path
            The path to the results directory
        """
        # Build directory name components
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.compression_ratio:.2f}",
        ]

        if self.threshold is not None:
            components[-1] = f"{self.threshold:.2f}"
        elif self.head_compression_ratio is not None:
            components[-1] = f"{self.head_compression_ratio:.2f}"
        if self.dataset_path is not None:
            components.append(Path(self.dataset_path).name)
        if self.n_samples is not None:
            components.append(f"nsamples{self.n_samples}")
        elif self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.min_context_length is not None:
            components.append(f"min_context{self.min_context_length}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.query_aware:
            components.append("query_aware")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")

        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        config_dir = output_dir / dir_name

        # Make sure the directory does not exist, if it does, add a number to the end
        # This is to avoid overwriting results
        if config_dir.exists():
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        config_dict = asdict(self)
        if self.threshold is not None or self.head_compression_ratio is not None:
            config_dict.pop("compression_ratio", None)
        if self.threshold is None:
            config_dict.pop("threshold", None)
        if self.head_compression_ratio is None:
            config_dict.pop("head_compression_ratio", None)
        with open(str(config_filename), "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: None | ScorerPress = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

    def _configure_micro_metrics(self, results_dir: Path) -> None:
        """Optionally enable VKvWR sparsity / relative attention-error logging."""
        if not self.config.enable_micro_metrics:
            return
        from kvpress.metric_logging import DEFAULT_MICRO_METRICS, MicroMetricLogger

        if self.config.micro_metrics:
            enabled = [m.strip() for m in self.config.micro_metrics.split(",") if m.strip()]
            unknown = [m for m in enabled if m not in DEFAULT_MICRO_METRICS]
            assert not unknown, f"unknown micro metrics {unknown}, expected some of {DEFAULT_MICRO_METRICS}"
        else:
            enabled = list(DEFAULT_MICRO_METRICS)

        MicroMetricLogger().configure_logging(
            log_path=str(results_dir),
            enabled_metrics=enabled,
            max_records=self.config.micro_metrics_max_records,
            sampling_factor=self.config.micro_metrics_sampling_factor,
        )
        logger.info(
            "Micro-metric logging enabled (%s) → %s/micro_metrics.jsonl",
            ", ".join(enabled),
            results_dir,
        )

    def _configure_head_stats(self, results_dir: Path) -> None:
        """Optionally enable per-head denominator / numerator / attention error statistics."""
        if not self.config.enable_head_stats:
            return
        from kvpress.metric_logging import HeadStatsCollector

        HeadStatsCollector().configure(
            log_path=str(results_dir),
            run_config=self._head_stats_run_config(),
            layer_stride=self.config.head_stats_layer_stride,
            step_stride=self.config.head_stats_step_stride,
            question_stride=self.config.head_stats_question_stride,
            max_rows=self.config.head_stats_max_rows,
        )
        logger.info("Per-head upstream error stats enabled -> %s/head_stats.parquet", results_dir)

    def _head_stats_run_config(self) -> dict:
        """Everything needed to interpret head_stats.parquet without its directory."""
        import torch
        import transformers

        cfg = {k: v for k, v in vars(self.config).items() if not k.startswith("_")}
        model_cfg = getattr(getattr(self.pipeline, "model", None), "config", None)
        if model_cfg is not None:
            cfg.update(
                n_layers=getattr(model_cfg, "num_hidden_layers", None),
                n_q_heads=getattr(model_cfg, "num_attention_heads", None),
                n_kv_heads=getattr(model_cfg, "num_key_value_heads", None),
                head_dim=getattr(model_cfg, "head_dim", None),
                attn_implementation=getattr(model_cfg, "_attn_implementation", None),
            )
        cfg.update(torch_version=torch.__version__, transformers_version=transformers.__version__)
        if torch.cuda.is_available():
            cfg["gpu_name"] = torch.cuda.get_device_name(0)
        return cfg

    def _flush_micro_metrics(self) -> None:
        from kvpress.metric_logging import HeadStatsCollector

        HeadStatsCollector().close()
        if not self.config.enable_micro_metrics:
            return
        from kvpress.metric_logging import MicroMetricLogger

        MicroMetricLogger().flush()

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            assert (
                self.config.head_compression_ratio is not None
            ), "head_compression_ratio must be set for DuoAttentionPress"
            press.head_compression_ratio = self.config.head_compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {press.head_compression_ratio}")
        elif isinstance(press, DMSPress):
            assert self.config.threshold is not None, "threshold must be set for DMSPress"
            press.threshold = self.config.threshold
            logger.info(f"Set DMSPress threshold to {press.threshold}")
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert (
                        key_channel_compression_ratio is not None
                    ), "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    # Check if compression_ratio attribute exists before setting
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        elif isinstance(press, DecodingPress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                f"Set DecodingPress compression_interval to {self.config.compression_interval}, target_size to {self.config.target_size}, hidden_states_buffer_size to {self.config.hidden_states_buffer_size}"
            )
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. This is expected is you set `no_press`."
                )

        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _filter_by_context_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows whose tokenized context length is in [min_context_length, max_context_length]."""
        min_len = self.config.min_context_length
        max_len = self.config.max_context_length
        assert min_len is not None
        tokenizer = self.pipeline.tokenizer  # type: ignore[union-attr]
        cap = (max_len + 1) if max_len is not None else min_len
        lengths: list[int] = []
        for text in tqdm(df["context"], desc="Filtering by context length"):
            n_tokens = len(
                tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=cap,
                )
            )
            lengths.append(n_tokens)
        token_counts = pd.Series(lengths, index=df.index)
        mask = token_counts >= min_len
        if max_len is not None:
            mask &= token_counts <= max_len
        filtered = df.loc[mask]
        upper = max_len if max_len is not None else "inf"
        logger.info(f"Kept {len(filtered)}/{len(df)} samples with context tokens in [{min_len}, {upper}].")
        if len(filtered) == 0:
            raise ValueError(f"No samples remain after filtering context tokens to [{min_len}, {upper}].")
        return filtered

    def _load_local_dataset(self, path: Path) -> pd.DataFrame:
        """Load a previously saved evaluation subset from disk or a parquet file."""
        logger.info(f"Loading local dataset from {path}")
        if path.is_file() and path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            ds = load_from_disk(str(path))
            if hasattr(ds, "keys") and "test" in ds:
                df = ds["test"].to_pandas()
            else:
                df = ds.to_pandas()
        logger.info(f"Loaded {len(df)} local samples.")
        return df

    def _load_and_prepare_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        if self.config.dataset_path:
            df = self._load_local_dataset(Path(self.config.dataset_path))
        elif dataset_name in (*RULER_CONFIG_SPLIT, *RULER_RAW_SPLIT):
            df = load_ruler_length_dataset(dataset_name, data_dir)
        elif dataset_name == "loft":
            df = load_loft_dataset(data_dir)
        elif dataset_name in AIME_CONFIGS:
            df = load_aime_dataset(dataset_name)
        elif dataset_name == "scbench":
            # scbench publishes its tasks as dataset configs, so the task, already normalized by the config,
            # is passed as the config name and not as data_dir: a config can be resolved from a local cache
            # without reaching the hub, which a data_dir cannot.
            logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (config: {data_dir})")
            df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir, split="test").to_pandas()
        else:
            logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
            df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()

            if self.config.min_context_length is not None:
                df = self._filter_by_context_length(df)

        if self.config.n_samples is not None:
            original_len = len(df)
            n = min(self.config.n_samples, original_len)
            if n < self.config.n_samples:
                logger.warning(
                    f"Requested {self.config.n_samples} samples but only {original_len} remain; using {n}."
                )
            df = df.sample(n=n, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} of {original_len} remaining samples.")
        elif fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
            )

        # scbench ships one shared context plus several follow-up turns per sample: expand it to one row per
        # turn, after sampling, so that fraction selects whole contexts
        if self.config.dataset == "scbench":
            df = flatten_multi_turns(
                df,
                task=data_dir,
                tokenizer=self.pipeline.tokenizer,
                max_context_length=self.config.max_context_length,
            )

        if isinstance(self.press, FinchPress):
            if not self.config.query_aware:
                logger.error("FinchPress requires 'query_aware' to be set to True.")
                raise ValueError("FinchPress requires query_aware to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.query_aware:
            logger.info("Query-aware compression: including question in context for compression.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config.model_kwargs or {}

        if self.config.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
            logger.info("FP8 quantization enabled.")

        if model_kwargs.get("attn_implementation"):
            logger.info(
                "attn_implementation '%s' requested explicitly; leaving it alone.",
                model_kwargs["attn_implementation"],
            )
        elif isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        elif isinstance(self.press, (BernoulliPress, VKvWRPress)):
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info(f"{type(self.press).__name__} detected, setting attn_implementation to 'sdpa'.")
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        pipeline_kwargs: Dict[str, Any] = {
            "model": model_name,
            "model_kwargs": model_kwargs,
        }
        if self.config.trust_remote_code:
            logger.warning(
                "trust_remote_code=True: the model repository can execute arbitrary Python code during loading."
            )
            pipeline_kwargs["trust_remote_code"] = True
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        if needs_text_only_load(model_name, self.config.trust_remote_code):
            # Ministral 3: the causal LM sits inside a VLM checkpoint with no AutoModelForCausalLM mapping
            model = load_text_only_causal_lm(
                model_name, model_kwargs, device_map="auto" if device == "auto" else device
            )
            self.pipeline = pipeline(
                "kv-press-text-generation", model=model, tokenizer=AutoTokenizer.from_pretrained(model_name)
            )
        else:
            self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)
        add_chat_eos(self.pipeline.model, self.pipeline.tokenizer)
        suppress_think(self.pipeline.model, self.pipeline.tokenizer)

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]

        if isinstance(self.press, DecodingPress):
            logger.info("DecodingPress detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
                output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[index, "predicted_answer"] = output["answer"]  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for context, df_group in tqdm(
                df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
            ):  # type: ignore[union-attr]
                questions = df_group["question"].to_list()
                # Use max_new_tokens from config, or fallback to dataset's default for the task
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]

                output = self.pipeline(  # type: ignore[misc]
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[df_group.index, "predicted_answer"] = output["answers"]  # type: ignore[union-attr]
                # Store the actual compression ratio used (if the press has one)
                self.df.loc[df_group.index, "compression_ratio"] = (
                    self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                )  # type: ignore[union-attr, attr-defined]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        logger.info("Inference completed.")

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_csv(
            str(save_filename), index=False
        )  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = scorer(self.df)  # type: ignore[call-arg]

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()

        results_dir = self.config.get_results_dir(output_dir)
        predictions_filename = results_dir / "predictions.csv"
        metrics_filename = results_dir / "metrics.json"
        config_filename = results_dir / "config.yaml"

        if predictions_filename.exists() and metrics_filename.exists():
            logger.info(
                f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
            )
            return

        self._setup_press()
        self._setup_model_pipeline()
        self._load_and_prepare_dataset()
        self._configure_micro_metrics(results_dir)
        self._configure_head_stats(results_dir)

        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename)
        self._flush_micro_metrics()
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
