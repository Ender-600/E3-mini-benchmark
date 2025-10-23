"""Data loading and prompt handling for E³ Mini-Benchmark."""

from .superglue import load_superglue_data
from .mmlu import load_mmlu_data
from .prompts import get_fewshot_prompt, select_fewshot_examples

__all__ = [
    "load_superglue_data",
    "load_mmlu_data", 
    "get_fewshot_prompt",
    "select_fewshot_examples"
]
