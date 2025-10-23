"""SuperGLUE data loading utilities."""

from datasets import load_dataset
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def load_superglue_data(
    tasks: List[str] = None,
    max_length: int = 256
) -> Dict[str, Any]:
    """Load SuperGLUE datasets for specified tasks."""
    
    if tasks is None:
        tasks = ["BoolQ", "RTE", "WiC", "CB"]
    
    datasets = {}
    
    for task in tasks:
        try:
            logger.info(f"Loading SuperGLUE task: {task}")
            
            # Load the dataset
            dataset = load_dataset("super_glue", task.lower())
            
            # Store train/validation splits
            datasets[task] = {
                "train": dataset["train"],
                "validation": dataset["validation"],
                "test": dataset.get("test", dataset["validation"])  # Use validation as test if no test split
            }
            
            logger.info(f"Loaded {task}: {len(datasets[task]['train'])} train, {len(datasets[task]['validation'])} validation")
            
        except Exception as e:
            logger.warning(f"Failed to load {task}: {e}")
            continue
    
    return datasets


def get_superglue_example_format(task: str) -> Dict[str, str]:
    """Get format information for SuperGLUE tasks."""
    
    formats = {
        "BoolQ": {
            "question": "question",
            "passage": "passage", 
            "label": "label",
            "label_names": ["False", "True"]
        },
        "RTE": {
            "premise": "premise",
            "hypothesis": "hypothesis",
            "label": "label", 
            "label_names": ["not_entailment", "entailment"]
        },
        "WiC": {
            "word": "word",
            "sentence1": "sentence1",
            "sentence2": "sentence2", 
            "label": "label",
            "label_names": ["False", "True"]
        },
        "CB": {
            "premise": "premise",
            "hypothesis": "hypothesis",
            "label": "label",
            "label_names": ["contradiction", "entailment", "neutral"]
        }
    }
    
    return formats.get(task, {})
