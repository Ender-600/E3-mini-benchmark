"""MMLU data loading utilities."""

from datasets import load_dataset
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def load_mmlu_data() -> Dict[str, Any]:
    """Load MMLU dataset."""
    try:
        logger.info("Loading MMLU dataset")
        dataset = load_dataset("lukaemon/mmlu", "all")
        
        # MMLU has train/validation/test splits
        return {
            "train": dataset["train"],
            "validation": dataset["validation"], 
            "test": dataset["test"]
        }
    except Exception as e:
        logger.error(f"Failed to load MMLU: {e}")
        return {}


def get_mmlu_subjects() -> List[str]:
    """Get list of MMLU subjects."""
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "college_psychology", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality", "international_law",
        "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
        "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]
    return subjects
