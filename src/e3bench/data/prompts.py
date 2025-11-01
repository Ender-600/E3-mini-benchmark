"""Prompt templates and few-shot example selection."""

import random
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def select_fewshot_examples(
    examples: List[Dict[str, Any]],
    num_examples: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Select few-shot examples deterministically."""
    random.seed(seed)
    return random.sample(examples, min(num_examples, len(examples)))


def get_fewshot_prompt(
    task: str,
    examples: List[Dict[str, Any]],
    test_example: Dict[str, Any],
    num_fewshot: int = 5
) -> str:
    """Generate few-shot prompt for a given task."""
    
    # Select examples
    selected_examples = select_fewshot_examples(examples, num_fewshot)
    
    if task == "BoolQ":
        return _get_boolq_prompt(selected_examples, test_example)
    elif task == "RTE":
        return _get_rte_prompt(selected_examples, test_example)
    elif task == "WiC":
        return _get_wic_prompt(selected_examples, test_example)
    elif task == "CB":
        return _get_cb_prompt(selected_examples, test_example)
    elif task == "mmlu":
        return _get_mmlu_prompt(selected_examples, test_example)
    elif task == "arc_challenge":
        return _get_arc_prompt(selected_examples, test_example)
    elif task == "hellaswag":
        return _get_hellaswag_prompt(selected_examples, test_example)
    else:
        raise ValueError(f"Unknown task: {task}")


def _get_boolq_prompt(examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
    """Generate BoolQ few-shot prompt."""
    prompt = "Answer the following yes/no questions based on the given passage.\n\n"
    
    for example in examples:
        label = "Yes" if example["label"] == 1 else "No"
        prompt += f"Passage: {example['passage']}\n"
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {label}\n\n"
    
    # Add test example
    prompt += f"Passage: {test_example['passage']}\n"
    prompt += f"Question: {test_example['question']}\n"
    prompt += "Answer:"
    
    return prompt


def _get_rte_prompt(examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
    """Generate RTE few-shot prompt."""
    prompt = "Determine if the hypothesis is entailed by the premise.\n\n"
    
    for example in examples:
        label = "entailment" if example["label"] == 1 else "not_entailment"
        prompt += f"Premise: {example['premise']}\n"
        prompt += f"Hypothesis: {example['hypothesis']}\n"
        prompt += f"Entailment: {label}\n\n"
    
    # Add test example
    prompt += f"Premise: {test_example['premise']}\n"
    prompt += f"Hypothesis: {test_example['hypothesis']}\n"
    prompt += "Entailment:"
    
    return prompt


def _get_wic_prompt(examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
    """Generate WiC few-shot prompt."""
    prompt = "Determine if the word has the same meaning in both sentences.\n\n"
    
    for example in examples:
        label = "True" if example["label"] == 1 else "False"
        prompt += f"Word: {example['word']}\n"
        prompt += f"Sentence 1: {example['sentence1']}\n"
        prompt += f"Sentence 2: {example['sentence2']}\n"
        prompt += f"Same meaning: {label}\n\n"
    
    # Add test example
    prompt += f"Word: {test_example['word']}\n"
    prompt += f"Sentence 1: {test_example['sentence1']}\n"
    prompt += f"Sentence 2: {test_example['sentence2']}\n"
    prompt += "Same meaning:"
    
    return prompt


def _get_cb_prompt(examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
    """Generate CB few-shot prompt."""
    label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
    prompt = "Determine the relationship between premise and hypothesis.\n\n"
    
    for example in examples:
        label = label_map.get(example["label"], "neutral")
        prompt += f"Premise: {example['premise']}\n"
        prompt += f"Hypothesis: {example['hypothesis']}\n"
        prompt += f"Relationship: {label}\n\n"
    
    # Add test example
    prompt += f"Premise: {test_example['premise']}\n"
    prompt += f"Hypothesis: {test_example['hypothesis']}\n"
    prompt += "Relationship:"
    
    return prompt


def _get_mmlu_prompt(examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
    """Generate MMLU few-shot prompt."""
    prompt = f"Answer the following multiple choice question about {test_example.get('subject', 'general knowledge')}.\n\n"
    
    for example in examples:
        choices = example["choices"]
        correct_idx = example["answer"]
        correct_choice = choices[correct_idx]
        
        prompt += f"Question: {example['question']}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += f"Answer: {chr(65+correct_idx)}. {correct_choice}\n\n"
    
    # Add test example
    choices = test_example["choices"]
    prompt += f"Question: {test_example['question']}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    
    return prompt


def _get_arc_prompt(examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
    """Generate ARC Challenge few-shot prompt."""
    prompt = "Answer the following multiple choice science question.\n\n"
    
    for example in examples:
        choices = example["choices"]["text"]
        correct_idx = example["answerKey"]
        correct_choice = choices[correct_idx]
        
        prompt += f"Question: {example['question']}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += f"Answer: {chr(65+correct_idx)}. {correct_choice}\n\n"
    
    # Add test example
    choices = test_example["choices"]["text"]
    prompt += f"Question: {test_example['question']}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    
    return prompt


def _get_hellaswag_prompt(examples: List[Dict[str, Any]], test_example: Dict[str, Any]) -> str:
    """Generate HellaSwag few-shot prompt."""
    prompt = "Complete the following sentence by choosing the best option.\n\n"
    
    for example in examples:
        choices = example["endings"]
        correct_idx = example["label"]
        correct_choice = choices[correct_idx]
        
        prompt += f"Context: {example['ctx']}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += f"Answer: {chr(65+correct_idx)}. {correct_choice}\n\n"
    
    # Add test example
    choices = test_example["endings"]
    prompt += f"Context: {test_example['ctx']}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    
    return prompt
