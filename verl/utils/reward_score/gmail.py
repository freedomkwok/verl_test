
import re
from typing import Optional


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    original_has_finish = extra_info.get("original_has_finish", False)
    success = extra_info.get("success", False)
    extra_info
    reward = 0.0
    if original_has_finish and success:
        reward += 0.5
    elif original_has_finish and not success:
        reward += -0.8
    elif not original_has_finish and success:
        reward += 1.0

    similarity_scores = extra_info.get("similarity_scores", 0.8)
    reward += (similarity_scores - 0.8)
    
    return {
        "score": reward,
        "acc": 0,
        "pred": 0,
    }