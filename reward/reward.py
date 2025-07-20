#!/usr/bin/env python3
"""
Reward Structure for GRPO Logical Reasoning Training
Combines answer correctness, logical validity, and format compliance
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RewardComponents:
    """Container for individual reward components"""
    answer_correctness: float = 0.0
    logical_validity: float = 0.0
    format_compliance: float = 0.0
    total_reward: float = 0.0

class LogicalReasoningReward:
    """
    Comprehensive reward function for logical reasoning tasks
    """
    
    def __init__(self):
        # Reward weights (should sum to 1.0 for interpretability)
        self.weights = {
            'answer_correctness': 0.35,   # 35% - Core task performance
            'logical_validity': 0.55,     # 55% - Logical soundness (primary focus)
            'format_compliance': 0.1      # 10% - Output structure
        }
        
        # Penalty factors
        self.severe_penalty = -0.5  # For completely wrong/invalid responses
        self.format_penalty = -0.2  # For format violations
    
    def calculate_answer_correctness(self, predicted_answer: str, expected_answer: str) -> float:
        """
        Calculate reward for answer correctness
        Returns: 1.0 for correct, 0.0 for incorrect
        """
        if not predicted_answer or not expected_answer:
            return 0.0
        
        # Normalize answers (A/B/C or True/False/Uncertain)
        pred_norm = self._normalize_answer(predicted_answer)
        exp_norm = self._normalize_answer(expected_answer)
        
        return 1.0 if pred_norm == exp_norm else 0.0
    
    def calculate_logical_validity(self, reasoning_steps: List[str], prover9_result: Dict) -> float:
        """
        Calculate reward for logical validity using Prover9 results
        Returns: 0.0 to 1.0 based on logical soundness
        """
        if not prover9_result:
            return 0.0
        
        # Base validity score
        base_score = 1.0 if prover9_result.get('valid', False) else 0.0
        
        # Bonus for additional logical quality metrics
        confidence_bonus = 0.0
        if 'confidence' in prover9_result:
            confidence_bonus = min(0.2, prover9_result['confidence'] * 0.2)
        
        # Penalty for logical errors
        error_penalty = 0.0
        if 'errors' in prover9_result and prover9_result['errors']:
            error_penalty = min(0.3, len(prover9_result['errors']) * 0.1)
        
        return max(0.0, base_score + confidence_bonus - error_penalty)
    
    def calculate_format_compliance(self, response: str) -> float:
        """
        Calculate reward for format compliance
        Expected format:
        <initial_reasoning>...</initial_reasoning>
        <steps>...</steps>
        <answer>...</answer>
        """
        if not response:
            return 0.0
        
        score = 0.0
        
        # Check for required tags (0.4 points each)
        required_tags = ['initial_reasoning', 'steps', 'answer']
        for tag in required_tags:
            if f'<{tag}>' in response and f'</{tag}>' in response:
                score += 0.33
        
        # Bonus for proper structure order (0.1 point)
        if self._check_tag_order(response):
            score += 0.01
        
        return min(1.0, score)
    
    def calculate_composite_reward(self, 
                                 response: str,
                                 expected_answer: str,
                                 prover9_result: Dict,
                                 reasoning_steps: List[str] = None) -> RewardComponents:
        """
        Calculate the complete reward structure
        """
        components = RewardComponents()
        
        # Extract answer from response
        predicted_answer = self._extract_answer_from_response(response)
        
        # Extract reasoning steps if not provided
        if reasoning_steps is None:
            reasoning_steps = self._extract_reasoning_steps(response)
        
        # Calculate individual components
        components.answer_correctness = self.calculate_answer_correctness(
            predicted_answer, expected_answer
        )
        
        components.logical_validity = self.calculate_logical_validity(
            reasoning_steps, prover9_result
        )
        
        components.format_compliance = self.calculate_format_compliance(response)
        
        # Calculate weighted total
        total = (
            components.answer_correctness * self.weights['answer_correctness'] +
            components.logical_validity * self.weights['logical_validity'] +
            components.format_compliance * self.weights['format_compliance']
        )
        
        # Apply penalties for severe violations
        if components.answer_correctness == 0.0 and components.logical_validity == 0.0:
            total += self.severe_penalty
        
        if components.format_compliance < 0.5:
            total += self.format_penalty
        
        components.total_reward = max(-1.0, min(1.0, total))  # Clamp to [-1, 1]
        
        return components
    
    # Helper methods
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer format"""
        answer = answer.strip().upper()
        
        # Map letter answers to semantic answers
        answer_map = {
            'A': 'TRUE',
            'B': 'FALSE', 
            'C': 'UNCERTAIN',
            'TRUE': 'TRUE',
            'FALSE': 'FALSE',
            'UNCERTAIN': 'UNCERTAIN'
        }
        
        return answer_map.get(answer, answer)
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from formatted response"""
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: look for A, B, C at the end
        fallback_match = re.search(r'([ABC])\s*$', response)
        return fallback_match.group(1) if fallback_match else ""
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from formatted response"""
        steps_match = re.search(r'<steps>(.*?)</steps>', response, re.DOTALL | re.IGNORECASE)
        if not steps_match:
            return []
        
        steps_text = steps_match.group(1).strip()
        # Split by lines and clean up
        steps = [step.strip() for step in steps_text.split('\n') if step.strip()]
        return steps
    
    def _check_tag_order(self, response: str) -> bool:
        """Check if tags appear in correct order"""
        tags = ['initial_reasoning', 'steps', 'answer']
        positions = []
        
        for tag in tags:
            match = re.search(f'<{tag}>', response)
            if match:
                positions.append(match.start())
            else:
                return False
        
        return positions == sorted(positions)

# Example usage and reward ranges
def get_reward_examples():
    """Example reward calculations for different response qualities"""
    
    examples = {
        'perfect_response': {
            'description': 'Correct answer, valid logic, perfect format',
            'expected_reward': 0.9 - 1.0
        },
        'correct_but_invalid_logic': {
            'description': 'Correct answer, invalid logic, good format',
            'expected_reward': 0.35 - 0.45  # Only gets answer correctness + format
        },
        'wrong_answer_valid_logic': {
            'description': 'Wrong answer, valid logic, good format',
            'expected_reward': 0.55 - 0.65  # Gets logical validity + format
        },
        'format_violation': {
            'description': 'Any quality but wrong format',
            'expected_reward': 'Base reward - 0.2'
        },
        'completely_wrong': {
            'description': 'Wrong answer, invalid logic, poor format',
            'expected_reward': -0.5 - 0.0
        }
    }
    
    return examples

# Preference pair creation helper
def create_preference_pair(response1: str, response2: str, 
                         expected_answer: str, prover9_result1: Dict, 
                         prover9_result2: Dict, reward_calculator: LogicalReasoningReward) -> Dict:
    """
    Create a preference pair for GRPO training
    Returns the better response as 'chosen' and worse as 'rejected'
    """
    
    reward1 = reward_calculator.calculate_composite_reward(
        response1, expected_answer, prover9_result1
    )
    
    reward2 = reward_calculator.calculate_composite_reward(
        response2, expected_answer, prover9_result2
    )
    
    if reward1.total_reward >= reward2.total_reward:
        return {
            'chosen': response1,
            'rejected': response2,
            'chosen_reward': reward1.total_reward,
            'rejected_reward': reward2.total_reward,
            'reward_difference': reward1.total_reward - reward2.total_reward
        }
    else:
        return {
            'chosen': response2,
            'rejected': response1,
            'chosen_reward': reward2.total_reward,
            'rejected_reward': reward1.total_reward,
            'reward_difference': reward2.total_reward - reward1.total_reward
        }