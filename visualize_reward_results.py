#!/usr/bin/env python3
"""
Visualization Script for Reward Model Test Results
Analyzes and visualizes results from test_reward_models.py
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

class RewardResultsAnalyzer:
    """Analyzer for reward model test results"""
    
    def __init__(self, results_file):
        self.results_file = results_file
        self.results = self.load_results()
        self.df = self.create_dataframe()
        
    def load_results(self):
        """Load test results from JSON file"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def create_dataframe(self):
        """Create pandas dataframe from results for easy analysis"""
        data = []
        
        for result in self.results:
            row = {
                'example_name': result['example_name'],
                'predicted_answer': result['predicted_answer'],
                'expected_answer': result['expected_answer'],
                'answer_correct': result['predicted_answer'] == result['expected_answer'],
                'prover9_valid': result['verification_result']['valid'],
                'simple_reward': result['rewards']['simple'],
                'complex_total_reward': result['rewards']['complex']['total_reward'],
                'answer_correctness': result['rewards']['complex']['answer_correctness'],
                'logical_validity': result['rewards']['complex']['logical_validity'],
                'format_compliance': result['rewards']['complex']['format_compliance'],
                'num_reasoning_steps': len(result['reasoning_statements']),
                'response_length': len(result['response']),
                'has_fol_errors': any('Error' in fol[1] for fol in result['fol_statements'])
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def print_basic_stats(self):
        """Print basic statistics"""
        print("📊 BASIC STATISTICS")
        print("=" * 50)
        
        total = len(self.df)
        correct_answers = self.df['answer_correct'].sum()
        prover9_valid = self.df['prover9_valid'].sum()
        
        print(f"Total Examples: {total}")
        print(f"Correct Answers: {correct_answers}/{total} ({correct_answers/total*100:.1f}%)")
        print(f"Prover9 Valid: {prover9_valid}/{total} ({prover9_valid/total*100:.1f}%)")
        
        print(f"\nReward Statistics:")
        print(f"Simple Reward - Mean: {self.df['simple_reward'].mean():.3f}, Std: {self.df['simple_reward'].std():.3f}")
        print(f"Complex Reward - Mean: {self.df['complex_total_reward'].mean():.3f}, Std: {self.df['complex_total_reward'].std():.3f}")
        
        print(f"\nComponent Analysis:")
        print(f"Answer Correctness - Mean: {self.df['answer_correctness'].mean():.3f}")
        print(f"Logical Validity - Mean: {self.df['logical_validity'].mean():.3f}")
        print(f"Format Compliance - Mean: {self.df['format_compliance'].mean():.3f}")
    
    def analyze_reward_components(self):
        """Analyze correlation between reward components"""
        print("\n🔍 REWARD COMPONENT ANALYSIS")
        print("=" * 50)
        
        components = ['answer_correctness', 'logical_validity', 'format_compliance']
        
        print("Correlation Matrix:")
        corr_matrix = self.df[components].corr()
        print(corr_matrix.round(3))
        
        # Identify issues
        print(f"\nIssue Analysis:")
        
        # Cases where answer is correct but logic is invalid
        correct_invalid = self.df[(self.df['answer_correct']) & (~self.df['prover9_valid'])]
        print(f"Correct answer but invalid logic: {len(correct_invalid)} cases")
        
        # Cases where logic is valid but answer is wrong  
        valid_wrong = self.df[(~self.df['answer_correct']) & (self.df['prover9_valid'])]
        print(f"Valid logic but wrong answer: {len(valid_wrong)} cases")
        
        # FOL conversion errors
        fol_errors = self.df[self.df['has_fol_errors']].shape[0]
        print(f"FOL conversion errors: {fol_errors} cases")
        
        return corr_matrix
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Reward Model Analysis Dashboard', fontsize=16)
        
        # 1. Reward Distribution
        axes[0, 0].hist(self.df['simple_reward'], alpha=0.7, label='Simple', bins=10)
        axes[0, 0].hist(self.df['complex_total_reward'], alpha=0.7, label='Complex', bins=10)
        axes[0, 0].set_xlabel('Reward Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Score Distributions')
        axes[0, 0].legend()
        
        # 2. Component Breakdown
        components = ['answer_correctness', 'logical_validity', 'format_compliance']
        means = [self.df[comp].mean() for comp in components]
        axes[0, 1].bar(components, means, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_ylabel('Mean Score')
        axes[0, 1].set_title('Reward Component Means')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Correctness vs Validity Scatter
        colors = ['red' if not correct else 'green' for correct in self.df['answer_correct']]
        axes[0, 2].scatter(self.df['logical_validity'], self.df['answer_correctness'], 
                          c=colors, alpha=0.6)
        axes[0, 2].set_xlabel('Logical Validity Score')
        axes[0, 2].set_ylabel('Answer Correctness Score')
        axes[0, 2].set_title('Logical Validity vs Answer Correctness')
        
        # 4. Simple vs Complex Reward Correlation
        axes[1, 0].scatter(self.df['simple_reward'], self.df['complex_total_reward'], alpha=0.6)
        axes[1, 0].plot([0, 10], [0, 1], 'r--', alpha=0.5)  # Reference line
        axes[1, 0].set_xlabel('Simple Reward')
        axes[1, 0].set_ylabel('Complex Reward')
        axes[1, 0].set_title('Simple vs Complex Reward Correlation')
        
        # 5. Response Length vs Reward
        axes[1, 1].scatter(self.df['response_length'], self.df['complex_total_reward'], alpha=0.6)
        axes[1, 1].set_xlabel('Response Length (chars)')
        axes[1, 1].set_ylabel('Complex Reward')
        axes[1, 1].set_title('Response Length vs Reward')
        
        # 6. Success Rate by Category
        categories = ['Answer Correct', 'Prover9 Valid', 'No FOL Errors']
        rates = [
            self.df['answer_correct'].mean(),
            self.df['prover9_valid'].mean(), 
            (~self.df['has_fol_errors']).mean()
        ]
        axes[1, 2].bar(categories, rates, color=['gold', 'orange', 'purple'])
        axes[1, 2].set_ylabel('Success Rate')
        axes[1, 2].set_title('Success Rates by Category')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"reward_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to: {plot_file}")
        
        plt.show()
    
    def identify_problematic_examples(self):
        """Identify and analyze problematic examples"""
        print("\n🚨 PROBLEMATIC EXAMPLES ANALYSIS")
        print("=" * 50)
        
        # Low reward examples
        low_reward_threshold = self.df['complex_total_reward'].quantile(0.25)
        low_reward_examples = self.df[self.df['complex_total_reward'] <= low_reward_threshold]
        
        print(f"Low Reward Examples (≤ {low_reward_threshold:.3f}):")
        for _, row in low_reward_examples.iterrows():
            print(f"  - {row['example_name']}: Reward={row['complex_total_reward']:.3f}, "
                  f"Correct={row['answer_correct']}, Valid={row['prover9_valid']}")
        
        # Inconsistent examples (correct answer but invalid logic or vice versa)
        inconsistent = self.df[
            (self.df['answer_correct'] & ~self.df['prover9_valid']) | 
            (~self.df['answer_correct'] & self.df['prover9_valid'])
        ]
        
        print(f"\nInconsistent Examples ({len(inconsistent)}):")
        for _, row in inconsistent.iterrows():
            status = "Correct but Invalid" if row['answer_correct'] and not row['prover9_valid'] else "Wrong but Valid"
            print(f"  - {row['example_name']}: {status}")
        
        return low_reward_examples, inconsistent
    
    def detailed_example_analysis(self, example_indices=None):
        """Provide detailed analysis of specific examples"""
        if example_indices is None:
            example_indices = list(range(min(3, len(self.results))))  # First 3 examples by default
        
        print(f"\n🔍 DETAILED EXAMPLE ANALYSIS")
        print("=" * 50)
        
        for idx in example_indices:
            if idx >= len(self.results):
                continue
                
            result = self.results[idx]
            print(f"\n📝 Example {idx + 1}: {result['example_name']}")
            print("-" * 30)
            print(f"Question: {result['question'][:100]}...")
            print(f"Expected: {result['expected_answer']} | Predicted: {result['predicted_answer']}")
            
            # Reward breakdown
            rewards = result['rewards']['complex']
            print(f"\nReward Breakdown:")
            print(f"  Answer Correctness: {rewards['answer_correctness']:.3f}")
            print(f"  Logical Validity:   {rewards['logical_validity']:.3f}")
            print(f"  Format Compliance:  {rewards['format_compliance']:.3f}")
            print(f"  Total:              {rewards['total_reward']:.3f}")
            
            # Reasoning analysis
            print(f"\nReasoning Steps ({len(result['reasoning_statements'])}):")
            for i, stmt in enumerate(result['reasoning_statements'][:3], 1):  # Show first 3
                print(f"  {i}. {stmt[:80]}...")
            
            # FOL conversion quality
            fol_errors = sum(1 for _, fol in result['fol_statements'] if 'Error' in fol)
            print(f"\nFOL Conversion: {len(result['fol_statements']) - fol_errors}/{len(result['fol_statements'])} successful")
            
            # Prover9 result
            prover9 = result['verification_result']
            print(f"Prover9 Verification: {'✅ VALID' if prover9['valid'] else '❌ INVALID'}")
            print(f"Details: {prover9['details']}")
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 50)
        
        # Analyze patterns
        avg_complex_reward = self.df['complex_total_reward'].mean()
        prover9_success_rate = self.df['prover9_valid'].mean()
        fol_error_rate = self.df['has_fol_errors'].mean()
        format_compliance = self.df['format_compliance'].mean()
        
        recommendations = []
        
        if avg_complex_reward < 0.5:
            recommendations.append("🔴 LOW OVERALL REWARDS: Consider adjusting reward weights or improving model performance")
        
        if prover9_success_rate < 0.3:
            recommendations.append("🔴 LOW PROVER9 SUCCESS: Check FOL conversion quality and Prover9 setup")
        
        if fol_error_rate > 0.5:
            recommendations.append("🔴 HIGH FOL ERRORS: Llama model may need fine-tuning for FOL conversion")
        
        if format_compliance < 0.8:
            recommendations.append("🔴 FORMAT ISSUES: Model needs better training on required output format")
        
        # Positive observations
        if self.df['answer_correctness'].mean() > 0.7:
            recommendations.append("🟢 GOOD ANSWER ACCURACY: Model performs well on basic correctness")
        
        if self.df['simple_reward'].mean() > 7.0:
            recommendations.append("🟢 GOOD FORMAT COMPLIANCE: Model follows output structure well")
        
        # Print recommendations
        for rec in recommendations:
            print(f"  {rec}")
        
        if not recommendations:
            print("  🟢 No major issues identified. System appears to be working well!")
        
        return recommendations

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Analyze Reward Model Test Results")
    
    parser.add_argument("results_file", type=str,
                      help="Path to JSON results file from test_reward_models.py")
    parser.add_argument("--save_plots", action="store_true", default=True,
                      help="Save visualization plots to file")
    parser.add_argument("--detailed_examples", type=int, nargs='+', 
                      help="Indices of examples for detailed analysis")
    parser.add_argument("--no_plots", action="store_true",
                      help="Skip creating plots (for headless environments)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = RewardResultsAnalyzer(args.results_file)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Run analysis
    print("🔍 REWARD MODEL RESULTS ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    analyzer.print_basic_stats()
    
    # Component analysis
    analyzer.analyze_reward_components()
    
    # Problem identification
    analyzer.identify_problematic_examples()
    
    # Detailed examples
    if args.detailed_examples:
        analyzer.detailed_example_analysis(args.detailed_examples)
    else:
        analyzer.detailed_example_analysis()  # Default first 3
    
    # Recommendations
    analyzer.generate_recommendations()
    
    # Visualizations (if not headless)
    if not args.no_plots:
        try:
            analyzer.create_visualizations(args.save_plots)
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")
            print("   (This might be expected in headless environments)")
    
    print("\n🎉 Analysis completed!")

if __name__ == "__main__":
    main()