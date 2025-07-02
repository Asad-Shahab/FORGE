
import os
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

def generate(question_list,model_path):
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        n=1,
        skip_special_tokens=True # hide special tokens such as "<|continue|>", "<|reflect|>", and "<|explore|>"
    )
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = [[output.text for output in output_item.outputs] for output_item in outputs]
    return completions

def prepare_prompt(question):
    prompt = f"<|im_start|>user\nSolve the following math problem efficiently and clearly.\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nProblem: {question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt
    
def run():
    model_path = "Satori-reasoning/Satori-7B-Round2"
    all_problems = [
        "which number is larger? 9.11 or 9.9?",
    ]
    completions = generate(
        [prepare_prompt(problem_data) for problem_data in all_problems],
        model_path
    )
    
    for completion in completions:
        print(completion[0])
if __name__ == "__main__":
    run()

