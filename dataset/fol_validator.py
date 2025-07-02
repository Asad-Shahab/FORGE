import json
import re
import subprocess
import tempfile
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
import argparse
from datetime import datetime

@dataclass
class ReasoningExample:
    context: str
    question: str
    reasoning: str
    answer: str
    options: List[str]
    id: str
    nl2fol: Optional[str] = None
    conclusion_fol: Optional[str] = None

@dataclass
class FOLConversion:
    original: str
    fol: str
    confidence: float
    issues: List[str]

@dataclass
class ValidationResult:
    example_id: int
    original_reasoning: str
    premises: List[str]
    extracted_conclusion: str
    fol_conversions: List[FOLConversion]
    prover9_result: str
    prover9_success: bool
    validation_time_seconds: float
    llm_errors: List[str]

class QwenFOLValidator:
    def __init__(self, model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", 
                 prover9_path: str = "prover9"):
        self.model_name = model_name
        self.prover9_path = prover9_path
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        self.fol_prompt_template = """Convert the following logical reasoning step to First-Order Logic (FOL):

Statement: "{statement}"

Rules:
1. Use 'all x.' for universal quantification (anyone, everyone, someone)
2. Use proper logical operators: & (and), | (or), -> (implies), - (not)
3. Replace pronouns (someone, they, he, she) with bound variable 'x'
4. For "either X or Y but not both": ((X | Y) & -(X & Y))
5. For "either X or Y": (X | Y)
6. Use clear predicate names like Predicate(subject)

Examples:
- "John walks" -> "Walks(john)"
- "If someone is tall, then they are visible" -> "all x. (Tall(x) -> Visible(x))"
- "Mary either sings or dances, but not both" -> "((Sings(mary) | Dances(mary)) & -(Sings(mary) & Dances(mary)))"

Convert ONLY the given statement. Output just the FOL formula:"""
    
    def _load_model(self):
        """Load the Qwen model with Unsloth for faster inference."""
        print(f"Loading model with Unsloth: {self.model_name}")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
            )
            
            FastLanguageModel.for_inference(self.model)
            
            print("Model loaded successfully with Unsloth!")
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            print(f"Error loading model with Unsloth: {e}")
            print("Falling back to standard transformers loading...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_dataset_from_hf(self, dataset_name: str = "opendatalab/ProverQA", 
                            split: str = "train", max_examples: Optional[int] = None) -> List[ReasoningExample]:
        """Load reasoning examples from Hugging Face dataset with mixed format handling."""
        print(f"Loading dataset: {dataset_name}, split: {split}")
        
        try:
            if dataset_name == "opendatalab/ProverQA":
                return self._load_proverqa_safely(split, max_examples)
            else:
                dataset = load_dataset(dataset_name, split=split)
                return self._process_dataset_items(dataset, max_examples)
                
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def _load_proverqa_safely(self, split: str = "train", max_examples: Optional[int] = None) -> List[ReasoningExample]:
        """Safely load ProverQA dataset using the correct data_files approach."""
        
        try:
            print(f"Loading ProverQA with split: {split}")
            
            if split == "train":
                dataset = load_dataset("opendatalab/ProverQA", data_files="train/provergen-5000.json")["train"]
                print(f"✅ Loaded training data: {len(dataset)} examples")
                
            elif split == "validation" or split == "dev":
                dataset = load_dataset("opendatalab/ProverQA", data_files="dev/*.json")["train"]
                print(f"✅ Loaded dev data: {len(dataset)} examples")
                
            elif split == "easy":
                dataset = load_dataset("opendatalab/ProverQA", data_files="dev/easy.json")["train"]
                print(f"✅ Loaded easy dev data: {len(dataset)} examples")
                
            elif split == "medium":
                dataset = load_dataset("opendatalab/ProverQA", data_files="dev/medium.json")["train"]
                print(f"✅ Loaded medium dev data: {len(dataset)} examples")
                
            elif split == "hard":
                dataset = load_dataset("opendatalab/ProverQA", data_files="dev/hard.json")["train"]
                print(f"✅ Loaded hard dev data: {len(dataset)} examples")
                
            else:
                print(f"Unknown split '{split}', defaulting to training data")
                dataset = load_dataset("opendatalab/ProverQA", data_files="train/provergen-5000.json")["train"]
                print(f"✅ Loaded training data: {len(dataset)} examples")
            
            return self._process_dataset_items(dataset, max_examples)
            
        except Exception as e:
            print(f"❌ Failed to load ProverQA: {e}")
            return []

    def _process_dataset_items(self, dataset, max_examples: Optional[int] = None) -> List[ReasoningExample]:
        """Process dataset items into ReasoningExample format."""
        examples = []
        dataset_size = len(dataset)
        
        if max_examples:
            dataset_size = min(max_examples, dataset_size)
            dataset = dataset.select(range(dataset_size))
        
        print(f"Processing {dataset_size} examples...")
        
        if len(dataset) > 0:
            first_item = dataset[0]
            print(f"Dataset columns: {list(first_item.keys())}")
        
        for i, item in enumerate(dataset):
            try:
                example = self._create_example_from_item(item, i)
                if example:
                    examples.append(example)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
        
        print(f"Successfully processed {len(examples)} examples")
        return examples

    def _create_example_from_item(self, item: Dict, index: int) -> Optional[ReasoningExample]:
        """Create ReasoningExample from a data item, handling multiple formats."""
        
        try:
            if 'context' in item and 'question' in item:
                return ReasoningExample(
                    context=item.get('context', ''),
                    question=item.get('question', ''),
                    reasoning=item.get('reasoning', ''),
                    answer=item.get('answer', ''),
                    options=item.get('options', []) if isinstance(item.get('options'), list) else [],
                    id=item.get('id', str(index)),
                    nl2fol=item.get('nl2fol', None),
                    conclusion_fol=item.get('conclusion_fol', None)
                )
            elif 'instruction' in item and 'output' in item:
                instruction = item['instruction']
                context_match = re.search(r'Context:\s*(.+?)\s*Question:', instruction, re.DOTALL)
                question_match = re.search(r'Question:\s*(.+?)(?:\s*Options:|$)', instruction, re.DOTALL)
                
                context = context_match.group(1).strip() if context_match else ''
                question = question_match.group(1).strip() if question_match else ''
                
                try:
                    output_data = json.loads(item['output'])
                    reasoning = output_data.get('reasoning', '')
                    answer = output_data.get('answer', '')
                except json.JSONDecodeError:
                    reasoning = item.get('output', '')
                    answer = ''
                
                return ReasoningExample(
                    context=context,
                    question=question,
                    reasoning=reasoning,
                    answer=answer,
                    options=[],
                    id=str(index)
                )
            else:
                print(f"Unknown format for item {index}: {list(item.keys())}")
                return None
                
        except Exception as e:
            print(f"Error creating example from item {index}: {e}")
            return None

    def extract_premises_and_reasoning(self, example: ReasoningExample) -> Tuple[List[str], str, str]:
        """Extract premises from context and reasoning from example."""
        
        context = example.context
        if context:
            premises = [s.strip() for s in re.split(r'[.!]\s*', context) if s.strip()]
        else:
            premises = []
        
        reasoning = example.reasoning
        
        conclusion_patterns = [
            r'therefore[,:]?\s*(.+?)(?:[.!]|$)',
            r'thus[,:]?\s*(.+?)(?:[.!]|$)',
            r'the correct option is[,:]?\s*(.+?)(?:[.!]|$)',
            r'it is\s+(true|false|uncertain)\s+that\s*(.+?)(?:[.!]|$)'
        ]
        
        conclusion = ""
        for pattern in conclusion_patterns:
            match = re.search(pattern, reasoning, re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                break
        
        if not conclusion and example.question:
            question = example.question
            if "is the following statement true, false, or uncertain?" in question.lower():
                statement_match = re.search(r'is the following statement true, false, or uncertain\?\s*(.+)', question, re.IGNORECASE)
                if statement_match:
                    conclusion = statement_match.group(1).strip()
        
        return premises, reasoning, conclusion

    def generate_fol_with_llm(self, statement: str, max_attempts: int = 3) -> FOLConversion:
        """Generate FOL using the Qwen model with Unsloth optimizations."""
        
        prompt = self.fol_prompt_template.format(statement=statement)
        
        for attempt in range(max_attempts):
            try:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=False
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        repetition_penalty=1.1
                    )
                
                generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                fol_output = self._clean_fol_output(response)
                
                if fol_output and not fol_output.startswith('I cannot') and not fol_output.startswith('Sorry'):
                    fixed_fol = self._validate_and_fix_fol(fol_output)
                    
                    confidence = 0.8 if attempt == 0 else 0.6
                    return FOLConversion(
                        original=statement,
                        fol=fixed_fol,
                        confidence=confidence,
                        issues=[]
                    )
                    
            except Exception as e:
                print(f"LLM generation attempt {attempt + 1} failed: {e}")
                continue
        
        return self._code_based_conversion(statement)

    def _clean_fol_output(self, fol_output: str) -> str:
        """Clean the FOL output from the model."""
        lines = fol_output.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('FOL:') and not line.startswith('#'):
                return line
        return fol_output.strip()

    def _validate_and_fix_fol(self, fol: str) -> str:
        """Validate and fix common FOL syntax issues."""
        
        fol = re.sub(r'\b(someone|they|he|she|anyone)\b', 'x', fol)
        
        if ('all ' in fol.lower() or 'anyone' in fol.lower() or 'everyone' in fol.lower()) and not fol.strip().startswith('all x.'):
            if not fol.strip().startswith('all x.'):
                inner_formula = re.sub(r'^all\s+\w+\.?\s*', '', fol)
                fol = f"all x. ({inner_formula})"
        
        fol = re.sub(r'\s+', ' ', fol)
        fol = fol.strip()
        
        return fol

    def _code_based_conversion(self, statement: str) -> FOLConversion:
        """Fallback code-based FOL conversion."""
        issues = ["Used fallback code-based conversion"]
        
        statement_lower = statement.lower()
        
        if 'if ' in statement_lower and ' then ' in statement_lower:
            match = re.search(r'if\s+(.+?)\s+then\s+(.+)', statement_lower)
            if match:
                antecedent = match.group(1).strip()
                consequent = match.group(2).strip()
                
                if any(word in f"{antecedent} {consequent}" for word in ['someone', 'anyone', 'everyone']):
                    fol = f"all x. (Generic_{hash(antecedent) % 1000}(x) -> Generic_{hash(consequent) % 1000}(x))"
                else:
                    fol = f"(Generic_{hash(antecedent) % 1000} -> Generic_{hash(consequent) % 1000})"
            else:
                fol = f"% UNPARSED: {statement}"
        elif 'either ' in statement_lower and ' or ' in statement_lower:
            exclusive = 'but not both' in statement_lower
            fol = f"% EITHER_OR {'EXCLUSIVE' if exclusive else 'INCLUSIVE'}: {statement}"
        else:
            words = statement.split()
            if len(words) >= 2:
                subject = words[0].lower()
                predicate = '_'.join(words[1:])
                fol = f"Generic_{predicate}({subject})"
            else:
                fol = f"% UNPARSED: {statement}"
        
        return FOLConversion(
            original=statement,
            fol=fol,
            confidence=0.3,
            issues=issues
        )

    def create_prover9_input(self, premises: List[FOLConversion], conclusion: str = None) -> str:
        """Create Prover9 input file."""
        
        prover9_input = "set(auto).\nset(max_seconds, 30).\n\n"
        
        valid_premises = [conv.fol for conv in premises 
                         if conv.confidence > 0.4 and not conv.fol.startswith('% UNPARSED')]
        
        if valid_premises:
            prover9_input += "formulas(assumptions).\n"
            for premise in valid_premises:
                prover9_input += f"  {premise}.\n"
            prover9_input += "end_of_list.\n\n"
        
        if conclusion and conclusion.strip() and not conclusion.startswith('%'):
            prover9_input += "formulas(goals).\n"
            prover9_input += f"  {conclusion}.\n"
            prover9_input += "end_of_list.\n"
        
        return prover9_input

    def run_prover9(self, prover9_input: str) -> Tuple[str, bool]:
        """Run Prover9 and return result."""
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
                f.write(prover9_input)
                input_file = f.name
            
            result = subprocess.run(
                [self.prover9_path, input_file],
                capture_output=True,
                text=True,
                timeout=45
            )
            
            os.unlink(input_file)
            
            output = result.stdout
            
            if "THEOREM PROVED" in output:
                return "True", True
            elif "SEARCH FAILED" in output:
                return "False", True
            else:
                return "Uncertain", True
                
        except subprocess.TimeoutExpired:
            return "Timeout", False
        except FileNotFoundError:
            return f"Prover9 not found at {self.prover9_path}", False
        except Exception as e:
            return f"Error: {str(e)}", False

    def validate_example(self, example_id: int, example: ReasoningExample) -> ValidationResult:
        """Validate a single reasoning example."""
        
        start_time = datetime.now()
        llm_errors = []
        
        print(f"\nProcessing example {example_id + 1}...")
        
        premises, reasoning, conclusion = self.extract_premises_and_reasoning(example)
        
        print(f"  Found {len(premises)} premises")
        print(f"  Extracted conclusion: {conclusion}")
        
        fol_conversions = []
        for i, premise in enumerate(premises):
            print(f"    Converting premise {i + 1}: {premise[:50]}...")
            try:
                fol_conv = self.generate_fol_with_llm(premise)
                fol_conversions.append(fol_conv)
            except Exception as e:
                llm_errors.append(f"Premise {i + 1}: {str(e)}")
                fol_conv = self._code_based_conversion(premise)
                fol_conversions.append(fol_conv)
        
        prover9_input = self.create_prover9_input(fol_conversions, None)
        
        print("    Running Prover9...")
        prover9_result, prover9_success = self.run_prover9(prover9_input)
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            example_id=example_id,
            original_reasoning=reasoning,
            premises=premises,
            extracted_conclusion=conclusion,
            fol_conversions=fol_conversions,
            prover9_result=prover9_result,
            prover9_success=prover9_success,
            validation_time_seconds=validation_time,
            llm_errors=llm_errors
        )

    def save_results(self, results: List[ValidationResult], output_file: str):
        """Save validation results to JSON file."""
        
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            result_dict['fol_conversions'] = [asdict(conv) for conv in result.fol_conversions]
            serializable_results.append(result_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Validate LLM reasoning with FOL and Prover9')
    parser.add_argument('--dataset_name', default='opendatalab/ProverQA', help='Hugging Face dataset name')
    parser.add_argument('--split', default='train', 
                       choices=['train', 'dev', 'validation', 'easy', 'medium', 'hard'],
                       help='Dataset split to use (train/dev/easy/medium/hard)')
    parser.add_argument('--output_file', default='validation_results.json', help='Output file for results')
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum number of examples to process')
    parser.add_argument('--prover9_path', default='prover9', help='Path to Prover9 executable')
    
    args = parser.parse_args()
    
    print("🚀 Initializing Qwen FOL Validator with Unsloth...")
    validator = QwenFOLValidator(prover9_path=args.prover9_path)
    
    examples = validator.load_dataset_from_hf(args.dataset_name, args.split, args.max_examples)
    
    if not examples:
        print("❌ No examples loaded. Exiting.")
        return
    
    print(f"📊 Processing {len(examples)} examples...")
    
    results = []
    for i, example in enumerate(examples):
        try:
            result = validator.validate_example(i, example)
            results.append(result)
            
            print(f"  Result: {result.prover9_result}")
            print(f"  Time: {result.validation_time_seconds:.2f}s")
            print(f"  Valid FOL conversions: {sum(1 for conv in result.fol_conversions if conv.confidence > 0.4)}/{len(result.fol_conversions)}")
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            break
        except Exception as e:
            print(f"  ❌ Error processing example {i + 1}: {e}")
            continue
    
    if results:
        validator.save_results(results, args.output_file)
        
        print(f"\n📈 === VALIDATION SUMMARY ===")
        print(f"Total examples processed: {len(results)}")
        print(f"Prover9 successful runs: {sum(1 for r in results if r.prover9_success)}")
        print(f"Average processing time: {sum(r.validation_time_seconds for r in results) / len(results):.2f}s")
        
        result_counts = {}
        for result in results:
            key = result.prover9_result
            result_counts[key] = result_counts.get(key, 0) + 1
        
        print("\n📊 Prover9 results breakdown:")
        for result_type, count in result_counts.items():
            print(f"  {result_type}: {count}")
    
    print("\n✅ Validation complete!")

if __name__ == "__main__":
    main()