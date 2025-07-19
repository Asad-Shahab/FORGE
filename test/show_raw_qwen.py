#!/usr/bin/env python3
"""
Show Raw Qwen Output - No processing, just show exactly what Qwen generates
"""

import torch
from setup_models import setup_qwen3

def show_raw_qwen_output():
    """Show complete raw output from Qwen without any processing"""
    
    print("📄 Raw Qwen Output Display")
    print("=" * 50)
    
    # Load model
    print("📥 Loading Qwen...")
    qwen_model, qwen_tokenizer = setup_qwen3()
    print("✅ Loaded")
    
    # Test prompt
    prompt = """Context: All cats are mammals. All mammals are warm-blooded. Fluffy is a cat.

Question: Is Fluffy warm-blooded?

Please solve this step by step using logical reasoning. Format your response exactly like this:

<reasoning>
[Your step-by-step logical reasoning, one statement per line]
[Work through the logic systematically]
[End with your conclusion]
</reasoning>

<answer>
[Yes/No/Uncertain]
</answer>

Solve the problem now:"""
    
    print("🤖 Generating with Qwen...")
    
    try:
        inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        device = next(qwen_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"📊 Input tokens: {inputs['input_ids'].shape[1]}")
        
        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=400,  # Generous token limit
                temperature=0.7,
                do_sample=True,
                pad_token_id=qwen_tokenizer.eos_token_id,
                eos_token_id=qwen_tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=False,
            )
        
        # Get the complete output
        complete_output = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get just the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_part = qwen_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"📊 Generated tokens: {len(generated_tokens)}")
        print(f"📊 Generated characters: {len(generated_part)}")
        print(f"📊 Last token ID: {outputs[0][-1].item()}")
        print(f"📊 EOS token ID: {qwen_tokenizer.eos_token_id}")
        print(f"📊 Ended with EOS: {outputs[0][-1].item() == qwen_tokenizer.eos_token_id}")
        
        print("\n" + "🔥" * 60)
        print("📄 COMPLETE RAW GENERATED OUTPUT:")
        print("🔥" * 60)
        print(repr(generated_part))  # Using repr to show exact characters including newlines
        print("🔥" * 60)
        
        print("\n" + "📜" * 60)
        print("📄 FORMATTED GENERATED OUTPUT:")
        print("📜" * 60)
        print(generated_part)
        print("📜" * 60)
        
        # Character-by-character analysis of the end
        print(f"\n🔍 LAST 100 CHARACTERS (with escape sequences):")
        last_chars = generated_part[-100:] if len(generated_part) >= 100 else generated_part
        print(repr(last_chars))
        
        # Show if it ends mid-sentence
        if not generated_part.strip().endswith(('>', '.', '!', '?')):
            print("⚠️  OUTPUT APPEARS TO END MID-SENTENCE")
        else:
            print("✅ OUTPUT APPEARS TO END PROPERLY")
            
        # Check for tags
        print(f"\n🏷️  TAG ANALYSIS:")
        print(f"Contains '<reasoning>': {'✅' if '<reasoning>' in generated_part else '❌'}")
        print(f"Contains '</reasoning>': {'✅' if '</reasoning>' in generated_part else '❌'}")
        print(f"Contains '<answer>': {'✅' if '<answer>' in generated_part else '❌'}")
        print(f"Contains '</answer>': {'✅' if '</answer>' in generated_part else '❌'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_raw_qwen_output()