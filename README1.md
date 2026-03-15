# ENGR-219-Project3
## Question 1
```python
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BATCH_SIZE = 16
NUM_QUESTIONS = 100

# Standardized system prompt to enforce step-by-step reasoning 
# and a strict format for the final numerical answer.
SYSTEM_PROMPT = (
    "You are a helpful and precise math assistant. "
    "Please solve the following math word problem step-by-step. "
    "After showing your reasoning, you MUST output the final numerical answer "
    "on a new line in the following exact format: 'Final Answer: [number]'."
)

# ==========================================
# 2. Helper Functions for Extraction
# ==========================================
def extract_ground_truth(answer_str):
    """
    Extracts the ground truth number from GSM8K's answer format.
    GSM8K always places the final answer after '#### '.
    """
    if "#### " in answer_str:
        ans = answer_str.split("#### ")[-1].strip()
        return ans.replace(",", "") # Remove commas for clean comparison
    return None

def extract_model_answer(generated_text):
    """
    Extracts the numerical answer from the model's generated text 
    based on the standardized prompt format 'Final Answer: [number]'.
    """
    # Look for 'Final Answer: ' followed by numbers (including optional negative sign and decimals)
    match = re.search(r"Final Answer:\s*\[?(-?\d[\d,]*(?:\.\d+)?)\]?", generated_text, re.IGNORECASE)
    if match:
        ans = match.group(1).replace(",", "")
        # Remove any trailing periods that might be captured
        if ans.endswith('.'):
            ans = ans[:-1]
        return ans
    return None

# ==========================================
# 3. Model and Dataset Initialization
# ==========================================
def main():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    
    # Initialize tokenizer with left-padding for batch generation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Initialize model (bfloat16 is recommended for modern GPUs like T4/A100)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("Loading GSM8K dataset...")
    # Load the test split and select the first 100 questions for the fixed subset
    dataset = load_dataset("gsm8k", "main", split=f"test[:{NUM_QUESTIONS}]")

    # ==========================================
# 4. Batched Evaluation Loop
    # ==========================================
    correct_count = 0
    
    print(f"Evaluating {NUM_QUESTIONS} questions with batch size {BATCH_SIZE}...")
    
    # Process dataset in chunks of BATCH_SIZE
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing Batches"):
        batch = dataset[i : i + BATCH_SIZE]
        questions = batch["question"]
        raw_answers = batch["answer"]
        
        # Prepare prompts using the model's chat template
        prompt_texts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompt_texts.append(text)
            
        # Tokenize the batch
        encoded_inputs = tokenizer(
            prompt_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=512,
                temperature=0.0, # Greedy decoding for deterministic evaluation
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Evaluate each sequence in the batch
        for j in range(len(questions)):
            # Isolate the newly generated tokens (ignore the prompt tokens)
            input_len = encoded_inputs.input_ids[j].shape[0]
            gen_tokens = outputs[j][input_len:]
            
            response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            model_ans = extract_model_answer(response_text)
            true_ans = extract_ground_truth(raw_answers[j])
            
            if model_ans is not None and true_ans is not None and model_ans == true_ans:
                correct_count += 1

    # ==========================================
    # 5. Results Calculation
    # ==========================================
    accuracy = (correct_count / NUM_QUESTIONS) * 100
    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS")
    print("="*50)
    print(f"Total Questions Evaluated : {NUM_QUESTIONS}")
    print(f"Total Correct Predictions : {correct_count}")
    print(f"Baseline Accuracy         : {accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
```
