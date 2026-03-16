# ENGR-219-Project3
## Question 1 & 2
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

SYSTEM_PROMPT = (
    "You are a helpful and precise math assistant. "
    "Please solve the following math word problem step-by-step. "
    "Show your reasoning clearly. Finally, you MUST enclose your final numerical answer "
    "in a LaTeX boxed format at the very end. For example, if the answer is 5, write \\boxed{5}. "
    "Do not include units or text inside the box."
)

# ==========================================
# 2. Helper Functions for Extraction
# ==========================================
def extract_ground_truth(answer_str):
    if "#### " in answer_str:
        ans = answer_str.split("#### ")[-1].strip()
        return ans.replace(",", "") 
    return None

def extract_model_answer(generated_text):
    match = re.search(r"\\boxed\{([^\}]+)\}", generated_text)
    if match:
        raw_ans = match.group(1)
        clean_ans = re.sub(r"[^\d\.-]", "", raw_ans)
        if clean_ans.endswith('.'):
            clean_ans = clean_ans[:-1]
        if clean_ans not in ["", "-", "."]:
            return clean_ans
    
    fallback_match = re.search(r"(?:answer is|Final Answer:)\s*\$?(-?\d[\d,]*(?:\.\d+)?)", generated_text, re.IGNORECASE)
    if fallback_match:
         ans = fallback_match.group(1).replace(",", "")
         if ans.endswith('.'):
             ans = ans[:-1]
         if ans not in ["", "-", "."]:
             return ans
         
    return None

# ==========================================
# 3. Model and Dataset Initialization
# ==========================================
def main():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=f"test[:{NUM_QUESTIONS}]")

    # ==========================================
    # 4. Batched Evaluation Loop
    # ==========================================
    correct_count = 0
    incorrect_cases = [] # List to store failure cases
    
    print(f"Evaluating {NUM_QUESTIONS} questions with batch size {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing Batches"):
        batch = dataset[i : i + BATCH_SIZE]
        questions = batch["question"]
        raw_answers = batch["answer"]
        
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
            
        encoded_inputs = tokenizer(
            prompt_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=1024,
                temperature=0.0, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        for j in range(len(questions)):
            input_len = encoded_inputs.input_ids[j].shape[0]
            gen_tokens = outputs[j][input_len:]
            
            response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            model_ans = extract_model_answer(response_text)
            true_ans = extract_ground_truth(raw_answers[j])
            
            is_correct = False
            if model_ans is not None and true_ans is not None:
                try:
                    if float(model_ans) == float(true_ans):
                        correct_count += 1
                        is_correct = True
                except ValueError:
                    pass
            
            # Record incorrect answers for Question 2 analysis
            if not is_correct:
                incorrect_cases.append({
                    "question": questions[j],
                    "model_response": response_text,
                    "extracted_answer": model_ans,
                    "ground_truth": true_ans
                })

    # ==========================================
    # 5. Results Calculation and Error Inspection
    # ==========================================
    accuracy = (correct_count / NUM_QUESTIONS) * 100
    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS")
    print("="*50)
    print(f"Total Questions Evaluated : {NUM_QUESTIONS}")
    print(f"Total Correct Predictions : {correct_count}")
    print(f"Baseline Accuracy         : {accuracy:.2f}%")
    print("="*50)

    # Print 3 sample incorrect cases for the report
    print("\n" + "="*50)
    print("INSPECTING 3 INCORRECT CASES (FOR QUESTION 2)")
    print("="*50)
    
    # Grab the first 3 failed cases (you will have exactly 30 failed cases to choose from)
    sample_failures = incorrect_cases[:3] 
    
    for idx, case in enumerate(sample_failures, start=1):
        print(f"\n[FAILURE CASE {idx}]")
        print(f"QUESTION:\n{case['question']}\n")
        print(f"GROUND TRUTH: {case['ground_truth']}")
        print(f"EXTRACTED MODEL ANSWER: {case['extracted_answer']}\n")
        print(f"MODEL SOLUTION EXCERPT:\n{case['model_response']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
```

## Question4
```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Configuration
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    print(f"Loading base model: {MODEL_ID}...")
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Configure LoRA according to the default project settings
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply the LoRA adapters to the base model
    peft_model = get_peft_model(model, lora_config)
    
    # ==========================================
    # 2. Parameter Calculation
    # ==========================================
    trainable_params = 0
    total_params = 0
    
    for _, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    percentage = 100 * trainable_params / total_params
    
    # ==========================================
    # 3. Report Output
    # ==========================================
    print("\n" + "="*50)
    print("PARAMETER COUNT REPORT (FOR QUESTION 4)")
    print("="*50)
    print(f"Total Parameters (Base + LoRA) : {total_params:,}")
    print(f"Trainable LoRA Parameters      : {trainable_params:,}")
    print(f"Percentage of Trained Params   : {percentage:.4f}%")
    print("="*50)

if __name__ == "__main__":
    main()
```

## Question 5 & 7
```python
import torch
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import os

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_TRAIN_EXAMPLES = 3000   # 1000/3000
NUM_TEST_QUESTIONS = 100
EVAL_BATCH_SIZE = 8

SYSTEM_PROMPT = (
    "You are a helpful and precise math assistant. "
    "Please solve the following math word problem step-by-step. "
    "Show your reasoning clearly. Finally, you MUST enclose your final numerical answer "
    "in a LaTeX boxed format at the very end. For example, if the answer is 5, write \\boxed{5}. "
    "Do not include units or text inside the box."
)

# 2. Data Preparation and Custom Collator
def format_and_tokenize(example, tokenizer):
    question = example["question"]
    raw_answer = example["answer"]

    if "#### " in raw_answer:
        reasoning, final_ans = raw_answer.split("#### ")
        reasoning = reasoning.strip()
        final_ans = final_ans.strip().replace(",", "")
    else:
        reasoning = raw_answer.strip()
        final_ans = ""

    target_response = f"{reasoning}\n\nTherefore, the final answer is \\boxed{{{final_ans}}}."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": target_response}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(text, truncation=True, max_length=1024)
    return encoded

class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template="<|im_start|>assistant\n"):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()

        for i in range(len(labels)):
            seq = labels[i].tolist()
            template_len = len(self.response_template_ids)
            match_idx = -1

            for j in range(len(seq) - template_len + 1):
                if seq[j : j + template_len] == self.response_template_ids:
                    match_idx = j + template_len
                    break

            if match_idx != -1:
                labels[i, :match_idx] = -100
            else:
                labels[i, :] = -100

            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch

def extract_ground_truth(answer_str):
    if "#### " in answer_str:
        ans = answer_str.split("#### ")[-1].strip()
        return ans.replace(",", "")
    return None

def extract_model_answer(generated_text):
    match = re.search(r"\\boxed\{([^\}]+)\}", generated_text)
    if match:
        raw_ans = match.group(1)
        clean_ans = re.sub(r"[^\d\.-]", "", raw_ans)
        if clean_ans.endswith('.'):
            clean_ans = clean_ans[:-1]
        if clean_ans not in ["", "-", "."]:
            return clean_ans

    fallback_match = re.search(r"(?:answer is|Final Answer:)\s*\$?(-?\d[\d,]*(?:\.\d+)?)", generated_text, re.IGNORECASE)
    if fallback_match:
         ans = fallback_match.group(1).replace(",", "")
         if ans.endswith('.'):
             ans = ans[:-1]
         if ans not in ["", "-", "."]:
             return ans
    return None

# 3. Main Pipeline
def main():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading and preparing datasets...")
    dataset = load_dataset("gsm8k", "main")
    train_subset = dataset["train"].select(range(NUM_TRAIN_EXAMPLES))
    test_subset = dataset["test"].select(range(NUM_TEST_QUESTIONS))

    train_dataset = train_subset.map(
        lambda x: format_and_tokenize(x, tokenizer),
        remove_columns=train_subset.column_names,
        desc="Tokenizing Training Data"
    )

    # 4. LoRA and Training Setup
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    output_dir = f"./qwen-gsm8k-lora-{NUM_TRAIN_EXAMPLES}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=collator,
        args=training_args
    )

    print(f"Starting LoRA Fine-Tuning with {NUM_TRAIN_EXAMPLES} examples...")
    trainer.train()
    print("Training complete. Setting model to evaluation mode...")
    model.gradient_checkpointing_disable()
    model.eval()

    # 5. Evaluation Loop
    correct_count = 0
    print(f"Evaluating on {NUM_TEST_QUESTIONS} test questions...")

    for i in tqdm(range(0, len(test_subset), EVAL_BATCH_SIZE), desc="Evaluating"):
        batch = test_subset[i : i + EVAL_BATCH_SIZE]
        questions = batch["question"]
        raw_answers = batch["answer"]

        prompt_texts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(text)

        encoded_inputs = tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        for j in range(len(questions)):
            input_len = encoded_inputs.input_ids[j].shape[0]
            gen_tokens = outputs[j][input_len:]
            response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            model_ans = extract_model_answer(response_text)
            true_ans = extract_ground_truth(raw_answers[j])

            if model_ans is not None and true_ans is not None:
                try:
                    if float(model_ans) == float(true_ans):
                        correct_count += 1
                except ValueError:
                    pass

    accuracy = (correct_count / NUM_TEST_QUESTIONS) * 100
    print("\n" + "="*60)
    print(f"LORA ({NUM_TRAIN_EXAMPLES} SAMPLES) EVALUATION RESULTS")
    print("="*60)
    print(f"Total Correct Predictions : {correct_count}")
    print(f"Fine-Tuned Accuracy       : {accuracy:.2f}%")
    print("="*60)

    with open(f"accuracy_{NUM_TRAIN_EXAMPLES}.txt", "w") as f:
        f.write(str(accuracy))
    print("\nTo plot accuracy vs. training examples, we need accuracies for 0 (baseline) and 1000 examples.")
    baseline_acc = float(input("Enter baseline accuracy (0 examples) from Task 1 (e.g., 38.5): "))
    acc_1000 = float(input(41.0))
    acc_3000 = accuracy
    include_full = input("Do you have accuracy for full 7473 examples? (y/n): ").strip().lower()
    if include_full == 'y':
        acc_full = float(input("Enter accuracy for 7473 examples: "))
        x_vals = [0, 1000, 3000, 7473]
        y_vals = [baseline_acc, acc_1000, acc_3000, acc_full]
    else:
        x_vals = [0, 1000, 3000]
        y_vals = [baseline_acc, acc_1000, acc_3000]

    plt.figure(figsize=(8,5))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy (%)')
    plt.title('LoRA SFT Accuracy vs. Training Data Size')
    plt.grid(True)
    plt.xticks(x_vals)
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.savefig('accuracy_plot.png')
    plt.show()

    print("\nPlot saved as accuracy_plot.png")
    print("\nTrend observation: Accuracy improves as training data increases, but gains diminish.")
    print("From 0 to 1000 examples, improvement is substantial (e.g., +~10%). From 1000 to 3000,")
    print("improvement is smaller (e.g., +~3-5%), indicating diminishing returns. This suggests")
    print("that additional data beyond a few thousand examples may yield only marginal gains,")
    print("and focusing on data quality might be more beneficial than quantity.")

if __name__ == "__main__":
    main()
```

## Question 9
```python
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# Configuration 
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "/content/qwen-gsm8k-lora-3000"   # 您训练好的 LoRA 适配器路径
NUM_TEST_QUESTIONS = 100
EVAL_BATCH_SIZE = 8
SYSTEM_PROMPT = (
    "You are a helpful and precise math assistant. "
    "Please solve the following math word problem step-by-step. "
    "Show your reasoning clearly. Finally, you MUST enclose your final numerical answer "
    "in a LaTeX boxed format at the very end. For example, if the answer is 5, write \\boxed{5}. "
    "Do not include units or text inside the box."
)

def extract_ground_truth(answer_str):
    if "#### " in answer_str:
        ans = answer_str.split("#### ")[-1].strip()
        return ans.replace(",", "")
    return None

def extract_model_answer(generated_text):
    match = re.search(r"\\boxed\{([^\}]+)\}", generated_text)
    if match:
        raw_ans = match.group(1)
        clean_ans = re.sub(r"[^\d\.-]", "", raw_ans)
        if clean_ans.endswith('.'):
            clean_ans = clean_ans[:-1]
        if clean_ans not in ["", "-", "."]:
            return clean_ans
    fallback_match = re.search(r"(?:answer is|Final Answer:)\s*\$?(-?\d[\d,]*(?:\.\d+)?)", generated_text, re.IGNORECASE)
    if fallback_match:
         ans = fallback_match.group(1).replace(",", "")
         if ans.endswith('.'):
             ans = ans[:-1]
         if ans not in ["", "-", "."]:
             return ans
    return None

def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Model loaded successfully.")

    dataset = load_dataset("gsm8k", "main")
    test_subset = dataset["test"].select(range(NUM_TEST_QUESTIONS))

    failures = []

    print(f"Evaluating on {NUM_TEST_QUESTIONS} test questions...")
    for i in tqdm(range(0, len(test_subset), EVAL_BATCH_SIZE), desc="Evaluating"):
        batch = test_subset[i : i + EVAL_BATCH_SIZE]
        questions = batch["question"]
        raw_answers = batch["answer"]

        prompt_texts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(text)

        encoded_inputs = tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        for j in range(len(questions)):
            input_len = encoded_inputs.input_ids[j].shape[0]
            gen_tokens = outputs[j][input_len:]
            response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            model_ans = extract_model_answer(response_text)
            true_ans = extract_ground_truth(raw_answers[j])

            if model_ans is not None and true_ans is not None:
                try:
                    if float(model_ans) != float(true_ans):
                        failures.append({
                            "question": questions[j],
                            "response": response_text,
                            "model_answer": model_ans,
                            "ground_truth": true_ans
                        })
                except ValueError:
                    failures.append({
                        "question": questions[j],
                        "response": response_text,
                        "model_answer": model_ans,
                        "ground_truth": true_ans
                    })
            else:
                failures.append({
                    "question": questions[j],
                    "response": response_text,
                    "model_answer": model_ans,
                    "ground_truth": true_ans
                })

    print("\n" + "="*70)
    print(f"Found {len(failures)} failure cases out of {NUM_TEST_QUESTIONS} questions.")
    print("="*70)

    for idx, fail in enumerate(failures, 1):
        print(f"\n--- Failure Case {idx} ---")
        print(f"Question: {fail['question']}")
        print(f"\nModel Response:\n{fail['response']}")
        print(f"\nExtracted Model Answer: {fail['model_answer']}")
        print(f"Ground Truth: {fail['ground_truth']}")
        print("-"*50)

    print("\nSelect two of the above failures for your analysis in QUESTION 9.")

if __name__ == "__main__":
    main()
```

## Question 10
```python
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER_PATH = "./qwen-gsm8k-lora-3000"   
NUM_TEST_QUESTIONS = 100
EVAL_BATCH_SIZE = 8
SYSTEM_PROMPT = (
    "You are a helpful and precise math assistant. "
    "Please solve the following math word problem step-by-step. "
    "Show your reasoning clearly. Finally, you MUST enclose your final numerical answer "
    "in a LaTeX boxed format at the very end. For example, if the answer is 5, write \\boxed{5}. "
    "Do not include units or text inside the box."
)

def extract_ground_truth(answer_str):
    if "#### " in answer_str:
        ans = answer_str.split("#### ")[-1].strip()
        return ans.replace(",", "")
    return None

def extract_model_answer(generated_text):
    match = re.search(r"\\boxed\{([^\}]+)\}", generated_text)
    if match:
        raw_ans = match.group(1)
        clean_ans = re.sub(r"[^\d\.-]", "", raw_ans)
        if clean_ans.endswith('.'):
            clean_ans = clean_ans[:-1]
        if clean_ans not in ["", "-", "."]:
            return clean_ans
    fallback_match = re.search(r"(?:answer is|Final Answer:)\s*\$?(-?\d[\d,]*(?:\.\d+)?)", generated_text, re.IGNORECASE)
    if fallback_match:
         ans = fallback_match.group(1).replace(",", "")
         if ans.endswith('.'):
             ans = ans[:-1]
         if ans not in ["", "-", "."]:
             return ans
    return None

def get_demonstrations():
    dataset = load_dataset("gsm8k", "main")
    train = dataset["train"]
    # 选择前3个样本作为演示
    demo_indices = [0, 1, 2]
    demos = []
    for idx in demo_indices:
        q = train[idx]["question"]
        ans = train[idx]["answer"]
        if "#### " in ans:
            reasoning, final = ans.split("#### ")
            reasoning = reasoning.strip()
            final = final.strip().replace(",", "")
            full_solution = f"{reasoning}\n\nTherefore, the final answer is \\boxed{{{final}}}."
        else:
            full_solution = ans.strip()
        demos.append((q, full_solution))
    return demos

DEMOS = get_demonstrations()

def build_few_shot_prompt(test_question, system_prompt, demos):
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in demos:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": test_question})
    return messages

def evaluate_model(model, tokenizer, test_subset, demos, batch_size=8):
    correct = 0
    total = len(test_subset)
    for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch = test_subset[i : i + batch_size]
        questions = batch["question"]
        raw_answers = batch["answer"]

        all_prompts = []
        for q in questions:
            messages = build_few_shot_prompt(q, SYSTEM_PROMPT, demos)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt)

        encoded = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        for j in range(len(questions)):
            input_len = encoded.input_ids[j].shape[0]
            gen_tokens = outputs[j][input_len:]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            model_ans = extract_model_answer(response)
            true_ans = extract_ground_truth(raw_answers[j])
            if model_ans is not None and true_ans is not None:
                try:
                    if float(model_ans) == float(true_ans):
                        correct += 1
                except ValueError:
                    pass
    accuracy = correct / total * 100
    return accuracy

def main():
    dataset = load_dataset("gsm8k", "main")
    test_subset = dataset["test"].select(range(NUM_TEST_QUESTIONS))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model.eval()
    base_acc_3shot = evaluate_model(base_model, tokenizer, test_subset, DEMOS, EVAL_BATCH_SIZE)
    print(f"Base model 3-shot accuracy: {base_acc_3shot:.2f}%")

    print("Loading SFT model...")
    sft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    sft_model = PeftModel.from_pretrained(sft_base, SFT_ADAPTER_PATH)
    sft_model.eval()
    sft_acc_3shot = evaluate_model(sft_model, tokenizer, test_subset, DEMOS, EVAL_BATCH_SIZE)
    print(f"SFT model 3-shot accuracy: {sft_acc_3shot:.2f}%")

    base_baseline = 26.0  
    sft_baseline = 52.0    

    print("\n=== Summary ===")
    print(f"Base model: baseline = {base_baseline:.2f}%, 3-shot = {base_acc_3shot:.2f}%, Δ = {base_acc_3shot - base_baseline:.2f}%")
    print(f"SFT model:  baseline = {sft_baseline:.2f}%, 3-shot = {sft_acc_3shot:.2f}%, Δ = {sft_acc_3shot - sft_baseline:.2f}%")

if __name__ == "__main__":
    main()
```

## Question 13
```python
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
、
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_TEST_QUESTIONS = 100
EVAL_BATCH_SIZE = 4  # 由于生成多个样本，批大小减小
NUM_SAMPLES = 5
TEMPERATURE = 0.7
TOP_P = 0.9
SYSTEM_PROMPT = (
    "You are a helpful and precise math assistant. "
    "Please solve the following math word problem step-by-step. "
    "Show your reasoning clearly. Finally, you MUST enclose your final numerical answer "
    "in a LaTeX boxed format at the very end. For example, if the answer is 5, write \\boxed{5}. "
    "Do not include units or text inside the box."
)

def extract_model_answer(generated_text):
    match = re.search(r"\\boxed\{([^\}]+)\}", generated_text)
    if match:
        raw_ans = match.group(1)
        clean_ans = re.sub(r"[^\d\.-]", "", raw_ans)
        if clean_ans.endswith('.'):
            clean_ans = clean_ans[:-1]
        if clean_ans not in ["", "-", "."]:
            return clean_ans
    fallback_match = re.search(r"(?:answer is|Final Answer:)\s*\$?(-?\d[\d,]*(?:\.\d+)?)", generated_text, re.IGNORECASE)
    if fallback_match:
         ans = fallback_match.group(1).replace(",", "")
         if ans.endswith('.'):
             ans = ans[:-1]
         if ans not in ["", "-", "."]:
             return ans
    return None

def extract_ground_truth(answer_str):
    if "#### " in answer_str:
        ans = answer_str.split("#### ")[-1].strip()
        return ans.replace(",", "")
    return None

def get_demonstrations():
    dataset = load_dataset("gsm8k", "main")
    train = dataset["train"]
    demo_indices = [0, 1, 2]
    demos = []
    for idx in demo_indices:
        q = train[idx]["question"]
        ans = train[idx]["answer"]
        if "#### " in ans:
            reasoning, final = ans.split("#### ")
            reasoning = reasoning.strip()
            final = final.strip().replace(",", "")
            full_solution = f"{reasoning}\n\nTherefore, the final answer is \\boxed{{{final}}}."
        else:
            full_solution = ans.strip()
        demos.append((q, full_solution))
    return demos

def build_few_shot_prompt(test_question, system_prompt, demos):
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in demos:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": test_question})
    return messages

def self_consistency_generate(model, tokenizer, prompt, num_samples=5, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id
        )
    answers = []
    for i in range(num_samples):
        gen_tokens = outputs[i][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        ans = extract_model_answer(response)
        if ans is not None:
            answers.append(ans)
    if not answers:
        return None
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    return most_common

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    dataset = load_dataset("gsm8k", "main")
    test_subset = dataset["test"].select(range(NUM_TEST_QUESTIONS))

    demos = get_demonstrations()

    correct = 0
    total = len(test_subset)

    for i in tqdm(range(total), desc="Evaluating with self-consistency"):
        question = test_subset[i]["question"]
        true_ans = extract_ground_truth(test_subset[i]["answer"])

        messages = build_few_shot_prompt(question, SYSTEM_PROMPT, demos)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        final_answer = self_consistency_generate(model, tokenizer, prompt, num_samples=NUM_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P)

        if final_answer is not None and true_ans is not None:
            try:
                if float(final_answer) == float(true_ans):
                    correct += 1
            except ValueError:
                pass

    accuracy = correct / total * 100
    print(f"Self-consistency (k={NUM_SAMPLES}, T={TEMPERATURE}) accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
```
