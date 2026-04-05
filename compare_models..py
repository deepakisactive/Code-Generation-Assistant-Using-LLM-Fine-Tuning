from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Model name
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 🔥 LOAD BASE MODEL (SEPARATELY)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# 🔥 LOAD ANOTHER BASE MODEL FOR FINE-TUNED
ft_base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
fine_tuned_model = PeftModel.from_pretrained(ft_base_model, "fine_tuned_model")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)
fine_tuned_model.to(device)

base_model.eval()
fine_tuned_model.eval()

# 🔥 Better prompts (more difference)
test_prompts = [
    "Fix the bug in this code: def factorial(n): return n * factorial(n-1)",
    "Convert this Python code to Java: def add(a,b): return a+b",
    "Write optimized Python code to remove duplicates from a list",
    "Create a FastAPI endpoint to upload a file"
]

def generate(model, prompt):
    formatted_prompt = f"""### Instruction:
{prompt}

### Response:
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False  # 🔥 deterministic (important)
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compare
with open("comparison_results.txt", "w", encoding="utf-8") as f:

    for prompt in test_prompts:
        base_output = generate(base_model, prompt)
        fine_output = generate(fine_tuned_model, prompt)

        result = f"""
==================================================
PROMPT:
{prompt}

BASE MODEL OUTPUT:
{base_output}

FINE-TUNED MODEL OUTPUT:
{fine_output}
==================================================
"""

        print(result)
        f.write(result)

print("\n✅ Comparison saved to comparison_results.txt")