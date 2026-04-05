from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import evaluate
import nltk

# Download tokenizer
nltk.download('punkt')

# Base model
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load fine-tuned model
model = PeftModel.from_pretrained(base_model, "fine_tuned_model")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Generate response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test samples (increase for better evaluation)
test_samples = [
    {"input": "Write a Python function to check palindrome", "output": "def is_palindrome(s): return s == s[::-1]"},
    {"input": "Write a Python function to reverse string", "output": "def reverse_string(s): return s[::-1]"},
    {"input": "Write a Python function for factorial", "output": "def factorial(n): return 1 if n==0 else n*factorial(n-1)"},
    {"input": "Write a Python program to print Hello World", "output": "print('Hello World')"},
    {"input": "Write a Python function to find max element in list", "output": "def max_element(lst): return max(lst)"}
]

predictions = []
references = []
correct = 0

# Evaluation loop
for sample in test_samples:
    prompt = f"""### Instruction:
{sample['input']}

### Response:
"""

    pred = generate_response(prompt)

    predictions.append(pred)
    references.append(sample["output"])

    # Accuracy check (exact match)
    if pred.strip() == sample["output"].strip():
        correct += 1

    print("\n🔹 PROMPT:", sample["input"])
    print("🔹 GENERATED:", pred)
    print("🔹 EXPECTED:", sample["output"])

# Metrics
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
rouge_score = rouge.compute(predictions=predictions, references=references)

accuracy = correct / len(test_samples)

# Print results
print("\n📊 FINAL RESULTS")
print("BLEU:", bleu_score)
print("ROUGE:", rouge_score)
print("ACCURACY:", accuracy)

# Save results
with open("results.txt", "w") as f:
    f.write("BLEU Score:\n")
    f.write(str(bleu_score) + "\n\n")
    
    f.write("ROUGE Score:\n")
    f.write(str(rouge_score) + "\n\n")

    f.write("Accuracy:\n")
    f.write(str(accuracy) + "\n")