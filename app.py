from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

# Enable CORS (IMPORTANT for UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class PromptRequest(BaseModel):
    prompt: str

# Load model
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, "fine_tuned_model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

@app.get("/")
def home():
    return {"message": "Code Generation API Running 🚀"}

@app.post("/generate")
def generate_code(request: PromptRequest):
    formatted_prompt = f"""### Instruction:
{request.prompt}

### Response:
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": result}