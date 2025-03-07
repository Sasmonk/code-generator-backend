import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Model Name
model_name = "codellama/CodeLlama-7b-Instruct-hf"

# Force CPU Execution
device = torch.device("cpu")  

# Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map={"": "cpu"})

class CodeRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

@app.post("/process")
def process_code(request: CodeRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
    return {"completed_code": tokenizer.decode(output[0], skip_special_tokens=True)}
