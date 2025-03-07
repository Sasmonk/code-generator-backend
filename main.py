from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load Model
model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Request Schema
class CodeRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

# API Endpoint
@app.post("/process")
def process_code(request: CodeRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
    return {"completed_code": tokenizer.decode(output[0], skip_special_tokens=True)}
