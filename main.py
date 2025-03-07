from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load CodeLlama Model
model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Initialize FastAPI
app = FastAPI()

class CodeRequest(BaseModel):
    prompt: str
    max_length: int = 256

@app.post("/generate")
def generate_code(request: CodeRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=request.max_length)
    return {"generated_code": tokenizer.decode(outputs[0], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

