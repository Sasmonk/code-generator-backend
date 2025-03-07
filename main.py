import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Get PORT from Render's environment
PORT = int(os.getenv("PORT", 8000))  # Default to 8000 if not set

# Load Model
model_name = "codellama/CodeLlama-7b-Instruct-hf"
device = torch.device("cpu")  # Render Free Tier does not have a GPU
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

# Run FastAPI on the correct port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
