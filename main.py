import os
import argparse
import logging
from typing import List, Optional, Dict, Any
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from huggingface_hub import login
import uvicorn
from utils import get_local_ip, find_available_gpu
import toml
config_path = os.path.join(os.path.dirname(__file__), "config", "config.toml")
config = toml.load(config_path)

model_id = config.get('model_id','Qwen/Qwen3-14B')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title=f"Local {model_id} API", description=f"API for serving {model_id} model locally")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
generator = None

class ChatMessage(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 32768
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True
    enable_thinking: Optional[bool] = True

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: Optional[int] = 32768
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True
    enable_thinking: Optional[bool] = True
    system_prompt: Optional[str] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

class CompletionResponse(BaseModel):
    generated_text: str
    model_name: str
    thinking_content: Optional[str] = None

def load_model(model_name=model_id, gpu_id=None):
    """Load the Qwen model and tokenizer"""
    global model, tokenizer, generator
    
    logger.info(f"Loading model {model_name}...")
    
    # Configure device map
    if gpu_id is not None:
        device_map = {"": f"cuda:{gpu_id}"}
    else:
        device_map = "auto"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype="auto",
    )    
    logger.info(f"Model loaded successfully on {'GPU ' + str(gpu_id) if gpu_id is not None else 'CPU'}")

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"status": "ok", "model": model_id}

@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest):
    """Generate text completion given a prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Direct approach using model.generate() for more control
        messages = [{"role": "user", "content": request.prompt}]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking
        )
        
        # Tokenize the input
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate text
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Extract only the newly generated tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content if enabled
        thinking_content = None
        if request.enable_thinking:
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            except ValueError:
                # No thinking token found
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        else:
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        logger.info(f"Generated text: {content}")
        
        return CompletionResponse(
            generated_text=content,
            model_name=model_id,
            thinking_content=thinking_content
        )
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=CompletionResponse)
def create_chat_completion(request: ChatCompletionRequest):
    """Generate chat completion given a series of messages"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare messages, adding system message if provided
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        # Add user messages
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking
        )
        
        # Tokenize the input
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate text
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Extract only the newly generated tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content if enabled
        thinking_content = None
        if request.enable_thinking:
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            except ValueError:
                # No thinking token found
                content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        else:
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        logger.info(f"Generated chat response: {content}")
        
        return CompletionResponse(
            generated_text=content,
            model_name=model_id,
            thinking_content=thinking_content
        )
    except Exception as e:
        logger.error(f"Error generating chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main function to start the API server"""
    parser = argparse.ArgumentParser(description=f"{model_id} API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (default: None for CPU)")
    args = parser.parse_args()
    
    # Handle Hugging Face authentication
    hf_token = config.get("HF_TOKEN")
    if hf_token:
        logger.info("Authenticating with Hugging Face...")
        login(token=hf_token)
    else:
        logger.warning("No Hugging Face token provided. If the model is gated, loading will fail.")
        logger.warning("You can provide a token using --token or by setting the HF_TOKEN environment variable")
    
    # Find available GPU if not specified
    gpu_id = args.gpu
    if gpu_id is None:
        gpu_id = find_available_gpu()
    
    # Load the model
    load_model(gpu_id=gpu_id)
    
    # Display server information
    local_ip = get_local_ip()
    logger.info(f"Starting server at http://{local_ip}:{args.port}")
    logger.info(f"API will be available at http://{local_ip}:{args.port}/v1/completions")
    logger.info(f"Chat API will be available at http://{local_ip}:{args.port}/v1/chat/completions")
    
    # Import and run uvicorn here to avoid any initialization issues
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()