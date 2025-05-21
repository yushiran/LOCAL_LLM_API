# Local Qwen3 API Server

This project provides an API server for running the Qwen3-14B language model locally. The API can be accessed from other devices on your internal network.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 16GB VRAM (recommended)
- CPU-only mode also supported (but will be much slower)

### Installation

1. Install dependencies using UV (recommended):
   ```bash
   # Install UV if you don't have it
   pip install uv
   # Install dependencies
   uv sync
   ```

   Alternatively, you can install dependencies directly:
   ```bash
   pip install fastapi uvicorn transformers torch accelerate bitsandbytes pydantic gputil streamlit
   ```

## Usage

### Starting the API Server

Run the following command to start the server:

```bash
./start_llm_api.sh
```

This script will:
1. Find the GPU with the most available memory
2. Start the API server in the background
3. Keep the server running even if you close the terminal
4. Create a log file in the `logs` directory

### Example Video

You can watch the following example video to see the API in action:

<video style="max-width: 100%; height: auto;" controls muted>
  <source src="https://github.com/user-attachments/assets/d25d0c4c-d5a8-43f5-aaa2-ab6cc697f784" type="video/mp4">
</video>

### Stopping the API Server

To stop the server:

```bash
./stop_llm_api.sh
```

### API Endpoints

#### Health Check
```
GET /
```

#### Chat Completion
```
POST /v1/chat/completions
```

Example request:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true,
  "enable_thinking": true,
  "system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
}
```

Example response:
```json
{
  "generated_text": "I'm doing well, thank you for asking! How can I help you today?",
  "model_name": "Qwen/Qwen3-14B",
  "thinking_content": "The user is greeting me and asking how I am. I should respond in a friendly manner..."
}
```

### Accessing from Other Devices

The API will be accessible from other devices on your network at:
```
http://<your-server-ip>:8000/v1/chat/completions
```

### Testing the API with Streamlit Client

You can use our built-in Streamlit client to test the API:

```bash
cd /home/uceesy4/workplace/LOCAL_LLM_API
./test/run_test_client.sh
```

This will launch a user-friendly web interface for interacting with the model.

## Configuration

The model can be configured in `config/config.toml`:
- `model_id`: Which model to use (default: "Qwen/Qwen3-14B")
- `HF_TOKEN`: Your HuggingFace token for downloading models

You can also modify the following in `start_llm_api.sh`:
- `PORT`: The port to run the API on (default: 8000)
- `LOG_FILE`: Location of the log file (default: ./logs/mistral_api.log)

## Troubleshooting

If you encounter any issues:
1. Check the log file in `logs/mistral_api.log`
2. Make sure your GPU has enough available memory (at least 16GB recommended)
3. Try running in CPU mode if no GPU is available
4. Verify your internet connection for downloading the model
5. Check that your HuggingFace token has permissions to access the Qwen model
