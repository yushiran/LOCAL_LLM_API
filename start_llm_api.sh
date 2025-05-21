#!/bin/bash

# Start Mistral API Server
# This script launches the Mistral-7B API server in the background and ensures it continues
# running even if the terminal is closed.

# Configuration
PORT=8000
LOG_FILE="./logs/mistral_api.log"
PID_FILE="./mistral_api.pid"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to check if service is already running
check_running() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
      echo "Mistral API is already running with PID $PID"
      echo "API available at http://$(hostname -I | awk '{print $1}'):$PORT"
      return 0
    else
      echo "Found stale PID file. Cleaning up..."
      rm "$PID_FILE"
    fi
  fi
  return 1
}


# Check if already running
if check_running; then
  exit 0
fi

GPU_ID=$?

# Start the server based on GPU availability
echo "Starting Mistral API Server..."

if [ $GPU_ID -eq 255 ]; then
  # No GPU, use CPU
  echo "Using CPU for inference"
  nohup python -u main.py --port $PORT > "$LOG_FILE" 2>&1 &
else
    echo "Using Python's GPU selection without token"
    nohup python -u main.py --port $PORT > "$LOG_FILE" 2>&1 &
fi

# Save PID for future reference
echo $! > "$PID_FILE"

# Give the server a moment to start
sleep 2

# Check if server started successfully
if ps -p $(cat "$PID_FILE") > /dev/null; then
  echo "Mistral API started successfully with PID $(cat "$PID_FILE")"
  echo "Log file: $LOG_FILE"
  IP_ADDR=$(hostname -I | awk '{print $1}')
  echo "API available at http://$IP_ADDR:$PORT"
  echo "Test the API with:"
  echo "curl -X POST http://$IP_ADDR:$PORT/v1/completions \\"
  echo "  -H 'Content-Type: application/json' \\"
  echo "  -d '{\"prompt\": \"Hello, how are you?\", \"max_new_tokens\": 100}'"
else
  echo "Failed to start Mistral API Server. Check logs at $LOG_FILE"
  exit 1
fi

# Stop server script
cat > ./stop_llm_api.sh << 'EOF'
#!/bin/bash
PID_FILE="./mistral_api.pid"
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if ps -p "$PID" > /dev/null; then
    echo "Stopping Mistral API (PID: $PID)..."
    kill $PID
    rm "$PID_FILE"
    echo "Mistral API stopped"
  else
    echo "Mistral API is not running (stale PID file)"
    rm "$PID_FILE"
  fi
else
  echo "Mistral API is not running (no PID file found)"
fi
EOF

chmod +x ./stop_llm_api.sh
echo "Created stop script: ./stop_llm_api.sh"
