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
