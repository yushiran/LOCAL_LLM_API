#!/bin/bash

# Run M        echo "Mistral API is already running (PID: $PID)"
        echo "API access address: http://$(hostname -I | awk '{print $1}'):8000"
    fi
fi

if [ "$API_RUNNING" = false ]; then
    echo "Warning: Mistral API service does not seem to be running."
    echo "You can run ./start_llm_api.sh first to start the API service."
    echo "Do you still want to start the test client? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Cancelled. Please start the API service first, then run the test client."
        exit 1
    fi
fi

# Start Streamlit application
echo "Starting Streamlit test client..."
echo "When finished, please access the displayed URL in your browser (usually http://localhost:8501)"
streamlit run test/chat_app.pylient
# This script starts a Streamlit application to test the Mistral-7B API

# Set working directory
cd "$(dirname "$0")/.."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing now..."
    pip install streamlit requests
fi

# Check if API service is running
API_RUNNING=false
if [ -f "./mistral_api.pid" ]; then
    PID=$(cat "./mistral_api.pid")
    if ps -p "$PID" > /dev/null; then
        API_RUNNING=true
        echo "Mistral API is already running (PID: $PID)"
        echo "API访问地址: http://$(hostname -I | awk '{print $1}'):8000"
    fi
fi

if [ "$API_RUNNING" = false ]; then
    echo "警告: Mistral API服务似乎没有运行。"
    echo "您可以先运行 ./start_llm_api.sh 来启动API服务。"
    echo "是否仍然要启动测试客户端？ (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "已取消。请先启动API服务，然后再运行测试客户端。"
        exit 1
    fi
fi

# 启动Streamlit应用
echo "正在启动Streamlit测试客户端..."
echo "完成后，请在浏览器中访问显示的URL（通常是 http://localhost:8501）"
streamlit run test/chat_app.py
