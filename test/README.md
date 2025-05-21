# Mistral-7B API Test Client

This is a simple web application built with Streamlit to test the dialogue functionality of the Mistral-7B API.

## Features

- Intuitive chat interface
- Configurable API parameters (temperature, maximum tokens, top-p value)
- Real-time API responses
- Conversation history
- Connection test functionality

## Installation Dependencies

Before using this test application, please ensure that you have installed the required dependencies:

```bash
pip install streamlit requests
```

## Running the Application

1. Make sure the Mistral-7B API service is running (using the `start_llm_api.sh` script)
2. Run the following command to start the Streamlit application:

```bash
cd /home/uceesy4/workplace/LOCAL_LLM_API
streamlit run test/chat_app.py
```

3. Access the Streamlit application in your browser (usually at http://localhost:8501)

## Usage Instructions

1. Set the API URL in the sidebar (default is http://localhost:8000)
2. Adjust the model parameters to change the characteristics of the generated text
3. Use the "Test Connection" button to verify that the API connection is working properly
4. Enter questions or prompts in the chat input box
5. View Mistral-7B's responses
6. You can use the "Clear Conversation" button to reset the conversation history

## Troubleshooting

- If the connection test fails, make sure the Mistral-7B API service is running
- Check if the API URL is set correctly
- If the generated text is truncated, try increasing the "Maximum Generated Tokens"
