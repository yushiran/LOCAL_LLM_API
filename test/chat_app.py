#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit application for testing the chat functionality of Qwen3 API
"""

import streamlit as st
import requests
import json
import os
import sys

# Add project root directory to Python path to import functions from main module (if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="API Test",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set API URL
DEFAULT_API_URL = "http://localhost:8000"

# Sidebar configuration
with st.sidebar:
    st.title("API Test")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    
    # Model parameters configuration
    st.subheader("Model Parameters")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Maximum Generated Tokens", min_value=1024, max_value=32768, value=32768, step=64)
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
    enable_thinking = st.checkbox("Enable Thinking Mode", value=True)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt", 
        value="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        height=100
    )
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}")
            if response.status_code == 200:
                st.success("Connection successful!")
                st.json(response.json())
            else:
                st.error(f"Connection failed: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

# Main interface - Chat window
st.title("🤖 Qwen3 Chat Test")

# Initialize session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Please enter your question"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show AI thinking
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call API
            headers = {
                "Content-Type": "application/json"
            }
            
            # Prepare message list
            api_messages = []
            for msg in st.session_state.messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Build request
            payload = {
                "messages": api_messages,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "enable_thinking": enable_thinking,
                "system_prompt": system_prompt
            }
            
            # Send request
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("generated_text", "No response")
                thinking_content = result.get("thinking_content", None)
                
                # Display thinking process (if available)
                if thinking_content:
                    with st.expander("Model's thinking process"):
                        st.markdown(thinking_content)
                
                # Update display content
                message_placeholder.markdown(ai_response)
                
                # Add AI response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_response
                })
            else:
                error_msg = f"API call failed: Status code {response.status_code}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg
            })

# Add clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

# Add footer
st.markdown("---")
st.markdown("**Qwen3 API Test Client** | Built with Streamlit")
