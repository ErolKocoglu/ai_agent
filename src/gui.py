import streamlit as st
import sys
import os

# Ensure src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.builder import build_agent

# Page Config
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI Agent with Tools")

# Initialize Agent (Cached)
@st.cache_resource
def get_agent():
    return build_agent()

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Failed to initialize agent: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.invoke(prompt)
                output_text = response.get("output", str(response))
                st.markdown(output_text)
                
                # Add agent message to history
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")
