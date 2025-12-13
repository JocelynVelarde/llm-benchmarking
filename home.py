import streamlit as st
from google import genai
import time
import pandas as pd
import plotly.express as px
from groq import Groq

st.set_page_config(page_title="LLM Benchmarking", layout="wide")

st.title("LLM Benchmarking")
st.subheader("Compare as many LLMs as you want side by side")
st.divider()

client = genai.Client(api_key="")
groq_client = Groq(api_key="")

def call_gemini(prompt):
    start = time.time()
    response = client.models.generate_content(model="gemini-2.5-flash", 
                                              contents=prompt)
    end = time.time()
    if response.usage_metadata:
        token_count = response.usage_metadata.total_token_count
    else:
        token_count = len(response.text) // 4
    return response.text, end - start, token_count

def call_llama(prompt):
    start = time.time()
    response_groq = groq_client.chat.completions.create(model='llama-3.1-8b-instant', 
                                                        messages=[{"role": "user", "content": prompt}], temperature=0.5)
    end = time.time()
    content = response_groq.choices[0].message
    token_count = response_groq.usage.total_tokens
    return content, end - start, token_count

with st.sidebar:
    st.title("Choose models")
    use_gemini = st.checkbox("Gemini 2.5 Flash", value=True)
    use_groq = st.checkbox("Llama-3.1", value=True)

prompt = st.chat_input("Enter your prompt")

if prompt:
    comparisions = []
    if use_gemini:
        comparisions.append("Gemini 2.5 Flash")
    if use_groq:
        comparisions.append("Llama 3.1")
    
    cols = st.columns(len(comparisions))
    results = []
    
    for i, comparision_name in enumerate(comparisions):
        with cols[i]:
            st.subheader(comparision_name)
            if comparision_name == "Gemini 2.5 Flash":
                content, latency, tokens = call_gemini(prompt)
            else:
                content, latency, tokens = call_llama(prompt)
