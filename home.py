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