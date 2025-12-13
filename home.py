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
groq_client = Groq()