import streamlit as st
from config.settings import SystemConfig

# Check if API key is set
if SystemConfig.GEMINI_API_KEY == "your-gemini-api-key-here":
    st.error("No API key is set. Please set the GEMINI_API_KEY environment variable.")
else:
    # Rest of the app code here
    pass