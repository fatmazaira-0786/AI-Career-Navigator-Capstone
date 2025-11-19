import time
from google.genai.errors import APIError
# Make sure you have 'import streamlit as st' at the top of your app.py

def safe_generate_content(client, model_name, contents, system_instruction, max_retries=5, tools=None):
    """
    Handles API calls with automatic retries on server errors (503, 504)
    to ensure the Streamlit app doesn't crash on temporary network issues.
    """
    # NOTE: You must ensure 'st' is defined globally or imported in your main app.py
    st.info(f"Agent running: {system_instruction.split(' ')[2]}... Retries are enabled for stability.")
    
    # Structure the API payload correctly
    payload = {
        "model": model_name,
        "contents": [{"parts": [{"text": contents}]}],
        "config": {
            "system_instruction": system_instruction,
            "tools": tools if tools else [],
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=payload["model"],
                contents=payload["contents"],
                config=payload["config"]
            )
            # If successful, return the response
            return response
        
        except APIError as e:
            # If the server is busy, wait and try again
            st.warning(f"Attempt {attempt + 1} failed (Server Busy/Timeout). Retrying in 5 seconds...")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                st.error("All retries failed. Please try again later.")
                raise
        
        except Exception as e:
            st.error(f"An unexpected error occurred during API call: {e}")
            raise

    return None
