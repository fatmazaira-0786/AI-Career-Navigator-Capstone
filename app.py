import streamlit as st
import os
import json
import time
from google import genai
from google.genai.errors import APIError
from io import StringIO

# --- 1. CONFIGURATION AND INITIAL SETUP ---
st.set_page_config(layout="wide", page_title="AI-Powered Career Navigator")

# Check for API Key in Streamlit Secrets
try:
    # Use st.secrets for Streamlit Cloud deployment
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    # Fallback for local development or if key is in environment variables
    API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in Streamlit Secrets or environment variables.")
    st.stop()

# Initialize the Gemini Client
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()

# Define the Google Search tool structure for grounding
google_search_tool = {"google_search": {}}

# --- 2. ROBUST API CALL FUNCTION (THE FIX) ---
@st.cache_data(show_spinner=False)
def safe_generate_content(client, model_name, contents, system_instruction, max_retries=5, tools=None):
    """
    Handles API calls with automatic retries on server errors (503, 504)
    to ensure the Streamlit app doesn't crash on temporary network issues.
    """
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
            # Catch API errors (503, 504) and automatically retry
            st.warning(f"Attempt {attempt + 1} failed (Server Busy/Timeout). Retrying in 5 seconds...")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                st.error("All retries failed. Please try again later.")
                raise # Re-raise the error if all attempts fail
        
        except Exception as e:
            st.error(f"An unexpected error occurred during API call: {e}")
            raise

    return None

# --- 3. AGENT FUNCTIONS (THE PIPELINE) ---

# --- AGENT 1: Resume Analyzer ---
def analyze_resume(uploaded_file, client):
    """Extracts skills, roles, and target career from a resume."""
    st.info("Agent 1 (Resume Analyzer) is processing your resume...")
    
    # 1. Read the file content
    string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    
    system_instruction = (
        "You are a professional Resume Analyst. Your task is to extract key information "
        "from the provided resume text. Respond ONLY with a clean JSON object containing "
        "the following keys: 'current_skills' (array of strings), 'current_roles' (array of strings), "
        "'target_career' (single string, the most likely next career path based on the resume)."
    )
    
    prompt = f"Analyze the following resume text and return the required JSON:\n\n---\n{string_data}"
    
    response = safe_generate_content(
        client,
        model_name="gemini-2.5-flash",
        contents=prompt,
        system_instruction=system_instruction,
        tools=None
    )
    
    try:
        # Assuming the model returns a parsable JSON string
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except Exception:
        st.error("Could not parse JSON output from Resume Analyzer. Using fallback data.")
        return {
            "current_skills": ["Python", "SQL", "Data Analysis"],
            "current_roles": ["Data Analyst", "Intern"],
            "target_career": "Senior Data Scientist"
        }

# --- AGENT 2: Market Researcher ---
def research_gaps(analysis_data, client):
    """Uses Google Search to find required skills and salary expectations."""
    st.info("Agent 2 (Market Researcher) is finding real-time job requirements...")
    
    target = analysis_data["target_career"]
    
    system_instruction = (
        f"You are a Career Market Researcher. Use Google Search to find the TOP 5 most "
        f"in-demand skills and the typical salary range (e.g., $120k - $180k) for a "
        f"'{target}' in the current market. Respond ONLY with a clean JSON object "
        f"with keys: 'required_skills' (array of strings) and 'salary_range' (string)."
    )
    
    prompt = f"Find the current market requirements for a {target}."
    
    response = safe_generate_content(
        client,
        model_name="gemini-2.5-pro", # Using Pro for better tool use and synthesis
        contents=prompt,
        system_instruction=system_instruction,
        tools=[google_search_tool]
    )
    
    try:
        # Check if the response contains function calls (if not implemented, assume direct text)
        if response and hasattr(response, 'function_calls') and response.function_calls:
            # Logic to handle function calls and tool output goes here
            # For simplicity, we assume the model directly returns the JSON after reasoning
            pass

        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except Exception:
        st.error("Could not parse JSON output from Market Researcher. Using fallback data.")
        return {
            "required_skills": ["Advanced LLMs", "Cloud Deployment (GCP)", "Prompt Engineering", "Vector Databases", "CI/CD"],
            "salary_range": "$140,000 - $200,000"
        }

# --- AGENT 3: Curriculum Designer ---
def design_curriculum(analysis_data, research_data, client):
    """Creates the final 6-month roadmap."""
    st.info("Agent 3 (Curriculum Designer) is synthesizing the final 6-Month Roadmap...")
    
    # 1. Prepare combined input data
    current_skills = ", ".join(analysis_data["current_skills"])
    required_skills = ", ".join(research_data["required_skills"])
    target = analysis_data["target_career"]
    
    # 2. Identify Skill Gaps
    gap_skills = [
        skill for skill in research_data["required_skills"] 
        if skill not in analysis_data["current_skills"]
    ]
    
    system_instruction = (
        "You are a world-class Curriculum Designer and Career Coach. "
        "Your task is to create a prescriptive, 6-Month Career Roadmap. "
        "The output MUST be a single, well-formatted Markdown document, not a JSON. "
        "Include sections for Summary, Skill Gaps, and the 6-Month Plan (Month 1 to Month 6)."
    )
    
    prompt = f"""
    --- USER PROFILE ---
    Target Career: {target} (Salary: {research_data["salary_range"]})
    Current Skills: {current_skills}
    Required Market Skills: {required_skills}
    Identified Skill Gaps: {', '.join(gap_skills) if gap_skills else 'None'}
    
    --- TASK ---
    Generate a 6-month curriculum (Roadmap) focused on closing the identified skill gaps. 
    Each month should have 3-4 specific, actionable learning items.
    """
    
    response = safe_generate_content(
        client,
        model_name="gemini-2.5-pro", # Use Pro for high-quality, long-form output
        contents=prompt,
        system_instruction=system_instruction,
        tools=None
    )
    
    return response.text if response else "Error: Could not generate roadmap."

# --- 4. STREAMLIT UI AND EXECUTION FLOW ---

st.title("AI-Powered Career Navigator")
st.markdown("A Multi-Agent Gemini System for generating personalized 6-Month Career Roadmaps.")

# Initialize session state for the roadmap results
if 'roadmap_result' not in st.session_state:
    st.session_state.roadmap_result = None

# Input: Resume Upload
uploaded_file = st.file_uploader(
    "Upload Your Resume (PDF or TXT) to Begin:", 
    type=["txt", "pdf"],
    accept_multiple_files=False
)

if uploaded_file:
    # Use the filename as a run identifier
    run_id = uploaded_file.name + str(uploaded_file.size)

    # Check if a previous run is already in cache
    if st.session_state.roadmap_result and st.session_state.roadmap_result.get('run_id') == run_id:
        st.success("Roadmap loaded from cache!")
        st.subheader("Your Personalized 6-Month Career Roadmap")
        st.markdown(st.session_state.roadmap_result['content'])
    else:
        # Start the pipeline when button is clicked
        if st.button("Generate My Roadmap (Takes ~30-60 seconds)", type="primary"):
            st.session_state.roadmap_result = None # Clear previous result
            with st.spinner("🚀 Running Multi-Agent Pipeline..."):
                try:
                    # AGENT 1
                    analysis_output = analyze_resume(uploaded_file, client)
                    
                    # AGENT 2
                    research_output = research_gaps(analysis_output, client)
                    
                    # AGENT 3
                    roadmap_markdown = design_curriculum(analysis_output, research_output, client)
                    
                    # Store result in session state
                    st.session_state.roadmap_result = {
                        'run_id': run_id,
                        'content': roadmap_markdown
                    }
                    
                    st.success("Roadmap Generation Complete!")
                    st.subheader("Your Personalized 6-Month Career Roadmap")
                    st.markdown(roadmap_markdown)
                
                except Exception as e:
                    st.error(f"A critical error stopped the pipeline: {e}")
                    st.session_state.roadmap_result = None
