import streamlit as st
import os
import json
import time
from google import genai
# Import the custom error class safely
try:
    from google.genai.errors import APIError
except ImportError:
    from google.api_core.exceptions import GoogleAPICallError as APIError

# --- 1. CONFIGURATION AND INITIAL SETUP ---
st.set_page_config(layout="wide", page_title="AI-Powered Career Navigator")

# Check for API Key in Streamlit Secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, AttributeError):
    API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in Streamlit Secrets or environment variables.")
    st.stop()

# Initialize the Gemini Client
try:
    # Initialize client globally
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()

# --- GLOBAL SESSION STATE INITIALIZATION ---
if 'roadmap_result' not in st.session_state:
    st.session_state.roadmap_result = None 
# --- END GLOBAL SETUP ---


# Define the Google Search tool structure for grounding
google_search_tool = {"google_search": {}}

# --- 2. ROBUST API CALL FUNCTION (STABILITY AND CACHE FIX) ---
@st.cache_data(show_spinner=False)
# FIX: Removed the non-hashable '_client' argument from the cached function signature.
# The cache key is now only based on hashable inputs (model_name, contents, etc.).
def safe_generate_content(model_name, contents, system_instruction, max_retries=7, tools=None):
    """
    Handles API calls with automatic retries. Accesses the global 'client' object.
    """
    # Access the global client here. This is safe because this function is only run on cache miss.
    global client 
    
    st.info(f"Agent running: Model '{model_name}' called. More built-in retries for stability.")
    
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
            # Use the global client object
            response = client.models.generate_content(
                model=payload["model"],
                contents=payload["contents"],
                config=payload["config"]
            )
            return response
        
        except APIError as e:
            # Check for API errors that might be server-side (503, 504) or rate limit (429)
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed (Server Busy/Quota). Retrying in 8 seconds...")
                time.sleep(8)
            else:
                st.error(f"All retries failed after APIError: {e}. Please try again later.")
                raise
        
        except Exception as e:
            st.error(f"An unexpected error occurred during API call: {e}")
            raise

    return None

# --- 3. AGENT FUNCTIONS (PIPELINE) ---

# --- AGENT 1: Resume Analyzer ---
# NOTE: This function is NOT cached because its result depends on the non-deterministic resume_text_input 
# and we want it to run first to set up the pipeline.
def analyze_resume(resume_text):
    """Extracts skills, roles, and target career from resume text."""
    st.info("Agent 1 (Resume Analyzer) is processing your resume summary...")
    
    system_instruction = (
        "You are a professional Resume Analyst. Your task is to extract key information "
        "from the provided resume text. Respond ONLY with a clean JSON object containing "
        "the following keys: 'current_skills' (array of strings), 'current_roles' (array of strings), "
        "'target_career' (single string, the most likely next career path based on the resume)."
    )
    
    prompt = f"Analyze the following resume text and return the required JSON:\n\n---\n{resume_text}"
    
    # FIX: No need to pass 'client' here. safe_generate_content is now cached and uses the global client.
    response = safe_generate_content(
        model_name="gemini-2.5-flash-preview-09-2025", 
        contents=prompt,
        system_instruction=system_instruction,
        tools=None
    )
    
    try:
        # Clean the response text by removing markdown code blocks
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
@st.cache_data(show_spinner=False)
def research_gaps(analysis_data, current_role, target_role):
    """Uses Google Search to find required skills and salary expectations, cache based on role inputs."""
    st.info("Agent 2 (Market Researcher) is finding real-time job requirements...")
    
    target = target_role 
    
    system_instruction = (
        f"You are a Career Market Researcher. Use Google Search to find the TOP 5 most "
        f"in-demand skills and the typical salary range (e.g., $120k - $180k) for a "
        f"'{target}' in the current market. Respond ONLY with a clean JSON object "
        f"with keys: 'required_skills' (array of strings) and 'salary_range' (string)."
    )
    
    prompt = f"Find the current market requirements for a {target}."
    
    # FIX: safe_generate_content call simplified
    response = safe_generate_content(
        model_name="gemini-2.5-flash-preview-09-2025", 
        contents=prompt,
        system_instruction=system_instruction,
        tools=[google_search_tool]
    )
    
    try:
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except Exception:
        st.error("Could not parse JSON output from Market Researcher. Using fallback data.")
        return {
            "required_skills": ["Advanced LLMs", "Cloud Deployment (GCP)", "Prompt Engineering", "Vector Databases", "CI/CD"],
            "salary_range": "$140,000 - $200,000"
        }

# --- AGENT 3: Curriculum Designer ---
@st.cache_data(show_spinner=False)
def design_curriculum(analysis_data, research_data, current_role, target_role):
    """Creates the final 6-month roadmap, cache based on role inputs."""
    st.info("Agent 3 (Curriculum Designer) is synthesizing the final 6-Month Roadmap...")
    
    # 1. Prepare combined input data
    current_skills = ", ".join(analysis_data["current_skills"])
    required_skills = ", ".join(research_data["required_skills"])
    target = target_role
    
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
    Current Career: {current_role}
    Target Career: {target} (Salary: {research_data["salary_range"]})
    Current Skills: {current_skills}
    Required Market Skills: {required_skills}
    Identified Skill Gaps: {', '.join(gap_skills) if gap_skills else 'None'}
    
    --- TASK ---
    Generate a 6-month curriculum (Roadmap) focused on closing the identified skill gaps. 
    Each month should have 3-4 specific, actionable learning items.
    """
    
    # FIX: safe_generate_content call simplified
    response = safe_generate_content(
        model_name="gemini-2.5-flash-preview-09-2025", 
        contents=prompt,
        system_instruction=system_instruction,
        tools=None
    )
    
    return response.text if response else "Error: Could not generate roadmap."

# --- 4. STREAMLIT UI AND EXECUTION FLOW ---

st.title("AI-Powered Career Navigator")
st.markdown("A Multi-Agent Gemini System for generating personalized 6-Month Career Roadmaps.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Your Current Role (e.g., Data Analyst)**")
    current_role_input = st.text_input("", "Data Analyst", label_visibility="collapsed", key="current_role")
    
    st.markdown("**Your Target Role (e.g., AI Engineer)**")
    target_role_input = st.text_input("", "AI Engineer", label_visibility="collapsed", key="target_role")
    
with col2:
    st.markdown("**Paste Your Resume/Experience Summary Here (Required)**")
    resume_text_input = st.text_area(
        "", 
        "5 years experience. Proficient in Python, SQL, Tableau, and team management. Led a small data team.", 
        height=150, 
        label_visibility="collapsed",
        key="resume_summary"
    )

if st.button("Generate Personalized Career Roadmap", type="primary"):
    
    # Clear cache for all functions if a new run is initiated
    st.cache_data.clear()
    
    if not resume_text_input.strip():
        st.error("Please provide your Resume/Experience Summary in the text area to begin.")
        st.stop()
        
    # --- The main pipeline logic ---
    
    # Use the combination of the text inputs as the unique cache ID for the whole run
    run_id = current_role_input + target_role_input + resume_text_input
    
    # Reset session state if inputs change
    if st.session_state.roadmap_result and st.session_state.roadmap_result.get('run_id') != run_id:
        st.session_state.roadmap_result = None 

    # Start the pipeline 
    with st.spinner("🚀 Running Multi-Agent Pipeline..."):
        try:
            # AGENT 1 - No client argument needed
            analysis_output = analyze_resume(resume_text_input)
            
            # AGENT 2 - No client argument needed
            research_output = research_gaps(
                analysis_output, 
                current_role_input, 
                target_role_input 
            )
            
            # AGENT 3 - No client argument needed
            roadmap_markdown = design_curriculum(
                analysis_output, 
                research_output, 
                current_role_input, 
                target_role_input 
            )
            
            # Store result in session state
            st.session_state.roadmap_result = {
                'run_id': run_id,
                'content': roadmap_markdown
            }
            
            st.success("Roadmap Generation Complete!")
            
        except Exception as e:
            st.error(f"A critical error stopped the pipeline: {e}")
            st.session_state.roadmap_result = None

# --- Display Results ---
if st.session_state.roadmap_result and st.session_state.roadmap_result.get('content'):
    st.subheader("Your Personalized 6-Month Career Roadmap")
    st.markdown(st.session_state.roadmap_result['content'])
