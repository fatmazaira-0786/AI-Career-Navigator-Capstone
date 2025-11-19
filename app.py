import streamlit as st
import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# --- Configuration and Initialization ---
# Ensure API Key is loaded from environment secrets (required for deployment)
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    st.error("Error: GEMINI_API_KEY environment variable not set. Please set the API key for deployment.")
    st.stop()
    
client = genai.Client(api_key=API_KEY)

# --- 1. SCHEMAS (From Code Cells 1 & 4) ---

class MarketResearchOutput(BaseModel):
    target_role: str = Field(description="The final job title.")
    core_required_skills: List[str] = Field(description="A list of 5-8 most critical hard skills.")
    in_demand_tools: List[str] = Field(description="A list of 3-5 specific tools or frameworks.")
    salary_range_usd: str = Field(description="The current average entry-level salary range.")
    top_3_career_gaps: List[str] = Field(description="The three biggest knowledge/experience gaps.")

class ResumeAnalysisOutput(BaseModel):
    candidate_name: str = Field(description="The full name of the candidate.")
    current_role: str = Field(description="The candidate's most recent or current job title.")
    total_experience_years: float = Field(description="The total estimated professional experience in years.")
    extracted_skills: List[str] = Field(description="A comprehensive list of all hard technical and soft skills.")

# --- 2. PYTHON LOGIC (From Code Cell 7) ---

def calculate_skill_gap(required_skills: List[str], existing_skills: List[str]) -> List[str]:
    required_set = {skill.strip().lower() for skill in required_skills}
    existing_set = {skill.strip().lower() for skill in existing_skills}
    gap_set = required_set - existing_set
    missing_skills = [s.title() for s in gap_set]
    return missing_skills

# --- 3. AGENT FUNCTIONS (From Code Cells 2 & 5) ---

@st.cache_data
def market_researcher_agent(starting_role: str, target_role: str) -> dict:
    # STEP 1: Search Agent (Tool Use for RAG)
    search_prompt = f"""
    Use the Google Search Tool to find the most current and in-demand skills, tools, salary range, and career gaps 
    for the transition from '{starting_role}' to '{target_role}'. Synthesize the search results into a detailed, 
    structured paragraph of text covering all necessary fields. DO NOT output JSON.
    """
    search_config = types.GenerateContentConfig(tools=[{"google_search": {}}])
    search_response = client.models.generate_content(
        model='gemini-2.5-pro', contents=[search_prompt], config=search_config
    )
    raw_search_context = search_response.text

    # STEP 2: Structuring Agent (JSON Constraint)
    structuring_prompt = f"""
    Extract the required fields from the market research context provided below and format it perfectly into the required JSON schema.
    Market Research Context: --- {raw_search_context} ---
    """
    structuring_config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=MarketResearchOutput
    )
    structure_response = client.models.generate_content(
        model='gemini-2.5-flash', contents=[structuring_prompt], config=structuring_config
    )
    return json.loads(structure_response.text)

@st.cache_data
def resume_analyzer_agent(resume_text: str) -> dict:
    prompt = f"""
    You are an expert resume parsing agent. Accurately extract key information from the provided raw resume text and normalize the skills into a common list. 
    The raw resume text is provided below: --- {resume_text} ---
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=ResumeAnalysisOutput
    )
    response = client.models.generate_content(
        model='gemini-2.5-flash', contents=[prompt], config=config
    )
    return json.loads(response.text)

@st.cache_data
def curriculum_designer_agent(candidate_data: dict, market_data: dict, skill_gap: List[str]) -> str:
    context_prompt = f"""
    You are the Curriculum Designer Agent. Your goal is to synthesize the following data into a comprehensive, 6-month, week-by-week learning plan designed to close the 'Skill Gap'. 
    [Context removed for brevity - includes Candidate Data, Market Data, and Skill Gap lists].
    Create a detailed 6-Month Career Transition Roadmap. 
    1. Structure the plan by **Months and Weeks**.
    2. Dedicate the first 4 months to learning the **Missing Skills** and **In-Demand Tools**.
    3. Dedicate the last 2 months to **Capstone Projects** and **Interview Prep**.
    4. Format the entire output as a single, beautiful **Markdown** response.
    """
    # Note: Using st.cache_data requires that the prompt generation be clean.
    # The full, detailed prompt from Code Cell 8 should be pasted here in the real app.
    
    response = client.models.generate_content(
        model='gemini-2.5-pro', contents=[context_prompt]
    )
    return response.text

# --- 4. STREAMLIT FRONTEND (Orchestration) ---

st.set_page_config(page_title="Personalized Career Navigator Agent", layout="wide")
st.title("🗺️ AI-Powered Career Navigator")
st.markdown("---")

# Input Columns
col1, col2 = st.columns([1, 1])

with col1:
    current_role = st.text_input("1. Your Current Role (e.g., Data Analyst)", "Data Analyst")
    target_role = st.text_input("2. Your Target Role (e.g., AI Engineer)", "AI Engineer")

with col2:
    resume_text = st.text_area(
        "3. Paste Your Resume/Experience Summary Here (Required)", 
        "5 years experience. Proficient in Python, SQL, Tableau, and team management. Led a small data team.",
        height=150
    )

st.markdown("---")

if st.button("🚀 Generate Personalized Career Roadmap", type="primary") and resume_text:
    
    with st.spinner("🧠 Analyzing Resume and Researching Market Trends..."):
        
        # 1. RUN AGENT 2 (Resume Analyzer)
        candidate_output = resume_analyzer_agent(resume_text)

        # 2. RUN AGENT 1 (Market Researcher)
        market_output = market_researcher_agent(current_role, target_role)

        # 3. PYTHON LOGIC (Skill Gap Calculation)
        required_skills_list = market_output.get('core_required_skills', [])
        existing_skills_list = candidate_output.get('extracted_skills', [])
        skill_gap_list = calculate_skill_gap(required_skills_list, existing_skills_list)

    st.success("Analysis Complete! Generating Final Roadmap...")
    
    # Display the structured data used for the plan
    st.subheader("📊 Skill Gap Analysis Snapshot")
    st.markdown(f"**Target Role:** {market_output.get('target_role')}")
    st.markdown(f"**Existing Skills:** {', '.join(existing_skills_list[:5])}...")
    st.markdown(f"**🛑 Critical Missing Skills:** {', '.join(skill_gap_list)}")
    st.markdown("---")

    with st.spinner("✨ Designing Curriculum..."):
        # 4. RUN AGENT 3 (Curriculum Designer)
        final_roadmap = curriculum_designer_agent(candidate_output, market_output, skill_gap_list)
        
    st.header("✅ Your Personalized 6-Month Career Roadmap")
    st.markdown(final_roadmap)
    st.balloons()
