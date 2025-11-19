# 🌟 AI-Powered Career Navigator: A Multi-Agent System Capstone

## Project Overview

This project was developed as the capstone for the Kaggle AI Intensive competition. The goal is to provide **personalized, real-time career transition planning** by overcoming the limitations of generic, static advice.

The system acts as a smart Concierge Agent, orchestrating three specialized AI components to perform a deep **Skill Gap Analysis** and generate a detailed learning roadmap. 

## 🛠️ Key Technologies

* **Google Gemini API:** Utilized `gemini-2.5-pro` for complex reasoning and `gemini-2.5-flash` for high-speed data structuring.
* **Dynamic RAG:** Integrated the **Google Search Tool** to fetch current market requirements, ensuring the advice is based on up-to-date industry trends.
* **Multi-Agent System:** Three agents (Analyzer, Researcher, Designer) communicate via structured data.
* **Pydantic:** Used to enforce **strict JSON schema** for clean, reliable data exchange between all system components.

## 🧠 System Workflow

The Career Navigator executes a four-step pipeline:

1.  **Resume Analyzer:** Extracts existing skills from raw text into structured JSON.
2.  **Market Researcher:** Uses RAG to find required skills for the target role and structures the market data.
3.  **Python Logic:** Calculates the precise skill gap (Required - Existing).
4.  **Curriculum Designer:** Synthesizes all structured data into a detailed, formatted 6-month Markdown roadmap.

## 🔗 Repository Contents

* `agent_system.py`: (Or your main file name) Contains all agent functions and Pydantic schemas.
* `README.md`: This project documentation file.
* `LICENSE`: (If you add one, recommend MIT for open source).

## 🚀 View the Full Project and Demonstration

The full execution, output, and detailed documentation are available in the public Kaggle Notebook. This notebook contains the final output and a walkthrough of the code.

| Component | Link |
| :--- | :--- |
| **Kaggle Notebook** | **[PASTE YOUR KAGGLE NOTEBOOK PUBLIC URL HERE]** |
| **Full Source Code** | **[PASTE THE URL OF THIS GITHUB REPOSITORY HERE]** |

***
