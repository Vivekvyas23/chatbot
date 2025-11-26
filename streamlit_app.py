import streamlit as st
import os
import re
import traceback
from typing import TypedDict, List, Union, Dict, Any, Optional
import inspect
import subprocess
import sys

# --- Robustness: Pre-import common standard libraries ---
# This acts as a safety net. If the generated agent forgets to import these,
# they will still be available in the execution environment.
import datetime
import json
import math
import random
import time
import uuid

from pydantic import BaseModel, Field

# --- Third-party Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="LangGraph Agent Builder", layout="wide")

st.title("🤖 LangGraph Meta-Agent Builder")
st.markdown("""
This agent builds other agents! Describe what you want, and the pipeline will:
1. **Analyze Requirements** 📝
2. **Draft Initial Code** 🏗️
3. **Identify APIs** 🔌
4. **Finalize & Polish Code** ✨
5. **Generate Standalone App** 🚀
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    model_options = [
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.5-pro-001",
        "gemini-1.0-pro",
        "gemini-pro"
    ]
    model_name = st.selectbox("Select Model", model_options)
    
    st.info("Get your key from Google AI Studio.")

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    user_request: str
    requirements: str
    initial_code: str
    api_needs: str
    final_code: str
    steps_log: List[str]
    error: str

# --- Helper Functions ---

def get_llm(api_key, model):
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.7)

def get_content_string(response_content: Union[str, List]) -> str:
    """Helper to handle if LLM returns a list of content parts instead of a string."""
    try:
        if isinstance(response_content, str):
            return response_content
        if isinstance(response_content, list):
            return "".join([str(item) for item in response_content])
        return str(response_content)
    except Exception as e:
        return f"Error parsing content: {str(e)}"

def extract_code(text: str) -> str:
    """Robustly extracts Python code from Markdown text using Regex."""
    try:
        # We split the string concatenation to avoid confusing the markdown parser
        # The pattern looks for triple backticks enclosing content
        code_block_pattern = r"``" + r"`(?:python)?\s*(.*?)``" + r"`"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        # Fallback: Check if it looks like code
        if "def " in text or "import " in text:
            return text.strip()
        
        return ""
    except Exception:
        return text

def sanitize_code(code: str) -> str:
    """
    Applies brute-force string replacements to fix common LLM generation errors
    that prompts sometimes fail to catch (like deprecated imports).
    """
    # Fix 1: Force Pydantic V2 over deprecated LangChain wrappers
    code = code.replace("from langchain_core.pydantic_v1", "from pydantic")
    code = code.replace("from langchain.pydantic_v1", "from pydantic")
    
    # Fix 2: Safety net for OpenAI -> Gemini if the LLM ignores instructions
    if "ChatOpenAI" in code:
        code = code.replace("from langchain_openai", "from langchain_google_genai")
        code = code.replace("ChatOpenAI", "ChatGoogleGenerativeAI")
        
    return code

# Mapping of common import names to their actual PyPI package names
PACKAGE_MAPPING = {
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
    "dateutil": "python-dateutil"
}

def install_missing_package(import_name: str):
    """Attempts to install a missing package via pip, handling aliases."""
    try:
        package_name = PACKAGE_MAPPING.get(import_name, import_name)
        if not re.match(r"^[a-zA-Z0-9_\-]+$", package_name):
            raise ValueError(f"Invalid package name: {package_name}")
            
        st.info(f"Attempting to install `{package_name}`...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            st.error(f"Pip Install Failed for `{package_name}`:\n\n{result.stderr}")
            return False
        return True
    except Exception as e:
        st.error(f"System Error during install of {package_name}: {e}")
        return False

# --- Node Logic ---

def node_requirements_analyst(state: AgentState):
    """Step 1: Analyzes user request."""
    try:
        llm = get_llm(st.session_state.get("api_key"), st.session_state.get("model"))
        prompt = f"""
        You are an expert AI Systems Analyst. 
        Analyze the following user request for a new AI Agent:
        "{state['user_request']}"
        
        Create a detailed technical requirement outline. 
        Specify:
        1. The Goal of the agent.
        2. The necessary Nodes (functions) required in the graph.
        3. The flow of data (Edges).
        4. What information needs to be stored in the State.
        5. ANY external API keys needed. **IMPORTANT: For the LLM, always specify 'Google API Key' (Gemini), NOT OpenAI.**
        6. UI Components needed (e.g., "A text input for destination").
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        content = get_content_string(response.content)
        return {"requirements": content, "steps_log": ["Requirements Analysis Complete"]}
    except Exception as e:
        return {"error": f"Analyst Error: {str(e)}", "steps_log": ["Requirements Analysis Failed"]}

def node_architect(state: AgentState):
    """Step 2: Writes the initial LangGraph code."""
    if state.get("error"): return {}
    
    try:
        llm = get_llm(st.session_state.get("api_key"), st.session_state.get("model"))
        prompt = f"""
        You are a Senior Python Developer specializing in Streamlit and LangGraph.
        Based on these requirements:
        {state['requirements']}
        
        Write a COMPLETE, STANDALONE Streamlit application (`app.py`).
        
        CRITICAL ARCHITECTURE RULES:
        1. **FORBIDDEN**: Do NOT use `langchain_openai` or `ChatOpenAI`. 
        2. **MANDATORY**: You MUST use `langchain_google_genai` and `ChatGoogleGenerativeAI`.
        3. **Sidebar**: In `st.sidebar`, ask for "Google Gemini API Key". Use this key to initialize the LLM.
        4. **Imports**: Start with `import streamlit as st`, `import os`, `from typing import...`, `from langgraph...`, `from langchain_google_genai import ChatGoogleGenerativeAI`.
        5. **Main UI**: Create `st.text_input`, `st.button`, etc., based on the requirements.
        6. **Graph Definition**: Define the State, Nodes, and Graph *inside* the script.
        7. **Execution**: When the user clicks the "Run" button, compile the graph and stream/invoke it. Display results using `st.write` or `st.markdown`.
        8. **Compatibility**: Use standard `pydantic` v2.
        
        The output must be a ready-to-run file.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        content = get_content_string(response.content)
        return {"initial_code": content, "steps_log": ["Initial App Drafted"]}
    except Exception as e:
        return {"error": f"Architect Error: {str(e)}", "steps_log": ["Architecture Failed"]}

def node_api_specialist(state: AgentState):
    """Step 3: Identifies missing APIs."""
    if state.get("error"): return {}
    
    try:
        llm = get_llm(st.session_state.get("api_key"), st.session_state.get("model"))
        prompt = f"""
        You are a Backend Integration Specialist.
        Review this initial code draft:
        {state['initial_code']}
        
        1. Ensure `st.sidebar.text_input` exists for "Google Gemini API Key".
        2. Ensure ANY other API keys (Serper, Weather, etc.) are also in the sidebar.
        3. Verify NO OpenAI keys are requested. If they are, replace them with Gemini.
        4. Suggest adding a comment at the top of the file listing `pip install` commands.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        content = get_content_string(response.content)
        return {"api_needs": content, "steps_log": ["API Dependencies Identified"]}
    except Exception as e:
        return {"error": f"API Specialist Error: {str(e)}", "steps_log": ["API Analysis Failed"]}

def node_code_reviewer(state: AgentState):
    """Step 4: Merges everything and outputs clean, runnable code."""
    if state.get("error"): return {}

    try:
        llm = get_llm(st.session_state.get("api_key"), st.session_state.get("model"))
        prompt = f"""
        You are a Lead Code Reviewer. 
        Your goal is to produce the FINAL, RUNNABLE `app.py` file.
        
        Inputs:
        - Draft Code: {state['initial_code']}
        - API Suggestions: {state['api_needs']}
        
        STRICT REQUIREMENTS:
        1. **Standalone File**: The code must run with `streamlit run app.py`.
        2. **LLM**: MUST be `ChatGoogleGenerativeAI`. REMOVE all references to OpenAI.
        3. **Sidebar**: Sidebar input MUST be `st.sidebar.text_input("Google Gemini API Key", type="password")`.
        4. **Dependencies**: Comment block at top with `pip install` commands (include `langchain-google-genai`).
        5. **Imports**: Ensure correct imports (`streamlit`, `langgraph`, `pydantic`, `langchain_google_genai`).
        6. **Structure**: Imports -> `st.set_page_config` -> Sidebar -> Classes/State -> Nodes -> Graph -> UI.
        7. **No `NameError`**: Define classes/functions before use.
        8. **Output**: ONLY Python code in markdown blocks.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        content = get_content_string(response.content)
        
        clean_code = extract_code(content)
        if not clean_code:
            clean_code = content.replace("```python", "").replace("```", "").strip()
            
        # --- Sanitization: Fix common AI errors ---
        clean_code = sanitize_code(clean_code)

        return {"final_code": clean_code, "steps_log": ["Final App Generated"]}
    except Exception as e:
        return {"error": f"Reviewer Error: {str(e)}", "steps_log": ["Code Generation Failed"]}

# --- Main App Logic ---

# PERSISTENCE FIX: Check if an agent exists and load it into session state
# This ensures that if the user refreshes the page, the agent code is not lost.
if "res_final" not in st.session_state:
    if os.path.exists("pages/generated_agent.py"):
        try:
            with open("pages/generated_agent.py", "r") as f:
                st.session_state["res_final"] = f.read()
                st.session_state["agent_deployed"] = True
        except Exception:
            pass

user_input = st.text_area("Describe the AI Agent you want to build:", height=100, 
                          placeholder="e.g., An agent that takes a topic, searches Google for recent news, summarizes it, and writes a LinkedIn post about it.")

start_btn = st.button("Build Agent", type="primary")

if start_btn:
    if not api_key:
        st.error("Please provide a Google Gemini API Key in the sidebar.")
    elif not user_input:
        st.warning("Please describe your agent first.")
    else:
        st.session_state["api_key"] = api_key
        st.session_state["model"] = model_name
        
        for key in ['res_req', 'res_draft', 'res_api', 'res_final', 'res_error', 'agent_deployed']:
            if key in st.session_state:
                del st.session_state[key]

        workflow = StateGraph(AgentState)
        workflow.add_node("analyst", node_requirements_analyst)
        workflow.add_node("architect", node_architect)
        workflow.add_node("api_specialist", node_api_specialist)
        workflow.add_node("reviewer", node_code_reviewer)
        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", "architect")
        workflow.add_edge("architect", "api_specialist")
        workflow.add_edge("api_specialist", "reviewer")
        workflow.add_edge("reviewer", END)
        app = workflow.compile()

        status_container = st.status("Building your agent...", expanded=True)
        
        initial_state = {
            "user_request": user_input, 
            "requirements": "", 
            "initial_code": "", 
            "api_needs": "", 
            "final_code": "",
            "steps_log": [],
            "error": ""
        }
        
        try:
            for output in app.stream(initial_state):
                for key, value in output.items():
                    if "steps_log" in value and value["steps_log"]:
                        status_container.write(f"✅ {value['steps_log'][0]}")
                    
                    if "error" in value and value["error"]:
                        st.session_state['res_error'] = value['error']
                        status_container.update(label="Error Occurred", state="error")
                        st.error(value['error'])
                        break

                    if "requirements" in value: st.session_state['res_req'] = value['requirements']
                    if "initial_code" in value: st.session_state['res_draft'] = value['initial_code']
                    if "api_needs" in value: st.session_state['res_api'] = value['api_needs']
                    if "final_code" in value: st.session_state['res_final'] = value['final_code']

            if not st.session_state.get('res_error'):
                status_container.update(label="Agent Generation Complete!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Critical System Error: {str(e)}")
            st.code(traceback.format_exc())

# --- Result Display & Dynamic Execution ---
if st.session_state.get('res_final'):
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Generated App Code", "Requirements 📋", "Draft Code 🏗️", "API Analysis 🔌"])
    
    with tab1:
        st.subheader("Your New Agent Application")
        st.success("Agent successfully generated! You can download it OR auto-deploy it to the sidebar.")
        
        code = st.session_state.get('res_final', '')
        st.code(code, language='python')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download my_new_agent.py",
                data=code,
                file_name="my_new_agent.py",
                mime="text/x-python",
                use_container_width=True
            )
            
        with col2:
            if st.button("🚀 Deploy Agent to Sidebar", type="primary", use_container_width=True):
                try:
                    os.makedirs("pages", exist_ok=True)
                    file_path = os.path.join("pages", "generated_agent.py")
                    with open(file_path, "w") as f:
                        f.write(code)
                    
                    st.session_state['agent_deployed'] = True
                    # Force refresh so file watcher catches the new page
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Deployment error: {e}")

        if st.session_state.get('agent_deployed', False):
            st.success("✅ Agent deployed!")
            st.markdown("### 👉 Look at the sidebar on the left!")
            st.markdown("You should see a page called **`generated_agent`**. Click it to run your new app.")
            st.info("Don't see it? It's safe to **refresh your browser** now. Your agent code is saved locally and will reappear here.")
            
            # Attempt link, but fail gracefully if Streamlit isn't ready yet
            try:
                st.page_link("pages/generated_agent.py", label="Open Agent Directly", icon="🤖", use_container_width=True)
            except Exception:
                pass # User can use the sidebar

    with tab2:
        st.markdown(st.session_state.get('res_req', 'Processing...'))

    with tab3:
        st.code(st.session_state.get('res_draft', 'Processing...'), language='python')
        
    with tab4:
        st.markdown(st.session_state.get('res_api', 'Processing...'))