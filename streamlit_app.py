import streamlit as st
import os
import re
import traceback
from typing import TypedDict, List, Union, Dict, Any
import inspect
import subprocess
import sys
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
5. **Run the Agent** ⚡
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
        # Resolve the actual package name from the import name
        package_name = PACKAGE_MAPPING.get(import_name, import_name)
        
        # Security check: basic sanitization to prevent command injection
        if not re.match(r"^[a-zA-Z0-9_\-]+$", package_name):
            raise ValueError(f"Invalid package name: {package_name}")
            
        st.info(f"Attempting to install `{package_name}`...")
        
        # Use subprocess.run to capture output for better debugging
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
        5. ANY external API keys needed (e.g., SendGrid, Spotify, OpenAI).
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
        You are a Senior Python Developer specializing in LangGraph and LangChain.
        Based on these requirements:
        {state['requirements']}
        
        Write the Python code to implement this using `langgraph` and `langchain_google_genai`.
        
        CRITICAL ARCHITECTURE RULES:
        1. The code MUST be wrapped in a function: 
        `def run_agent(user_input: str, llm_api_key: str, secrets: dict = None) -> str:`
        
        2. Any external API keys must be retrieved from the `secrets` dictionary.
        Example: `weather_key = secrets.get("WEATHER_API_KEY")`
        
        3. Define a helper function to list required keys:
        `def get_required_api_keys() -> list[str]:`
        Example: return ["WEATHER_API_KEY"]
        
        4. COMPATIBILITY: Use standard `pydantic` v2. Do NOT use `langchain_core.pydantic_v1`.
        
        5. IMPORTS: Include `from typing import Dict, List, Any` and `from pydantic import BaseModel, Field`.
        
        - Use `ChatGoogleGenerativeAI` with `google_api_key=llm_api_key`.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        content = get_content_string(response.content)
        return {"initial_code": content, "steps_log": ["Initial Code Drafted"]}
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
        
        1. List what specific external APIs are needed based on the user request: "{state['user_request']}".
        2. Explicitly list the VARIABLE NAMES for these keys (e.g., 'SENDGRID_API_KEY').
        3. Suggest how to integrate them into the `secrets` dictionary pattern.
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
        Your goal is to produce the FINAL, RUNNABLE Python file.
        
        Inputs:
        - Draft Code: {state['initial_code']}
        - API Suggestions: {state['api_needs']}
        
        STRICT REQUIREMENTS:
        1. The code MUST be self-contained (imports at top).
        
        2. DEFINE THIS EXACT FUNCTION SIGNATURE for the main entry point:
        `def run_agent(user_input: str, llm_api_key: str, secrets: dict = None) -> str:`
        
        3. DEFINE THIS HELPER FUNCTION:
        `def get_required_api_keys() -> list[str]:`
        - It must return a list of strings of the keys needed (e.g., ["SENDGRID_API_KEY"]).
        - If no extra keys are needed, return [].
        
        4. Inside `run_agent`:
        - Initialize `ChatGoogleGenerativeAI(google_api_key=llm_api_key, ...)`
        - Access third-party keys via `secrets.get("KEY_NAME")`.
        - Define the State, Nodes, and Workflow.
        - Compile and run the workflow.
        - Return the final text output.
        
        5. Assume `langchain_community` is installed.
        6. COMPATIBILITY: Use standard `pydantic` v2 (e.g. `from pydantic import BaseModel`). Do NOT use `langchain_core.pydantic_v1`.
        7. IMPORTS: Include `from typing import Dict, List, Any` and `from pydantic import BaseModel, Field` at the top.
        8. OUTPUT ONLY THE PYTHON CODE. Wrap it in markdown code blocks.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        content = get_content_string(response.content)
        
        clean_code = extract_code(content)
        if not clean_code:
            clean_code = content.replace("```python", "").replace("```", "").strip()

        return {"final_code": clean_code, "steps_log": ["Final Code Generated"]}
    except Exception as e:
        return {"error": f"Reviewer Error: {str(e)}", "steps_log": ["Code Generation Failed"]}

# --- Main App Logic ---

user_input = st.text_area("Describe the AI Agent you want to build:", height=100, 
                          placeholder="e.g., An agent that takes a topic, searches Google for recent news, summarizes it, and writes a LinkedIn post about it.")

start_btn = st.button("Build Agent", type="primary")

if start_btn:
    if not api_key:
        st.error("Please provide a Google Gemini API Key in the sidebar.")
    elif not user_input:
        st.warning("Please describe your agent first.")
    else:
        # Store config
        st.session_state["api_key"] = api_key
        st.session_state["model"] = model_name
        
        # Reset previous results
        for key in ['res_req', 'res_draft', 'res_api', 'res_final', 'res_error']:
            if key in st.session_state:
                del st.session_state[key]

        # --- Build the Graph ---
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("analyst", node_requirements_analyst)
        workflow.add_node("architect", node_architect)
        workflow.add_node("api_specialist", node_api_specialist)
        workflow.add_node("reviewer", node_code_reviewer)

        # Add Edges
        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", "architect")
        workflow.add_edge("architect", "api_specialist")
        workflow.add_edge("api_specialist", "reviewer")
        workflow.add_edge("reviewer", END)

        # Compile
        app = workflow.compile()

        # Execute
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
            # Stream the updates
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚡ Test Agent", "Final Code 🚀", "Requirements 📋", "Draft Code 🏗️", "API Analysis 🔌"])
    
    with tab1:
        st.subheader("Interactive Agent Playground")
        st.info("The agent code has been loaded into memory. Try it out below!")
        
        local_scope = {}
        generated_code = st.session_state['res_final']
        required_keys = []
        parsing_error = False
        
        try:
            # Pre-load common modules
            import langchain_community
            import langchain_core
            import langchain_google_genai
            import langgraph
            
            exec(generated_code, globals(), local_scope)
            
            if 'get_required_api_keys' in local_scope:
                try:
                    required_keys = local_scope['get_required_api_keys']()
                except Exception as e:
                    st.warning(f"Could not automatically detect API keys: {e}")
            else:
                st.warning("Agent code structure incomplete: missing `get_required_api_keys`.")

        except ModuleNotFoundError as e:
            parsing_error = True
            missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
            
            st.warning(f"🚨 Missing Module Detected: `{missing_module}`")
            
            # Simple heuristic: don't try to install submodules like 'langchain_core.pydantic_v1'
            if "." in missing_module:
                st.error(f"Cannot auto-install submodule `{missing_module}`. This is likely a version incompatibility (e.g., using deprecated code). Please regenerate the agent.")
            else:
                if st.button(f"📥 Install {missing_module} and Reload"):
                    with st.spinner(f"Installing {missing_module}..."):
                        if install_missing_package(missing_module):
                            st.success("Installed! Reloading app...")
                            st.rerun()

        except SyntaxError as e:
            parsing_error = True
            st.error(f"Syntax Error in generated code: {e}")
            st.markdown("Check the 'Final Code' tab. The AI might have included text outside the code block.")
        except Exception as e:
            parsing_error = True
            st.error(f"Error loading agent: {e}")

        if not parsing_error:
            user_secrets = {}
            if required_keys:
                st.warning(f"This agent requires external API keys: {', '.join(required_keys)}")
                for key_name in required_keys:
                    val = st.text_input(f"Enter {key_name}", type="password", key=f"secret_{key_name}")
                    user_secrets[key_name] = val

            test_query = st.text_input("Talk to your new agent:", placeholder="Enter a query...")
            
            if st.button("Run Agent"):
                if not test_query:
                    st.warning("Please enter a query.")
                else:
                    missing_keys = [k for k in required_keys if not user_secrets.get(k)]
                    if missing_keys:
                        st.error(f"Missing keys: {', '.join(missing_keys)}")
                    else:
                        with st.spinner("Running your custom agent..."):
                            try:
                                if 'run_agent' in local_scope:
                                    sig = inspect.signature(local_scope['run_agent'])
                                    if 'secrets' in sig.parameters:
                                        result = local_scope['run_agent'](test_query, api_key, user_secrets)
                                    else:
                                        result = local_scope['run_agent'](test_query, api_key)
                                        
                                    st.success("Result:")
                                    st.write(result)
                                else:
                                    st.error("Function `run_agent` not found in generated code.")
                            except ModuleNotFoundError as e:
                                st.error(f"Runtime Missing Module: {str(e)}")
                            except Exception as e:
                                st.error(f"Runtime Error: {str(e)}")
                                st.code(traceback.format_exc())

    with tab2:
        st.subheader("Your Custom Agent Code")
        st.code(st.session_state.get('res_final', ''), language='python')
        st.download_button(
            label="Download Python File",
            data=st.session_state.get('res_final', ''),
            file_name="my_new_agent.py",
            mime="text/x-python"
        )

    with tab3:
        st.markdown(st.session_state.get('res_req', 'Processing...'))

    with tab4:
        st.code(st.session_state.get('res_draft', 'Processing...'), language='python')
        
    with tab5:
        st.markdown(st.session_state.get('res_api', 'Processing...'))