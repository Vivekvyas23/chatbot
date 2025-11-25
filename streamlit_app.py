import streamlit as st
import os
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
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
    # Updated model names to include 2.5 flash and other variants
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
    # We keep a log of steps for the UI
    steps_log: List[str]

# --- Node Logic ---

def get_llm(api_key, model):
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.7)

def node_requirements_analyst(state: AgentState):
    """Step 1: Analyzes user request and creates a requirement outline."""
    llm = get_llm(st.session_state.get("api_key"), st.session_state.get("model"))
    
    prompt = f"""
    You are an expert AI Systems Analyst. 
    Analyze the following user request for a new AI Agent:
    "{state['user_request']}"
    
    Create a detailed technical requirement outline. 
    Specificy:
    1. The Goal of the agent.
    2. The necessary Nodes (functions) required in the graph.
    3. The flow of data (Edges).
    4. What information needs to be stored in the State.
    5. ANY external API keys needed (e.g., SendGrid, Spotify, OpenAI).
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "requirements": response.content,
        "steps_log": ["Requirements Analysis Complete"]
    }

def node_architect(state: AgentState):
    """Step 2: Writes the initial LangGraph code based on requirements."""
    llm = get_llm(st.session_state.get("api_key"), st.session_state.get("model"))
    
    prompt = f"""
    You are a Senior Python Developer specializing in LangGraph and LangChain.
    Based on these requirements:
    {state['requirements']}
    
    Write the Python code to implement this using `langgraph` and `langchain_google_genai`.
    
    CRITICAL ARCHITECTURE RULE:
    1. The code MUST be wrapped in a function: 
       `def run_agent(user_input: str, llm_api_key: str, secrets: dict = None) -> str:`
    
    2. Any external API keys must be retrieved from the `secrets` dictionary.
       Example: `weather_key = secrets.get("WEATHER_API_KEY")`
    
    3. Define a helper function to list required keys:
       `def get_required_api_keys() -> list[str]:`
       Example: return ["WEATHER_API_KEY"]
    
    - Use `ChatGoogleGenerativeAI` with `google_api_key=llm_api_key`.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "initial_code": response.content,
        "steps_log": ["Initial Code Drafted"]
    }

def node_api_specialist(state: AgentState):
    """Step 3: Identifies missing APIs and adds integration logic."""
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
    return {
        "api_needs": response.content,
        "steps_log": ["API Dependencies Identified"]
    }

def node_code_reviewer(state: AgentState):
    """Step 4: Merges everything and outputs clean, runnable code."""
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
    6. OUTPUT ONLY THE PYTHON CODE. No markdown backticks.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Simple cleanup to remove markdown code blocks if the LLM adds them despite instructions
    clean_code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {
        "final_code": clean_code,
        "steps_log": ["Final Code Generated"]
    }

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
        # Store config in session state for nodes to access
        st.session_state["api_key"] = api_key
        st.session_state["model"] = model_name

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
            "steps_log": []
        }
        
        try:
            # Stream the updates
            for output in app.stream(initial_state):
                for key, value in output.items():
                    if "steps_log" in value:
                        status_container.write(f"✅ {value['steps_log'][0]}")
                    
                    # Store intermediate results to display later
                    if "requirements" in value:
                        st.session_state['res_req'] = value['requirements']
                    if "initial_code" in value:
                        st.session_state['res_draft'] = value['initial_code']
                    if "api_needs" in value:
                        st.session_state['res_api'] = value['api_needs']
                    if "final_code" in value:
                        st.session_state['res_final'] = value['final_code']

            status_container.update(label="Agent Generation Complete!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Result Display & Dynamic Execution ---
if st.session_state.get('res_final'):
    st.divider()
    
    # Use tabs for clean organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚡ Test Agent", "Final Code 🚀", "Requirements 📋", "Draft Code 🏗️", "API Analysis 🔌"])
    
    with tab1:
        st.subheader("Interactive Agent Playground")
        st.info("The agent code has been loaded into memory. Try it out below!")
        
        # 1. Prepare execution namespace & Parse Keys
        local_scope = {}
        generated_code = st.session_state['res_final']
        required_keys = []
        
        try:
            exec(generated_code, globals(), local_scope)
            if 'get_required_api_keys' in local_scope:
                required_keys = local_scope['get_required_api_keys']()
        except Exception as e:
            st.error(f"Error parsing agent code: {e}")

        # 2. Collect Secrets
        user_secrets = {}
        if required_keys:
            st.warning(f"This agent requires external API keys: {', '.join(required_keys)}")
            for key_name in required_keys:
                user_secrets[key_name] = st.text_input(f"Enter {key_name}", type="password")

        # 3. Run Agent
        test_query = st.text_input("Talk to your new agent:", placeholder="Enter a query for the agent you just built...")
        
        if st.button("Run Agent"):
            if not test_query:
                st.warning("Please enter a query.")
            else:
                # Check if all keys are provided
                missing_keys = [k for k in required_keys if not user_secrets.get(k)]
                if missing_keys:
                    st.error(f"Missing keys: {', '.join(missing_keys)}")
                else:
                    with st.spinner("Running your custom agent..."):
                        try:
                            # 4. Check for the specific entry point function we requested
                            if 'run_agent' in local_scope:
                                # 5. Run the function with secrets
                                result = local_scope['run_agent'](test_query, api_key, user_secrets)
                                st.success("Result:")
                                st.write(result)
                            else:
                                st.error("The generated code did not define the required `run_agent` function. Please regenerate.")
                        except ModuleNotFoundError as e:
                            st.error(f"Missing Dependency Error: {str(e)}")
                            st.info("It seems the generated agent needs a library that isn't installed. Please check 'requirements.txt' and ensure `langchain_community` or other requested libraries are added.")
                        except Exception as e:
                            st.error(f"Execution Error: {str(e)}")
                            st.markdown("Try checking the 'Final Code' tab to see if there are syntax errors.")

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