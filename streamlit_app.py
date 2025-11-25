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
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    model_name = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    
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
    - Define the TypedDict State.
    - Define the Nodes.
    - Define the Graph compilation.
    - Do not worry about specific API keys yet, use placeholders.
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
    
    1. List what specific external APIs are needed (e.g., Google Search, Weather API, Database) based on the user request: "{state['user_request']}".
    2. Suggest how to integrate them into the existing nodes.
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
    
    Instructions:
    1. Merge the API logic into the draft code.
    2. Ensure all imports are correct (langgraph, langchain, etc.).
    3. Add comments explaining how to run it.
    4. OUTPUT ONLY THE PYTHON CODE. No markdown backticks, no conversational text before or after.
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

            # --- Display Results ---
            st.divider()
            
            # Use tabs for clean organization
            tab1, tab2, tab3, tab4 = st.tabs(["Final Code 🚀", "Requirements 📋", "Draft Code 🏗️", "API Analysis 🔌"])
            
            with tab1:
                st.subheader("Your Custom Agent Code")
                st.code(st.session_state.get('res_final', ''), language='python')
                st.download_button(
                    label="Download Python File",
                    data=st.session_state.get('res_final', ''),
                    file_name="my_new_agent.py",
                    mime="text/x-python"
                )

            with tab2:
                st.markdown(st.session_state.get('res_req', 'Processing...'))

            with tab3:
                st.code(st.session_state.get('res_draft', 'Processing...'), language='python')
                
            with tab4:
                st.markdown(st.session_state.get('res_api', 'Processing...'))

        except Exception as e:
            st.error(f"An error occurred: {e}")