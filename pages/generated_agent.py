# --- Installation Requirements ---
# pip install streamlit langgraph langchain-google-genai pydantic
# Note: Pydantic is required for structured output models (BaseModel, Field).

import streamlit as st
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# CRITICAL: Using pydantic_v1 for structured output schema definition, 
# which is the standard mechanism in langchain_core for reliable schema passing to LLMs.
from langchain_core.pydantic_v1 import BaseModel, Field

# --- 1. State Definition ---

class TouristSpots(BaseModel):
    """Pydantic model for structured extraction of tourist spots."""
    spots: List[str] = Field(description="A list of exactly 3 highly-rated or popular tourist attractions relevant to the destination.")

class TravelAgentState(TypedDict):
    """Represents the state of our sequential travel planning process."""
    destination: str
    spots_list: List[str]
    itinerary_text: str
    budget_string: str
    current_status: str # Used for tracking progress

# --- 2. LLM Initialization and Caching ---

@st.cache_resource
def get_llm(api_key: str):
    """Initializes and caches the ChatGoogleGenerativeAI model."""
    if not api_key:
        return None
    try:
        # Using gemini-2.5-flash for a balance of speed and capability
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.4
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

# --- 3. Node Definitions (LLM Agents) ---

def scout_locations(state: TravelAgentState) -> TravelAgentState:
    """Node 1: Identifies 3 highly-rated tourist spots."""
    st.info(f"📍 Node running: Scouting locations for {state['destination']}...")
    llm = st.session_state.llm
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert travel researcher. Identify exactly 3 highly-rated or popular tourist attractions/spots relevant to the user's destination. Respond ONLY with the JSON object."),
        ("human", f"Destination: {state['destination']}")
    ])
    
    chain = prompt | llm.with_structured_output(TouristSpots)
    
    try:
        response = chain.invoke({})
        return {
            "spots_list": response.spots,
            "current_status": "Locations scouted successfully."
        }
    except Exception as e:
        st.error(f"Error in scout_locations: {e}")
        return {"spots_list": ["Error scouting locations"], "current_status": "Failed scouting."}

def plan_itinerary(state: TravelAgentState) -> TravelAgentState:
    """Node 2: Creates a structured, detailed 1-day schedule."""
    st.info("🗓️ Node running: Planning 1-day itinerary...")
    llm = st.session_state.llm
    
    spots = ", ".join(state['spots_list'])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative itinerary planner. Create a structured, detailed 1-day schedule (Morning, Afternoon, Evening) for the user's destination, making sure to incorporate the following 3 specific spots."),
        ("human", f"Destination: {state['destination']}. Spots to include: {spots}. Format the output as a clean, easy-to-read text block with clear headings.")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({})
        return {
            "itinerary_text": response.content,
            "current_status": "Itinerary planned successfully."
        }
    except Exception as e:
        st.error(f"Error in plan_itinerary: {e}")
        return {"itinerary_text": "Error generating itinerary.", "current_status": "Failed planning."}

def budget_estimator(state: TravelAgentState) -> TravelAgentState:
    """Node 3: Calculates a rough estimated cost for the itinerary."""
    st.info("💰 Node running: Estimating budget...")
    llm = st.session_state.llm
    
    itinerary = state['itinerary_text']
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst specializing in travel costs. Based on the provided destination and itinerary, calculate and format a rough, estimated cost breakdown (e.g., transportation, entrance fees, food) for one person for this 1-day trip. Provide a final summary cost in USD."),
        ("human", f"Destination: {state['destination']}. Itinerary: {itinerary}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({})
        return {
            "budget_string": response.content,
            "current_status": "Budget estimated successfully."
        }
    except Exception as e:
        st.error(f"Error in budget_estimator: {e}")
        return {"budget_string": "Error estimating budget.", "current_status": "Failed budgeting."}

# --- 4. Graph Construction ---

def create_travel_agent_graph():
    """Defines and compiles the sequential LangGraph workflow."""
    workflow = StateGraph(TravelAgentState)
    
    # Add Nodes
    workflow.add_node("scout_locations", scout_locations)
    workflow.add_node("plan_itinerary", plan_itinerary)
    workflow.add_node("budget_estimator", budget_estimator)
    
    # Define the sequential Edges (Start -> Scout -> Plan -> Budget -> End)
    workflow.set_entry_point("scout_locations")
    
    workflow.add_edge("scout_locations", "plan_itinerary")
    workflow.add_edge("plan_itinerary", "budget_estimator")
    workflow.add_edge("budget_estimator", END)
    
    return workflow.compile()

# --- 5. Streamlit Application ---

def run_app():
    st.set_page_config(page_title="Smart Travel Agent (STA)", layout="wide")
    st.title("✈️ Smart Travel Agent (STA)")
    st.caption("A sequential planning agent powered by LangGraph and Google Gemini.")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Google Gemini API Key", 
            type="password",
            value=os.getenv("GEMINI_API_KEY") # Pre-populate if environment variable exists
        )
        
        if api_key:
            st.session_state.llm = get_llm(api_key)
            st.success("LLM Initialized!")
        else:
            st.session_state.llm = None
            st.warning("Please enter your Gemini API Key to run the agent.")

    # --- Main UI and Execution Setup ---
    
    if 'graph' not in st.session_state:
        st.session_state.graph = create_travel_agent_graph()
    
    if 'results' not in st.session_state:
        st.session_state.results = None

    destination_input = st.text_input(
        "Where are you planning to go?",
        placeholder="e.g., Paris, France",
        key="destination_input"
    )

    run_button = st.button(
        "Generate 1-Day Travel Plan", 
        disabled=not (st.session_state.llm and destination_input)
    )

    # --- Execution Logic ---
    if run_button and st.session_state.llm and destination_input:
        
        # Initial State (Simulating the 'Start' node input)
        initial_state = TravelAgentState(
            destination=destination_input,
            spots_list=[],
            itinerary_text="",
            budget_string="",
            current_status="Starting travel plan generation..."
        )
        
        st.session_state.results = None
        
        # Progress Indicator
        status_placeholder = st.empty()
        status_placeholder.info(f"Agent starting for: {destination_input}")
        
        try:
            with st.spinner("Processing request through sequential nodes..."):
                # Invoke the compiled graph
                final_state = st.session_state.graph.invoke(initial_state)
                st.session_state.results = final_state
                
                status_placeholder.empty()
                st.success("✅ Travel Plan Generation Complete!")

        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred during graph execution. Please check the API key and try again: {e}")
            st.session_state.results = None

    # --- Display Final Results ---
    if st.session_state.results:
        results = st.session_state.results
        
        st.header(f"Travel Plan for {results['destination']}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🗺️ Top Tourist Spots")
            if results['spots_list']:
                st.markdown(
                    "\n".join([f"- **{spot}**" for spot in results['spots_list']])
                )
            else:
                st.warning("Could not identify specific spots.")
                
            st.subheader("💲 Estimated Budget (1 Day)")
            st.markdown(results['budget_string'])

        with col2:
            st.subheader("📅 Detailed 1-Day Itinerary")
            st.markdown(results['itinerary_text'])

if __name__ == "__main__":
    run_app()