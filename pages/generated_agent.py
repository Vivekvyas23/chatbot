import streamlit as st
import os
from typing import TypedDict, List, Dict, Any
from functools import partial

# --- Dependencies ---
# pip install streamlit langgraph langchain_core langchain-google-genai
# --------------------

# LangGraph and LangChain components
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. State Management ---

class AgentState(TypedDict):
    """
    Represents the state of our graph, persisting variables across nodes.
    """
    destination: str
    list_of_spots: List[str]
    itinerary_text: str
    budget_string: str

# --- 2. LLM Initialization ---

@st.cache_resource
def initialize_llm(api_key: str):
    """Initializes the Gemini LLM."""
    if not api_key:
        return None
    try:
        # Using gemini-2.5-flash for a balance of speed and planning capability
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key, 
            temperature=0.3
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM. Check your API key or permissions: {e}")
        return None

# --- 3. Graph Nodes (LLM Functions) ---

def scout_locations(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """Identifies three highly-rated tourist spots."""
    st.info(f"🔎 Node 1: Scouting locations in {state['destination']}...")
    destination = state['destination']
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert travel researcher. Identify exactly three diverse and highly-rated tourist spots suitable for a single day trip in the specified destination. Respond ONLY with a comma-separated list of the three location names, nothing else. Example: Tower of London, British Museum, Hyde Park."),
        ("human", f"Destination: {destination}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    
    # Process the response string into a list
    spots_str = response.content.strip()
    list_of_spots = [spot.strip() for spot in spots_str.split(',') if spot.strip()][:3]
    
    if len(list_of_spots) < 3:
        st.warning(f"Could only find {len(list_of_spots)} spots. Proceeding anyway.")
        
    return {"list_of_spots": list_of_spots}


def plan_itinerary(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """Creates a cohesive 1-day schedule."""
    st.info("🗓️ Node 2: Planning the itinerary (9 AM to 6 PM)...")
    destination = state['destination']
    spots = state['list_of_spots']
    
    if not spots:
        # This should ideally be caught by the streaming loop, but good for node safety
        st.error("Cannot plan itinerary: No spots were identified in the previous step.")
        return {"itinerary_text": "Planning failed: No spots identified."}

    spots_list = "\n- " + "\n- ".join(spots)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert itinerary planner. Create a detailed, cohesive 1-day schedule (9 AM to 6 PM) for the provided destination and spots. Logically sequence the spots, include suggested transition times, and brief activity descriptions. Format the output as clean, structured markdown text with clear time blocks."),
        ("human", f"Destination: {destination}\nSpots to include:{spots_list}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    
    return {"itinerary_text": response.content}


def budget_estimator(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """Provides a rough, high-level cost estimate."""
    st.info("💰 Node 3: Estimating preliminary budget...")
    destination = state['destination']
    itinerary = state['itinerary_text']
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst specializing in travel costs. Based on the destination and itinerary provided, give a rough, high-level cost estimate for a single traveler (entry fees, transportation, 2 meals). Provide the result as a numerical range in USD (e.g., $150 - $250 USD). Respond ONLY with the estimate string, including the currency."),
        ("human", f"Destination: {destination}\nItinerary:\n{itinerary}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    
    return {"budget_string": response.content.strip()}


def final_output(state: AgentState) -> Dict[str, Any]:
    """Compiles all data into the final state."""
    st.success("✅ Node 4: Planning complete! Compiling final report.")
    return state

# --- 4. Graph Definition ---

def create_travel_graph(llm: ChatGoogleGenerativeAI):
    """
    Defines and compiles the LangGraph StateGraph with a deterministic, linear flow.
    """
    workflow = StateGraph(AgentState)

    # Define Nodes (using partial to inject the LLM dependency)
    workflow.add_node("scout_locations", partial(scout_locations, llm=llm))
    workflow.add_node("plan_itinerary", partial(plan_itinerary, llm=llm))
    workflow.add_node("budget_estimator", partial(budget_estimator, llm=llm))
    workflow.add_node("final_output", final_output) 

    # Set Entry Point
    workflow.set_entry_point("scout_locations") # Start directly at the first processing node

    # Define Edges (linear flow)
    workflow.add_edge("scout_locations", "plan_itinerary")
    workflow.add_edge("plan_itinerary", "budget_estimator")
    workflow.add_edge("budget_estimator", "final_output")

    # Define End Point
    workflow.add_edge("final_output", END)

    return workflow.compile()

# --- 5. Streamlit Application ---

def main():
    st.set_page_config(page_title="Smart Travel Agent (Gemini/LangGraph)", layout="wide")
    st.title("🗺️ Smart Travel Agent: 1-Day Planner")
    st.caption("A sequential planning agent powered by LangGraph and Google Gemini.")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    google_api_key = st.sidebar.text_input(
        "Google Gemini API Key", 
        type="password", 
        help="Required for accessing the Gemini model."
    )
    
    llm = None
    if google_api_key:
        llm = initialize_llm(google_api_key)
    else:
        st.sidebar.warning("Please enter your Gemini API Key to proceed.")

    # --- Main UI Inputs ---
    st.header("1. Input")
    destination = st.text_input(
        "Destination (City, Country)",
        placeholder="e.g., Rome, Italy",
        key="destination_input"
    )

    is_ready = llm and destination
    run_button = st.button(
        "✈️ Plan My Trip", 
        type="primary", 
        disabled=(not is_ready)
    )

    # --- Execution Logic ---
    if run_button:
        st.divider()
        
        # Initialize the graph
        try:
            travel_planner = create_travel_graph(llm)
        except Exception as e:
            st.error(f"Failed to compile the graph: {e}")
            return

        # Initial State
        initial_state = {
            "destination": destination,
            "list_of_spots": [],
            "itinerary_text": "",
            "budget_string": ""
        }
        
        final_state = initial_state # State dictionary to be updated by the stream

        # Run the graph using stream for live updates
        st.subheader("2. Execution Status")
        
        try:
            # Iterate through the stream to get state changes in real-time
            for chunk in travel_planner.stream(initial_state):
                # The chunk contains updates from the node that just executed
                for key, value in chunk.items():
                    if key != "__end__":
                        # Merge the output from the current node into the final state
                        final_state.update(value)
            
            # --- Output Display ---
            st.divider()
            st.header(f"3. Final Trip Report for: {final_state['destination']}")

            col1, col2 = st.columns([1, 2])
            
            # Column 1: Spots and Budget
            with col1:
                st.markdown("### 💰 Estimated Budget")
                budget_display = final_state.get('budget_string')
                if budget_display:
                    st.info(f"**{budget_display}**")
                else:
                    st.warning("Budget estimation failed.")

                st.markdown("### 📍 Key Locations Identified")
                spots = final_state.get('list_of_spots', [])
                spots_md = "\n".join([f"* {spot}" for spot in spots])
                st.markdown(spots_md if spots_md else "*No locations found.*")

            # Column 2: Itinerary
            with col2:
                st.markdown("### 🗓️ Detailed 1-Day Itinerary (9 AM - 6 PM)")
                st.markdown(final_state.get('itinerary_text', "*Itinerary planning failed.*"))

        except Exception as e:
            st.error("An error occurred during graph execution. Check the status messages above for details.")
            st.exception(e)

if __name__ == "__main__":
    main()