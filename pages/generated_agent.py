# --- INSTALLATION REQUIREMENTS ---
# pip install streamlit langgraph langchain-openai
# ---------------------------------

import streamlit as st
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

# --- 1. State Management Definition (Required by LangGraph) ---

class TravelAgentState(TypedDict):
    """
    Represents the state of the Smart Travel Agent during execution.
    """
    destination: str
    tourist_spots: List[str]
    itinerary_text: str
    estimated_budget: str
    error_message: str

# --- Streamlit UI Setup (Sidebar for API Keys) ---

st.set_page_config(page_title="Smart Travel Agent (LangGraph)", layout="wide")
st.title("🌍 Smart Travel Agent")
st.caption("Powered by LangGraph and Streamlit. Transforms a destination into a detailed 1-day travel plan.")

# Sidebar for API Key Collection
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for all LLM processing.")
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("OpenAI Key Loaded.")
    else:
        st.warning("Please enter your OpenAI API Key to run the agent.")

# --- 2. Node Functions (Processing Logic) ---

def initialize_llm():
    """Initializes the LLM based on the API key."""
    # Ensure the key is set in the environment before initialization
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        # Fallback check if the user cleared the input after initialization
        if not openai_api_key:
            st.error("OpenAI API key is missing.")
            return None
    try:
        # Using a powerful model for complex reasoning and structured output
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def scout_locations(state: TravelAgentState) -> dict:
    """
    Node 1: Identifies 3 distinct tourist spots for the given destination,
    using robust JSON output parsing.
    """
    destination = state["destination"]
    llm = initialize_llm()
    if not llm: return {"error_message": "LLM initialization failed."}

    st.info(f"🔍 Scouting top 3 locations in {destination}...")

    # Configure LLM for strict JSON output using the OpenAI API parameter
    llm_json = llm.with_config({"response_format": {"type": "json_object"}})
    
    system_prompt = (
        "You are an expert travel scout. Your task is to identify the top 3 highly-rated and distinct "
        "tourist spots (e.g., historical, natural, cultural) for the given destination. "
        "You MUST output the result as a JSON list of strings, with no extra text. "
        "Example output: [\"Spot A\", \"Spot B\", \"Spot C\"]"
    )
    
    prompt = f"Destination: {destination}. Provide the 3 best spots."

    response = None # Initialize response for error handling scope
    try:
        response = llm_json.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        
        # Since response_format is set, the content should be clean JSON
        spots = json.loads(response.content)
        
        # Basic validation
        if not isinstance(spots, list) or len(spots) == 0:
            raise ValueError("LLM failed to return a valid JSON list of spots.")
            
        return {"tourist_spots": spots}

    except Exception as e:
        raw_content = response.content if response else "N/A"
        return {"error_message": f"Error in scout_locations (JSON parsing): {e}. Raw content: {raw_content[:100]}..."}


def plan_itinerary(state: TravelAgentState) -> dict:
    """
    Node 2: Generates a logical, time-based, 1-day itinerary.
    """
    destination = state["destination"]
    tourist_spots = state["tourist_spots"]
    llm = initialize_llm()
    if not llm: return {"error_message": "LLM initialization failed."}

    # Handle case where scouting failed but flow continued
    if not tourist_spots:
        return {"error_message": "Itinerary planning skipped because no tourist spots were identified."}

    st.info("🗓️ Planning the 1-day itinerary...")

    spot_list = "\n- " + "\n- ".join(tourist_spots)

    system_prompt = (
        "You are a meticulous travel planner. Generate a detailed, time-based, 1-day itinerary "
        "from 8:00 AM to 6:00 PM utilizing ALL the provided tourist spots in a logical order. "
        "Include suggested transition times (e.g., transit/walking) between locations. "
        "Format the output clearly using Markdown headings and bullet points."
    )
    
    prompt = (
        f"Destination: {destination}\n"
        f"Spots to include:\n{spot_list}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        return {"itinerary_text": response.content}
    except Exception as e:
        return {"error_message": f"Error in plan_itinerary: {e}"}


def budget_estimator(state: TravelAgentState) -> dict:
    """
    Node 3: Calculates a rough cost estimate for the day.
    """
    destination = state["destination"]
    itinerary_text = state["itinerary_text"]
    llm = initialize_llm()
    if not llm: return {"error_message": "LLM initialization failed."}

    # Handle case where itinerary failed but flow continued
    if not itinerary_text or itinerary_text == "Running...":
        return {"error_message": "Budget estimation skipped because the itinerary text is missing or incomplete."}

    st.info("💰 Estimating the budget...")

    system_prompt = (
        "You are a financial analyst specializing in travel costs. Analyze the destination and the provided itinerary. "
        "Calculate a rough cost estimate for the entire day. This must include estimated entry fees for the locations, "
        "rough local transit costs, and a small allowance ($15-$25 equivalent) for miscellaneous expenses (e.g., coffee/snack). "
        "Provide the final output as a single, clear, formatted string (e.g., '€120 - €160 EUR'). "
        "Do not include any other explanatory text."
    )
    
    prompt = (
        f"Destination: {destination}\n"
        f"Itinerary:\n---\n{itinerary_text}\n---\n"
        "Provide the rough total cost estimate for this plan."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        return {"estimated_budget": response.content.strip()}
    except Exception as e:
        return {"error_message": f"Error in budget_estimator: {e}"}


# --- 3. Main UI and Execution Logic ---

# Initialize output placeholders
if 'plan_output' not in st.session_state:
    st.session_state.plan_output = {
        "tourist_spots": [],
        "itinerary_text": "Plan not generated yet.",
        "estimated_budget": "N/A",
        "error_message": ""
    }

# Input Field
destination_input = st.text_input(
    "Where are you traveling for a single day?", 
    placeholder="e.g., Rome, Italy or Kyoto, Japan"
)

# Execution Button
if st.button("Generate Travel Plan", type="primary"):
    
    # 1. Pre-check requirements
    if not openai_api_key:
        st.error("Please provide the OpenAI API Key in the sidebar.")
    elif not destination_input:
        st.error("Please enter a destination.")
    else:
        # Reset state for new run
        st.session_state.plan_output = {
            "tourist_spots": [],
            "itinerary_text": "Running...",
            "estimated_budget": "Running...",
            "error_message": ""
        }
        
        # Use a spinner to show activity
        with st.spinner("Executing Smart Travel Agent Graph..."):
            
            # 2. Define the Graph Structure
            workflow = StateGraph(TravelAgentState)

            # Add nodes
            workflow.add_node("scout", scout_locations)
            workflow.add_node("plan", plan_itinerary)
            workflow.add_node("budget", budget_estimator)

            # Define edges (Linear flow)
            workflow.set_entry_point("scout")
            workflow.add_edge("scout", "plan")
            workflow.add_edge("plan", "budget")
            workflow.add_edge("budget", END)

            # Compile the graph
            app = workflow.compile()

            # 3. Initial State
            initial_state = {
                "destination": destination_input,
                "tourist_spots": [],
                "itinerary_text": "",
                "estimated_budget": "",
                "error_message": ""
            }

            # 4. Invoke the Graph
            try:
                # The graph executes sequentially
                final_state = app.invoke(initial_state)
                
                # Update session state with the final result
                st.session_state.plan_output = final_state
                
            except Exception as e:
                # Catch unexpected graph execution errors
                st.session_state.plan_output["error_message"] = f"Graph Execution Failed: {e}"

# --- 4. Output Display ---

output = st.session_state.plan_output

if output["error_message"]:
    st.error(f"Execution Error: {output['error_message']}")

st.divider()
st.header("✨ Your Smart Travel Plan")

col1, col2 = st.columns([1, 2])

# Output Panel: Tourist Spots
with col1:
    st.subheader("1. Top Tourist Spots")
    if output["tourist_spots"]:
        st.write("The agent identified these must-see locations:")
        # Ensure output is displayed as a list if it's successfully parsed
        if isinstance(output["tourist_spots"], list):
            st.markdown(
                "\n".join(f"- **{spot}**" for spot in output["tourist_spots"])
            )
        else:
            st.write(str(output["tourist_spots"])) # Display raw content if parsing was weird
    elif output["itinerary_text"] != "Running...":
        st.warning("No spots identified yet.")

# Output Panel: Itinerary
with col2:
    st.subheader("2. Detailed 1-Day Itinerary")
    if output["itinerary_text"] and output["itinerary_text"] != "Running...":
        st.markdown(output["itinerary_text"])
    else:
        st.write(output["itinerary_text"])


# Final Output Panel: Budget
st.markdown("---")
st.subheader("3. Estimated Budget Summary")

if output["estimated_budget"] and output["estimated_budget"] != "Running...":
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid #1E90FF;">
            <h4 style="margin: 0; color: #1E90FF;">Total Estimated Cost for the Day:</h4>
            <p style="font-size: 24px; font-weight: bold; margin: 5px 0 0 0;">{output['estimated_budget']}</p>
            <p style="font-size: 12px; color: gray;">(Includes entry fees, local transit, and miscellaneous allowance)</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.write(output["estimated_budget"])