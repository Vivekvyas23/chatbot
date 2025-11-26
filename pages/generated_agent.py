"""
To run this application, ensure you have the necessary libraries installed:
pip install streamlit langchain-google-genai langgraph pydantic langchain_core
"""
import streamlit as st
import os
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Pydantic Definitions for Structured Output ---

class IdentifiedSkills(BaseModel):
    """Structured list of crucial skills identified for the role."""
    technical: List[str] = Field(description="List of 3 crucial technical skills.")
    soft: List[str] = Field(description="List of 3 crucial soft skills.")

class GeneratedQuestion(BaseModel):
    """A single interview question mapped to the skill it tests."""
    skill_tested: str = Field(description="The specific skill (from technical or soft list) this question targets.")
    question: str = Field(description="The challenging, realistic interview question.")

# Wrapper for robust List generation with structured output
class QuestionList(BaseModel):
    questions: List[GeneratedQuestion]

# --- 2. LangGraph State Schema (TypedDict) ---

class InterviewCoachState(TypedDict):
    """
    The centralized state object for the LangGraph workflow.
    """
    job_role: str
    experience_level: str
    identified_skills: IdentifiedSkills
    generated_questions: List[GeneratedQuestion]
    raw_guidance_text: AnyValue  # Detailed STAR method guidance (String)
    final_prep_guide: str

# --- 3. LLM and Utility Functions ---

def get_llm(api_key: str):
    """Initializes and returns the ChatGoogleGenerativeAI instance."""
    if not api_key:
        raise ValueError("Google Gemini API Key is required.")
    
    # Use gemini-2.5-flash for cost-effectiveness and speed
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

# --- 4. LangGraph Nodes (Functions) ---

def skill_analyzer(state: InterviewCoachState) -> Dict[str, Any]:
    """Node 1: Analyzes role/level and synthesizes crucial skills."""
    st.session_state.status_message = "Analyzing skills required for the role..."
    
    llm = get_llm(st.session_state.api_key)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert AI HR analyst. Analyze the following job role and experience level. "
         "Identify and categorize 3 most crucial technical skills and 3 most crucial soft skills. "
         "Format the output strictly as JSON following the provided schema."
        ),
        ("human", 
         "Job Role: {job_role}\nExperience Level: {experience_level}"
        )
    ])
    
    # Use with_structured_output to enforce Pydantic schema
    chain = prompt_template | llm.with_structured_output(IdentifiedSkills)
    
    result = chain.invoke({
        "job_role": state["job_role"],
        "experience_level": state["experience_level"],
    })
    
    return {"identified_skills": result}

def question_generator(state: InterviewCoachState) -> Dict[str, Any]:
    """Node 2: Generates three challenging questions tied to the identified skills."""
    st.session_state.status_message = "Generating targeted interview questions..."

    llm = get_llm(st.session_state.api_key)
    
    skill_list = state["identified_skills"].technical + state["identified_skills"].soft
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert interviewer. Generate exactly three challenging, realistic interview questions. "
         "Each question must target one specific skill from the provided list. "
         "The output must be a JSON object containing a list of questions following the provided schema."
        ),
        ("human", 
         "Target Skills: {skills}"
        )
    ])
    
    # Use the QuestionList wrapper for robust list generation
    chain = prompt_template | llm.with_structured_output(QuestionList)
    
    result_wrapper = chain.invoke({
        "skills": ", ".join(skill_list),
    })
    
    return {"generated_questions": result_wrapper.questions}

def answer_guide(state: InterviewCoachState) -> Dict[str, Any]:
    """Node 3: Creates detailed answer guidance using the STAR method."""
    st.session_state.status_message = "Developing detailed STAR method answer guidance..."

    llm = get_llm(st.session_state.api_key)
    
    questions_data = "\n".join([
        f"Q{i+1}: {q.question} (Tests: {q.skill_tested})" 
        for i, q in enumerate(state["generated_questions"])
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert career coach. For each provided interview question, "
         "create a detailed, multi-paragraph guide explaining how to structure a successful answer. "
         "Crucially, emphasize and detail the STAR method (Situation, Task, Action, Result) for every answer. "
         "The output should be a single structured text block ready for final formatting."
        ),
        ("human", 
         f"Job Role: {state['job_role']}\nQuestions to guide:\n{questions_data}"
        )
    ])
    
    chain = prompt | llm
    result = chain.invoke({"job_role": state['job_role'], "questions_to_guide": questions_data})
    
    return {"raw_guidance_text": result.content}

def output_formatter(state: InterviewCoachState) -> Dict[str, Any]:
    """Node 4: Compiles all content into the final, readable preparation guide."""
    st.session_state.status_message = "Compiling final preparation guide..."

    llm = get_llm(st.session_state.api_key)
    
    # Prepare inputs for the formatter
    skills_text = (
        "**Technical Skills:** " + ", ".join(state["identified_skills"].technical) + "\n\n"
        "**Soft Skills:** " + ", ".join(state["identified_skills"].soft)
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a professional technical writer. Take the raw content provided "
         "and compile it into a single, cohesive, highly readable interview preparation document. "
         "Use clear Markdown headings, bullet points, and bold text. Ensure the STAR guidance "
         "is clearly associated with the corresponding question."
        ),
        ("human", 
         "--- METADATA ---\n"
         "Job Role: {job_role}\nExperience Level: {experience_level}\n\n"
         "--- IDENTIFIED SKILLS ---\n{skills_text}\n\n"
         "--- RAW GUIDANCE TEXT ---\n{guidance_text}"
        )
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "job_role": state["job_role"],
        "experience_level": state["experience_level"],
        "skills_text": skills_text,
        "guidance_text": state["raw_guidance_text"]
    })
    
    return {"final_prep_guide": result.content}

# --- 5. LangGraph Definition ---

def create_coach_graph():
    """Defines and compiles the linear LangGraph workflow."""
    workflow = StateGraph(InterviewCoachState)

    # Add Nodes
    workflow.add_node("skill_analyzer", skill_analyzer)
    workflow.add_node("question_generator", question_generator)
    workflow.add_node("answer_guide", answer_guide)
    workflow.add_node("output_formatter", output_formatter)

    # Define Edges (Linear Flow)
    workflow.set_entry_point("skill_analyzer") 
    workflow.add_edge("skill_analyzer", "question_generator")
    workflow.add_edge("question_generator", "answer_guide")
    workflow.add_edge("answer_guide", "output_formatter")
    workflow.add_edge("output_formatter", END)
    
    return workflow.compile()

# --- 6. Streamlit UI Implementation ---

st.set_page_config(page_title="Job Interview Coach Agent", layout="wide")
st.title("🌟 AI Job Interview Coach (LangGraph + Gemini)")

# Initialize session state for status and API key
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Ready to start."
if 'api_key' not in st.session_state:
    # Use environment variable if available
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    st.session_state.api_key = st.text_input(
        "Google Gemini API Key", 
        type="password", 
        value=st.session_state.api_key,
        help="Required to run the LLM nodes."
    )
    st.markdown("[Get your Gemini API Key here](https://ai.google.dev/gemini-api/docs/api-key)")

# Main Input Area
col1, col2 = st.columns(2)

with col1:
    job_role = st.text_input(
        "Job Role",
        placeholder="e.g., Senior Data Scientist, Junior UX Designer",
        value="Cloud Security Engineer"
    )

with col2:
    experience_level = st.selectbox(
        "Experience Level",
        options=["Junior", "Mid-Level", "Senior", "Principal"],
        index=2
    )

generate_button = st.button("🚀 Generate Interview Guide", type="primary")

# Status Indicator
status_placeholder = st.empty()
status_placeholder.info(f"Status: {st.session_state.status_message}")

# Output Display
st.subheader("Interview Preparation Guide")
output_placeholder = st.empty()

if generate_button:
    if not st.session_state.api_key:
        status_placeholder.error("Please enter your Google Gemini API Key in the sidebar.")
        st.stop()
        
    if not job_role:
        status_placeholder.error("Please enter a Job Role.")
        st.stop()

    # Initialize the graph and state
    coach_app = create_coach_graph()
    
    initial_state: InterviewCoachState = {
        "job_role": job_role,
        "experience_level": experience_level,
        "identified_skills": None,
        "generated_questions": None,
        "raw_guidance_text": "",
        "final_prep_guide": ""
    }
    
    # Clear previous output
    output_placeholder.empty()

    try:
        # Run the graph and update the status dynamically
        with st.status(f"Starting Coach Agent...", expanded=True) as status:
            
            final_state = initial_state
            
            # Use `stream` to show progress in the status container
            for step in coach_app.stream(initial_state):
                node_name, state_update = list(step.items())[0]
                
                # Update the status message based on the current node
                if node_name != '__end__':
                    # The node function updates st.session_state.status_message
                    status.update(label=f"Processing: {st.session_state.status_message}", state="running")
                
                final_state.update(state_update)

            # Final success message
            status.update(label="✅ Generation Complete!", state="complete")
            status_placeholder.success("Guide successfully generated!")
            
            # Display the final output
            output_placeholder.markdown(final_state["final_prep_guide"])

    except Exception as e:
        status_placeholder.error(f"An error occurred during execution: {e}")
        st.exception(e)