# To run this application, install the required packages:
# pip install streamlit langgraph langchain-google-genai pydantic
# -------------------------------------------------------------------

import streamlit as st
import os
import json
from typing import TypedDict, Annotated, List

# LangGraph and LangChain components
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. State Management Definition ---

class InterviewCoachState(TypedDict):
    """
    Represents the state of the LangGraph workflow.
    Keys match the requirements outline.
    """
    job_role: str
    experience_level: str
    identified_skills: List[str]
    generated_questions: List[str]
    final_interview_guide_text: str
    status: str

# --- 2. Pydantic Schemas for Structured Output ---

class SkillsList(BaseModel):
    """A list containing 6 critical skills (3 technical, 3 soft) for the role."""
    skills: List[str] = Field(description="A list of exactly 6 identified skills.")

class QuestionsList(BaseModel):
    """A list containing 3 challenging interview questions."""
    questions: List[str] = Field(description="A list of exactly 3 generated interview questions.")

# --- 3. LLM Initialization Helper ---

def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Initializes the Gemini LLM with the provided API key."""
    if not api_key:
        raise ValueError("Gemini API Key is missing.")
    # Using gemini-2.5-flash for fast, capable generation
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.4
    )

def get_llm_from_session() -> ChatGoogleGenerativeAI:
    """Retrieves or initializes the LLM instance using the key stored in session state."""
    api_key = st.session_state.get("gemini_api_key")
    if not api_key:
        raise ValueError("Gemini API Key is missing from session state.")
    return get_llm(api_key)

# --- 4. Node Definitions ---

def Input_Initializer(state: InterviewCoachState) -> dict:
    """Captures and validates user inputs, initializing the state object."""
    st.session_state.status = "Initializing inputs..."
    return {
        "job_role": state["job_role"],
        "experience_level": state["experience_level"],
        "status": "Inputs initialized. Starting skill analysis."
    }

def skill_analyzer(state: InterviewCoachState) -> dict:
    """Node 1: Analyzes role/level to identify critical skills (3 Tech, 3 Soft)."""
    st.session_state.status = "Analyzing critical skills (Node 1)..."
    
    try:
        llm = get_llm_from_session()
    except ValueError as e:
        st.error(str(e))
        return {"status": "Failed due to missing API key."}
        
    role = state["job_role"]
    level = state["experience_level"]
    
    # Use structured output for reliable JSON parsing
    structured_llm = llm.with_structured_output(SkillsList)
    
    prompt = f"""
    You are an expert career coach. Analyze the job role: '{role}' at the '{level}' experience level.
    
    Identify the 6 most critical skills required for success:
    1. Exactly 3 highly relevant Technical/Hard Skills.
    2. Exactly 3 highly relevant Soft/Behavioral Skills.
    
    Return the skills as a JSON list.
    """
    
    try:
        response = structured_llm.invoke([SystemMessage(prompt)])
        identified_skills = response.skills
        if len(identified_skills) != 6:
             st.warning(f"Skill analysis returned {len(identified_skills)} skills, expected 6. Using results.")
             
    except Exception as e:
        st.error(f"Error in skill_analyzer LLM call: {e}")
        # Provide a safe fallback list if structured output fails entirely
        identified_skills = ["Communication", "Problem Solving", "Technical Leadership"] 

    return {
        "identified_skills": identified_skills,
        "status": f"Skills analyzed: {len(identified_skills)} found."
    }

def question_generator(state: InterviewCoachState) -> dict:
    """Node 2: Generates 3 challenging interview questions mapping to the identified skills."""
    st.session_state.status = "Generating challenging questions (Node 2)..."
    
    try:
        llm = get_llm_from_session()
    except ValueError as e:
        st.error(str(e))
        return {"status": "Failed due to missing API key."}

    role = state["job_role"]
    level = state["experience_level"]
    skills = state["identified_skills"]
    
    structured_llm = llm.with_structured_output(QuestionsList)
    
    prompt = f"""
    You are an expert interviewer for the position of {level} {role}.
    
    Based on the following critical skills, generate 3 challenging, specific, and situational/behavioral interview questions:
    Critical Skills: {', '.join(skills)}
    
    Ensure the questions require detailed, structured answers. Return only the 3 questions as a JSON list.
    """
    
    try:
        response = structured_llm.invoke([SystemMessage(prompt)])
        generated_questions = response.questions
        if len(generated_questions) != 3:
            st.warning(f"Question generation returned {len(generated_questions)} questions, expected 3. Using results.")
    except Exception as e:
        st.error(f"Error in question_generator LLM call: {e}")
        generated_questions = ["Tell me about a project failure.", "How do you prioritize competing deadlines?", "Describe a conflict with a peer."] # Safe defaults

    return {
        "generated_questions": generated_questions,
        "status": "Questions generated."
    }

def answer_guide(state: InterviewCoachState) -> dict:
    """Node 3: Creates a detailed response guide for each question, using the STAR Method."""
    st.session_state.status = "Generating detailed STAR method guide (Node 3)..."
    
    try:
        llm = get_llm_from_session()
    except ValueError as e:
        st.error(str(e))
        return {"status": "Failed due to missing API key."}

    role = state["job_role"]
    level = state["experience_level"]
    questions = state["generated_questions"]
    skills = state["identified_skills"]
    
    guide_parts = []
    
    introduction = f"""# 🌟 Interview Coach Guide: {level} {role}\n\n"""
    introduction += f"**Targeted Skills:** {', '.join(skills)}\n\n"
    introduction += "---"
    guide_parts.append(introduction)
    
    for i, question in enumerate(questions):
        prompt = f"""
        You are an expert interview coach. Provide detailed, actionable guidance for answering the following interview question for a {level} {role} position.
        
        Question {i+1}: "{question}"
        
        The guidance MUST explicitly structure the recommended answer approach using the **STAR Method** (Situation, Task, Action, Result).
        
        Provide the output in clean Markdown format.
        """
        
        response = llm.invoke([SystemMessage(prompt)])
        
        markdown_output = f"## Q{i+1}: {question}\n\n"
        markdown_output += response.content
        markdown_output += "\n\n---\n"
        guide_parts.append(markdown_output)

    final_guide = "\n\n".join(guide_parts)
    
    return {
        "final_interview_guide_text": final_guide,
        "status": "Complete. Guide generated."
    }

# --- 5. Graph Construction and Execution ---

def build_and_run_graph(job_role: str, experience_level: str):
    """Builds, compiles, and runs the LangGraph workflow."""
    
    # 1. Define the Graph
    workflow = StateGraph(InterviewCoachState)
    
    # 2. Add Nodes
    workflow.add_node("Input_Initializer", Input_Initializer)
    workflow.add_node("skill_analyzer", skill_analyzer)
    workflow.add_node("question_generator", question_generator)
    workflow.add_node("answer_guide", answer_guide)
    
    # 3. Define Edges (Sequential Flow)
    workflow.set_entry_point("Input_Initializer") 
    
    workflow.add_edge("Input_Initializer", "skill_analyzer")
    workflow.add_edge("skill_analyzer", "question_generator")
    workflow.add_edge("question_generator", "answer_guide")
    
    # End node
    workflow.add_edge("answer_guide", END)
    
    # 4. Compile the Graph
    app = workflow.compile()
    
    # 5. Define Initial State
    initial_state = {
        "job_role": job_role,
        "experience_level": experience_level,
        "identified_skills": [],
        "generated_questions": [],
        "final_interview_guide_text": "",
        "status": "Starting..."
    }
    
    try:
        # 6. Invoke the Graph
        st.session_state.status = "Starting workflow execution..."
        
        # Invoke runs the sequential graph until END
        final_state = app.invoke(initial_state)
        
        # 7. Update UI with final result
        st.session_state.status = final_state["status"]
        st.session_state.final_output = final_state["final_interview_guide_text"]
        
    except Exception as e:
        st.error(f"A critical error occurred during workflow execution: {e}")
        st.session_state.status = "Execution Failed."
        
    st.rerun()

# --- 6. Streamlit Main Application ---

def main():
    st.set_page_config(page_title="Job Interview Coach Agent (LangGraph)", layout="wide")
    st.title("🧠 LangGraph Job Interview Coach Agent")
    st.caption("Personalized interview preparation powered by LangGraph and Google Gemini.")

    # Initialize session state keys if they don't exist
    if 'status' not in st.session_state:
        st.session_state.status = "Awaiting input."
    if 'final_output' not in st.session_state:
        st.session_state.final_output = ""
        
    # --- Sidebar for API Key ---
    with st.sidebar:
        st.header("1. Configuration")
        gemini_api_key = st.text_input(
            "Google Gemini API Key", 
            type="password", 
            help="Required for all LLM processing nodes."
        )
        if gemini_api_key:
            st.session_state["gemini_api_key"] = gemini_api_key
        else:
            st.session_state.pop("gemini_api_key", None)
        
        st.markdown("---")
        st.markdown("### Agent Status")
        st.info(st.session_state.status, icon="⚙️")

    # --- Main Input Area ---
    st.header("2. Define Your Target Role")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        job_role = st.text_input(
            "Job Role", 
            placeholder="e.g., Cloud Architect, UX Designer, Marketing Manager",
            key="job_role_input"
        )
        
    with col2:
        experience_level = st.selectbox(
            "Experience Level",
            options=["Junior", "Mid-Level", "Senior", "Executive"],
            key="level_input"
        )

    # --- Action Button ---
    st.markdown("---")
    
    can_run = job_role and gemini_api_key
    
    if st.button("🚀 Generate Coach Guide", type="primary", use_container_width=True, disabled=not can_run):
        if not gemini_api_key:
            st.error("Please provide your Google Gemini API Key in the sidebar.")
        elif not job_role:
            st.error("Please specify the Job Role.")
        else:
            # Start the graph execution
            build_and_run_graph(job_role, experience_level)
            
    # --- Output Display Area ---
    st.header("3. Interview Guide Output")
    
    if 'final_output' in st.session_state and st.session_state.final_output:
        st.markdown(st.session_state.final_output)
    else:
        st.info("The detailed STAR Method interview guide will appear here after generation. Click the button to start.")

if __name__ == "__main__":
    main()