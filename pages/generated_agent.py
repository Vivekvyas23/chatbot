# REQUIRED INSTALLATIONS:
# pip install streamlit langgraph langchain-core langchain-google-genai pydantic

import streamlit as st
import os
import json
from datetime import datetime
from typing import TypedDict, List, Union, Dict, Any

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Pydantic Schemas and State Definition ---

# Schema for structured output of identified skills (Node 2)
class IdentifiedSkills(BaseModel):
    """Structured list of critical skills for the specified job role."""
    technical_skills: List[str] = Field(description="Three critical technical competencies for the role.")
    soft_skills: List[str] = Field(description="Three critical soft skills/behaviors for the role.")

# State definition for the LangGraph workflow
class InterviewCoachState(TypedDict):
    """The state object carrying context and results through the graph."""
    job_role: str
    experience_level: str
    timestamp: datetime
    identified_skills: Union[IdentifiedSkills, None]
    generated_questions: List[str]
    final_guide_text: str

# --- 2. Node Functions ---

def input_validation_node(state: InterviewCoachState) -> InterviewCoachState:
    """Node 1: Validates inputs and initializes the state."""
    job_role = state.get("job_role")
    level = state.get("experience_level")

    if not job_role or len(job_role.strip()) < 3:
        raise ValueError("Job role is required and must be descriptive.")
    if not level:
        raise ValueError("Experience level must be selected.")

    st.toast(f"Starting preparation for {level} {job_role}...")
    
    return {
        "job_role": job_role.strip(),
        "experience_level": level,
        "timestamp": datetime.now(),
        # Initialize intermediate fields
        "identified_skills": None,
        "generated_questions": [],
        "final_guide_text": ""
    }

def skill_identifier_node(state: InterviewCoachState, llm: ChatGoogleGenerativeAI) -> InterviewCoachState:
    """Node 2: Uses LLM to identify critical competencies based on role and level."""
    role = state["job_role"]
    level = state["experience_level"]
    
    st.info("Step 1/4: Identifying core competencies for the role...")

    prompt = f"""
    Analyze the role: '{role}' at the '{level}' level.
    Identify the 3 most critical technical competencies and the 3 most critical soft skills required for success in this specific position.
    Structure the output strictly according to the provided JSON schema.
    """
    
    # Use structured output feature for reliable JSON parsing
    structured_llm = llm.with_structured_output(IdentifiedSkills)
    
    response = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {"identified_skills": response}

def question_generator_node(state: InterviewCoachState, llm: ChatGoogleGenerativeAI) -> InterviewCoachState:
    """Node 3: Generates 3 challenging interview questions mapped to the identified skills."""
    skills: IdentifiedSkills = state["identified_skills"]
    role = state["job_role"]
    
    st.info("Step 2/4: Generating challenging, tailored interview questions...")

    skill_list = skills.technical_skills + skills.soft_skills
    
    prompt = f"""
    You are an expert interviewer. The candidate is interviewing for a '{role}' role.
    The most critical skills are: {', '.join(skill_list)}.
    
    Generate exactly three (3) highly relevant, challenging interview questions. 
    Ensure each question is designed to probe at least one of the critical skills listed above.
    
    Output format: A simple JSON list of strings, e.g., ["Question 1?", "Question 2?", "Question 3?"]. Do not include any other text or markdown formatting outside the list.
    """
    
    # Use JSON mode for reliable list output
    llm_json = llm.with_config({"structured_output": {"type": "json"}})
    response = llm_json.invoke([HumanMessage(content=prompt)])
    
    try:
        # The response content should be a JSON string representing the list
        questions = json.loads(response.content)
        if not isinstance(questions, list) or len(questions) != 3:
             # If LLM returns malformed JSON, try splitting by newline as a fallback
             questions = [q.strip() for q in response.content.split('\n') if q.strip()]
             questions = questions[:3]
             if len(questions) < 3:
                 raise ValueError("Could not reliably parse 3 questions.")
    except Exception:
        # Final fallback
        st.warning("Could not parse structured questions. Using generic placeholders.")
        questions = [f"Tell me about a time you handled complexity in {role}", "Describe a project failure.", "How do you prioritize competing demands?"]
        
    return {"generated_questions": questions}

def star_guide_generator_node(state: InterviewCoachState, llm: ChatGoogleGenerativeAI) -> InterviewCoachState:
    """Node 4: Generates the final, detailed STAR method coaching guide."""
    questions = state["generated_questions"]
    skills: IdentifiedSkills = state["identified_skills"]
    role = state["job_role"]
    
    st.info("Step 3/4: Compiling comprehensive STAR method preparation guide...")
    
    guide_parts = []
    
    guide_parts.append(f"# 🚀 Interview Prep Guide: {state['experience_level']} {role}")
    guide_parts.append(f"*(Generated on: {state['timestamp'].strftime('%Y-%m-%d %H:%M')})*")
    guide_parts.append("---")
    
    # Summarize skills
    guide_parts.append("## 🎯 Critical Competencies Targeted")
    guide_parts.append("Based on your role and level, the following skills are essential for your preparation:")
    guide_parts.append(f"**Technical:** {', '.join(skills.technical_skills)}")
    guide_parts.append(f"**Behavioral/Soft:** {', '.join(skills.soft_skills)}")
    guide_parts.append("---")

    
    prompt = f"""
    You are an expert interview coach. For each question provided below, generate a detailed, comprehensive guide on how the candidate should structure their answer using the STAR (Situation, Task, Action, Result) methodology.

    The guidance must be highly specific, advising them what kind of content they should include in the S, T, A, and R sections, ensuring the answer effectively demonstrates the underlying skills required for a '{role}' role.

    Critical Skills: {', '.join(skills.technical_skills + skills.soft_skills)}

    Questions to Address:
    {json.dumps(questions, indent=2)}

    Format the output using Markdown. Structure the response with clear H3 headings for each question, followed by four detailed bulleted lists (S, T, A, R) providing specific coaching points for that question. Do NOT include any introductory or concluding text outside of the guidance for the questions themselves.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    guide_parts.append("## 📝 Detailed STAR Method Coaching")
    guide_parts.append(response.content)

    final_guide = "\n\n".join(guide_parts)
    
    return {"final_guide_text": final_guide}

# --- 3. Graph Initialization ---

def initialize_graph(api_key: str):
    """Initializes the LLM and compiles the sequential LangGraph."""
    
    # Initialize LLM using the provided API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.3,
        convert_to_google_format=True
    )

    # Wrap LLM-dependent node functions to pass the LLM instance
    # Note: LangGraph requires node functions to only accept the state, 
    # but we can wrap them to inject the LLM dependency.
    def skill_id_wrapper(state):
        return skill_identifier_node(state, llm)
    
    def question_gen_wrapper(state):
        return question_generator_node(state, llm)

    def guide_gen_wrapper(state):
        return star_guide_generator_node(state, llm)

    # Build Graph
    builder = StateGraph(InterviewCoachState)

    builder.add_node("validate", input_validation_node)
    builder.add_node("identify_skills", skill_id_wrapper)
    builder.add_node("generate_questions", question_gen_wrapper)
    builder.add_node("generate_guide", guide_gen_wrapper)

    # Define edges (Sequential Flow)
    builder.set_entry_point("validate")
    builder.add_edge("validate", "identify_skills")
    builder.add_edge("identify_skills", "generate_questions")
    builder.add_edge("generate_questions", "generate_guide")
    builder.add_edge("generate_guide", END)

    return builder.compile()

# --- 4. Streamlit UI and Execution ---

def main():
    st.set_page_config(page_title="LangGraph Interview Coach", layout="wide")
    st.title("👨‍💼 AI Job Interview Coach (LangGraph + Gemini)")
    st.markdown("Prepare for your next interview using a structured, multi-step AI agent.")

    # --- Sidebar for API Key ---
    with st.sidebar:
        st.header("Configuration")
        google_api_key = st.text_input(
            "Google Gemini API Key", 
            type="password",
            help="Required for running the LLM nodes (Gemini 2.5 Flash)."
        )
        st.markdown("[Get your Gemini API Key here](https://aistudio.google.com/app/apikey)")

    # --- Main Input Form ---
    if not google_api_key:
        st.warning("Please enter your Google Gemini API Key in the sidebar to proceed.")
        st.stop()
        
    col1, col2 = st.columns(2)
    
    with col1:
        job_role = st.text_input(
            "Target Job Role", 
            placeholder="e.g., Senior Data Scientist, Frontend Developer",
            key="job_role_input"
        )
    
    with col2:
        experience_level = st.selectbox(
            "Experience Level",
            options=["", "Junior", "Mid-Level", "Senior", "Executive/Principal"],
            index=0,
            key="level_select"
        )

    run_button = st.button("🚀 Generate Interview Guide", type="primary", use_container_width=True)
    
    st.markdown("---")

    # --- Execution Logic ---
    if run_button:
        if not job_role or not experience_level:
            st.error("Please fill out both the Job Role and Experience Level.")
            return

        try:
            # Initialize and compile the graph
            interview_coach_app = initialize_graph(google_api_key)
            
            # Initial state input
            initial_state = {
                "job_role": job_role,
                "experience_level": experience_level
            }

            # Run the graph with a spinner for feedback
            with st.spinner("Processing interview guide... (This may take 10-20 seconds as the agent runs all four steps sequentially)"):
                # The invoke call executes the entire graph
                final_state = interview_coach_app.invoke(initial_state)

            st.success("✅ Guide Generation Complete!")
            
            # Store final result in session state for display
            st.session_state['final_guide_text'] = final_state['final_guide_text']
            st.session_state['last_role'] = job_role

        except ValueError as e:
            st.error(f"Input Error: {e}")
        except Exception as e:
            st.error(f"An error occurred during graph execution: {e}")
            st.info("Check your API key and ensure the inputs are valid.")


    # --- Results Display ---
    if 'final_guide_text' in st.session_state:
        st.header(f"Results for {st.session_state.get('last_role', 'Job Role')}")
        st.markdown(st.session_state['final_guide_text'])

if __name__ == "__main__":
    main()