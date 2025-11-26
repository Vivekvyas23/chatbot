# --- INSTALLATION REQUIRED ---
# pip install streamlit langgraph langchain-google-genai pydantic

import streamlit as st
import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Note: JsonOutputParser is imported but the structured call relies on .with_structured_output
from langchain_core.output_parsers import JsonOutputParser 

# --- 1. Pydantic Schemas for Structured Output ---

class Skill(BaseModel):
    """Schema for a single skill."""
    name: str = Field(description="The name of the skill (e.g., 'Python', 'Stakeholder Management').")
    type: str = Field(description="Type of skill: 'Technical' or 'Soft'.")

class SkillList(BaseModel):
    """Schema for the full list of identified skills."""
    list_of_skills: List[Skill] = Field(
        description="A list containing exactly 3 Technical skills and 3 Soft skills relevant to the role."
    )

class QuestionList(BaseModel):
    """Schema for the generated interview questions."""
    list_of_questions: List[str] = Field(
        description="A list of 3 unique, challenging, scenario-based interview questions."
    )

# --- 2. State Management (LangGraph) ---

class InterviewCoachState(TypedDict):
    """
    Represents the state of the interview coaching workflow.
    """
    job_role: str
    experience_level: str
    list_of_skills: List[Skill]
    list_of_questions: List[str]
    final_guide_text: str

# --- 3. LLM Initialization Helper ---

@st.cache_resource
def get_llm(api_key: str):
    """Initializes and caches the ChatGoogleGenerativeAI instance."""
    if not api_key:
        # The key check happens in main(), but this prevents caching an error state
        return None
    try:
        # Use a powerful model for complex reasoning and structured output
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# --- 4. Node Functions (Processing Steps) ---

# Node 1: skill_analyzer
def skill_analyzer(state: InterviewCoachState, config: dict) -> dict:
    st.status("Node 1: Analyzing job requirements and identifying core skills...", state="running")
    llm = config["configurable"]["llm"]
    role = state["job_role"]
    level = state["experience_level"]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert AI systems analyst. Identify the crucial skills for the given role. "
                "You MUST return exactly 3 Technical Skills and 3 Soft Skills. Format the output strictly as JSON following the provided schema.",
            ),
            (
                "human",
                f"Analyze the requirements for a {level} {role}. What are the 6 most critical skills?"
            ),
        ]
    )

    # Use .with_structured_output for robust, type-safe JSON generation with Gemini
    chain = prompt | llm.with_structured_output(SkillList)
    
    try:
        result = chain.invoke({})
        st.status("Node 1: Skill analysis complete.", state="complete")
        return {"list_of_skills": result.list_of_skills}
    except Exception as e:
        st.error(f"Error in skill_analyzer: {e}")
        st.status("Node 1: Failed.", state="error")
        # Return an empty list to prevent downstream nodes from crashing
        return {"list_of_skills": []}

# Node 2: question_generator
def question_generator(state: InterviewCoachState, config: dict) -> dict:
    st.status("Node 2: Generating challenging interview questions...", state="running")
    llm = config["configurable"]["llm"]
    skills = state["list_of_skills"]
    
    if not skills:
        st.status("Node 2: Skipped due to missing skills.", state="error")
        return {"list_of_questions": ["Error: Skills list was empty."]}

    skill_names = [s.name for s in skills]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior interviewer. Based on the provided skills, generate 3 unique, challenging, scenario-based interview questions. "
                "Format the output strictly as JSON following the provided schema.",
            ),
            (
                "human",
                f"Generate 3 challenging questions based on these core skills: {', '.join(skill_names)}."
            ),
        ]
    )

    chain = prompt | llm.with_structured_output(QuestionList)
    
    try:
        result = chain.invoke({})
        st.status("Node 2: Question generation complete.", state="complete")
        return {"list_of_questions": result.list_of_questions}
    except Exception as e:
        st.error(f"Error in question_generator: {e}")
        st.status("Node 2: Failed.", state="error")
        return {"list_of_questions": ["Error: Question generation failed."]}

# Node 3: answer_guide
def answer_guide(state: InterviewCoachState, config: dict) -> dict:
    st.status("Node 3: Compiling final guidance and STAR method tips...", state="running")
    llm = config["configurable"]["llm"]
    questions = state["list_of_questions"]
    skills = state["list_of_skills"]
    role = state["job_role"]

    if not questions or "Error" in questions[0]:
        guide_text = "## Generation Failed\nCannot generate guide due to errors in previous steps."
        return {"final_guide_text": guide_text}

    skill_details = "\n".join([f"* **{s.name}** ({s.type})" for s in skills])
    question_list = "\n".join([f"1. {q}" for q in questions])

    prompt_text = f"""
    Generate a comprehensive interview preparation guide in Markdown format for the role of a {role}.

    ### 1. Core Skills Targeted
    The interviewer will focus on these areas:
    {skill_details}

    ### 2. Practice Interview Questions
    Prepare structured answers for the following challenging questions:
    {question_list}

    ### 3. Essential Guidance: The STAR Method
    Provide a detailed, easy-to-read explanation of the STAR method and why it is critical for behavioral and scenario-based questions. Use clear formatting to emphasize the components (Situation, Task, Action, Result).
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert content generator, specializing in professional interview guides. Ensure the output is formatted beautifully in Markdown."),
        ("human", prompt_text)
    ])

    chain = prompt | llm
    
    try:
        result = chain.invoke({})
        st.status("Node 3: Final guide compiled.", state="complete")
        return {"final_guide_text": result.content}
    except Exception as e:
        st.error(f"Error in answer_guide: {e}")
        st.status("Node 3: Failed.", state="error")
        return {"final_guide_text": "## Generation Failed\nAn error occurred while compiling the final guide."}

# --- 5. LangGraph Definition and Compilation ---

def build_graph(llm):
    """Defines and compiles the linear LangGraph workflow."""
    workflow = StateGraph(InterviewCoachState)

    # Add Nodes, passing the LLM via config
    workflow.add_node("skill_analyzer", lambda state: skill_analyzer(state, {"configurable": {"llm": llm}}))
    workflow.add_node("question_generator", lambda state: question_generator(state, {"configurable": {"llm": llm}}))
    workflow.add_node("answer_guide", lambda state: answer_guide(state, {"configurable": {"llm": llm}}))

    # Define Edges (Linear Flow)
    workflow.set_entry_point(START)
    workflow.add_edge(START, "skill_analyzer")
    workflow.add_edge("skill_analyzer", "question_generator")
    workflow.add_edge("question_generator", "answer_guide")
    workflow.add_edge("answer_guide", END)

    return workflow.compile()

# --- 6. Streamlit UI and Execution ---

def main():
    st.set_page_config(page_title="Job Interview Coach Agent", layout="wide")
    st.title("🧠 Job Interview Coach Agent")
    st.caption("Powered by LangGraph and Google Gemini")

    # --- Sidebar for API Key ---
    with st.sidebar:
        st.header("Configuration")
        google_api_key = st.text_input(
            "Google Gemini API Key", 
            type="password",
            help="Get your key from Google AI Studio.",
            key="gemini_key"
        )
        if google_api_key:
            # Setting environment variable for good measure, though the LLM initialization uses the passed key
            os.environ["GEMINI_API_KEY"] = google_api_key
        
        st.markdown("---")
        st.markdown("### Agent Workflow")
        st.code("""
START -> skill_analyzer
      -> question_generator
      -> answer_guide -> END
        """)

    # --- Main Input Form ---
    st.subheader("Define Your Target Role")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_role = st.text_input(
            "Target Job Role", 
            value="Senior Python Developer", 
            placeholder="e.g., Data Scientist, Product Manager"
        )
    
    with col2:
        experience_level = st.selectbox(
            "Experience Level", 
            options=["Junior", "Mid-Level", "Senior", "Principal"],
            index=2
        )

    run_button = st.button("Generate Interview Guide", type="primary")

    if run_button:
        if not google_api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar to proceed.")
            return

        # Initialize LLM and Graph
        llm = get_llm(google_api_key)
        if not llm:
            # Error message handled inside get_llm if initialization fails for other reasons
            return

        graph = build_graph(llm)
        
        # Initial State
        initial_state = InterviewCoachState(
            job_role=job_role,
            experience_level=experience_level,
            list_of_skills=[],
            list_of_questions=[],
            final_guide_text=""
        )

        st.divider()
        st.subheader("Workflow Execution")
        
        # Execution Block (Loading Indicator)
        with st.container():
            with st.spinner("Running complex multi-step analysis..."):
                try:
                    # Invoke the graph
                    # Note: LangGraph's invoke handles the state transitions sequentially
                    final_state = graph.invoke(initial_state)
                    
                    st.success("Analysis Complete! Guide Generated.")
                    st.balloons()
                    
                    # Display Final Output
                    st.markdown("---")
                    st.subheader(f"Interview Preparation Guide for {experience_level} {job_role}")
                    st.markdown(final_state["final_guide_text"])
                    
                except Exception as e:
                    st.error(f"An error occurred during graph execution: {e}")

if __name__ == "__main__":
    main()