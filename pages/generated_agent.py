import streamlit as st
import os
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- Installation Instructions ---
# Run this command in your terminal to install dependencies:
# pip install streamlit langgraph langchain-core langchain-google-genai pydantic
# ---------------------------------

# --- 1. Pydantic Schemas for Structured LLM Output ---

class Skill(BaseModel):
    """Schema for a single identified skill."""
    type: str = Field(description="Either 'Technical' or 'Soft'")
    name: str = Field(description="The name of the skill (e.g., 'Kubernetes', 'Conflict Resolution')")
    justification: str = Field(description="Why this skill is critical for the role.")

# --- 2. LangGraph State Definition ---

class AgentState(TypedDict):
    """
    The state maintained throughout the LangGraph workflow.
    """
    job_role: str
    experience_level: str
    identified_skills: List[Skill]
    generated_questions: List[str]
    final_guide_text: str

# --- 3. Utility Functions ---

def get_llm(api_key: str):
    """Initializes the ChatGoogleGenerativeAI model."""
    if not api_key:
        return None
    
    # Using 'gemini-2.5-flash' for reasoning and speed
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

# --- 4. LangGraph Nodes (Functions) ---

# N0: initial_input_parser (Utility)
def initial_input_parser(state: AgentState) -> AgentState:
    """Validates and standardizes user inputs."""
    with st.status("N0: Parsing initial inputs...", expanded=True) as status:
        
        if not state.get('job_role') or not state.get('experience_level'):
            status.update(label="N0: Input Error", state="error", expanded=False)
            raise ValueError("Job role and experience level must be provided.")
            
        status.update(label=f"N0: Inputs validated for {state['experience_level']} {state['job_role']}.", state="complete", expanded=False)
        return {
            "job_role": state["job_role"],
            "experience_level": state["experience_level"]
        }

# N1: skill_analyzer (LLM Call)
def skill_analyzer(state: AgentState) -> AgentState:
    """Identifies and ranks the top 3 technical and 3 soft skills."""
    with st.status("N1: Analyzing skills required for the role...", expanded=True) as status:
        llm = get_llm(st.session_state.get("llm_api_key"))
        if not llm: return state

        role = state["job_role"]
        level = state["experience_level"]
        
        # Define the parser using the Pydantic list of skills
        parser = JsonOutputParser(pydantic_object=List[Skill])
        format_instructions = parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(f"""You are an expert HR analyst. Your task is to identify the most critical skills for a candidate applying for the role of '{role}' at the '{level}' level.
            You must identify exactly 3 essential technical skills and exactly 3 essential soft skills (total of 6 skills).
            The output MUST be a JSON list following this structure: {format_instructions}"""),
            HumanMessage(f"Analyze the requirements for a {level} {role}.")
        ])

        chain = prompt | llm | parser
        
        try:
            skills_data = chain.invoke({})
            status.update(label=f"N1: Identified {len(skills_data)} skills.", state="complete", expanded=False)
            return {"identified_skills": skills_data}
        except Exception as e:
            status.update(label=f"N1 Error: {e}", state="error", expanded=True)
            raise e

# N2: question_generator (LLM Call)
def question_generator(state: AgentState) -> AgentState:
    """Generates 3 challenging interview questions based on identified skills."""
    with st.status("N2: Generating challenging interview questions...", expanded=True) as status:
        llm = get_llm(st.session_state.get("llm_api_key"))
        if not llm: return state

        role = state["job_role"]
        level = state["experience_level"]
        
        # Format skills for the prompt
        skills_list = [f"- {s.name} ({s.type}): {s.justification}" for s in state["identified_skills"]]
        skills = "\n".join(skills_list)
        
        prompt_text = f"""
        Based on the following required skills for a {level} {role}, generate exactly 3 challenging, open-ended, situational, or behavioral interview questions.
        
        Required Skills:
        {skills}
        
        The questions must be designed to test the candidate's deep proficiency and practical experience related to these skills.
        
        Output Format: A JSON list of 3 strings, where each string is one full question.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage("You are an expert interviewer. Generate highly specific and challenging interview questions."),
            HumanMessage(prompt_text)
        ])

        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        try:
            questions = chain.invoke({})
            # Ensure it's a list of strings and limit to 3
            questions = [str(q) for q in questions if isinstance(q, (str, dict))][:3]
            
            status.update(label=f"N2: Generated {len(questions)} questions.", state="complete", expanded=False)
            return {"generated_questions": questions}
        except Exception as e:
            status.update(label=f"N2 Error: {e}", state="error", expanded=True)
            raise e

# N3: answer_guide_generator (LLM Call)
def answer_guide_generator(state: AgentState) -> AgentState:
    """Creates a detailed STAR method answer guide for each question."""
    with st.status("N3: Creating detailed STAR method answer guides...", expanded=True) as status:
        llm = get_llm(st.session_state.get("llm_api_key"))
        if not llm: return state

        questions = state["generated_questions"]
        role = state["job_role"]
        
        guide_output = []
        
        system_prompt = f"""You are a professional interview coach specializing in the STAR method (Situation, Task, Action, Result).
        For each interview question, generate a detailed, multi-paragraph guide on how the candidate should structure their answer.
        
        The guide MUST explicitly break down the answer structure using the STAR headings (S: Situation, T: Task, A: Action, R: Result).
        
        The guide should provide actionable advice on what kind of content to include in each STAR section to impress the interviewer for a {role} position.
        
        Output MUST be in clean Markdown format."""
        
        
        for i, question in enumerate(questions):
            human_prompt = f"Generate the detailed STAR answer guide for this question: '{question}'"
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(system_prompt),
                HumanMessage(human_prompt)
            ])
            
            chain = prompt | llm | StrOutputParser()
            
            try:
                guide_text = chain.invoke({})
                
                guide_output.append(
                    f"### Question {i+1}: {question}\n\n"
                    f"{guide_text}\n\n"
                    f"---\n"
                )
            except Exception as e:
                st.error(f"Error generating guide for question {i+1}: {e}")
                guide_output.append(f"### Question {i+1}: {question}\n\n *Error generating guide.*\n\n---\n")

        
        final_guide_text = "\n\n".join(guide_output)
        status.update(label="N3: All answer guides generated.", state="complete", expanded=False)
        return {"final_guide_text": final_guide_text}

# N4: final_output_formatter (Utility)
def final_output_formatter(state: AgentState) -> AgentState:
    """Compiles the state data into a clean, markdown-formatted final document."""
    with st.status("N4: Compiling and formatting final output...", expanded=True) as status:
        
        role = state["job_role"]
        level = state["experience_level"]
        skills = state["identified_skills"]
        
        # 1. Header
        header = f"# 🚀 Interview Preparation Guide: {role} ({level})\n\n"
        header += "This personalized guide was generated to help you master challenging behavioral and situational questions using the industry-standard **STAR Method**.\n\n"
        header += "--- \n"
        
        # 2. Identified Skills Section
        skills_section = "## 🎯 Critical Skills Analyzed\n"
        skills_section += f"The analysis identified these key skills essential for success as a {level} {role}:\n\n"
        
        # Ensure skills are handled as dictionaries for formatting
        skills_dicts = [s.model_dump() if isinstance(s, BaseModel) else s for s in skills]
        
        tech_skills = [s for s in skills_dicts if s.get('type') == 'Technical']
        soft_skills = [s for s in skills_dicts if s.get('type') == 'Soft']
        
        skills_section += "### Technical Skills:\n"
        for s in tech_skills:
            skills_section += f"- **{s.get('name', 'N/A')}**: *{s.get('justification', 'No justification provided')}*\n"
            
        skills_section += "\n### Soft Skills:\n"
        for s in soft_skills:
            skills_section += f"- **{s.get('name', 'N/A')}**: *{s.get('justification', 'No justification provided')}*\n"
            
        skills_section += "\n--- \n"

        # 3. Guide Content (from N3)
        guide_content = "## 💬 Targeted Interview Questions & STAR Guides\n"
        guide_content += state["final_guide_text"]
        
        final_text = header + skills_section + guide_content
        
        status.update(label="N4: Formatting complete.", state="complete", expanded=False)
        return {"final_guide_text": final_text}

# --- 5. LangGraph Definition ---

def create_graph():
    """Defines and compiles the sequential LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("initial_input_parser", initial_input_parser)
    workflow.add_node("skill_analyzer", skill_analyzer)
    workflow.add_node("question_generator", question_generator)
    workflow.add_node("answer_guide_generator", answer_guide_generator)
    workflow.add_node("final_output_formatter", final_output_formatter)

    # Define Edges (Sequential Flow)
    workflow.add_edge("initial_input_parser", "skill_analyzer")
    workflow.add_edge("skill_analyzer", "question_generator")
    workflow.add_edge("question_generator", "answer_guide_generator")
    workflow.add_edge("answer_guide_generator", "final_output_formatter")

    # Set Finish Point
    workflow.add_edge("final_output_formatter", END)

    # Set Entry Point
    workflow.set_entry_point("initial_input_parser")

    return workflow.compile()

# --- 6. Streamlit UI and Execution ---

def main():
    st.set_page_config(page_title="Job Interview Coach Agent", layout="wide")
    st.title("⭐️ Job Interview Coach Agent (LangGraph + Gemini)")
    st.caption("A sequential AI agent designed to generate personalized interview prep materials using the STAR method.")

    # Sidebar Configuration (Requirement 3)
    st.sidebar.title("Configuration")
    google_api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", "")
    )
    if google_api_key:
        st.session_state["llm_api_key"] = google_api_key
    else:
        st.sidebar.warning("Please enter your Gemini API Key.")

    # Main UI Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        job_role = st.text_input(
            "1. Enter the specific Job Role:",
            value="Senior Python Developer",
            placeholder="e.g., 'Senior Data Scientist', 'DevOps Manager'"
        )
        
    with col2:
        experience_level = st.selectbox(
            "2. Select Experience Level:",
            options=["Junior", "Mid-Level", "Senior", "Staff", "Principal"],
            index=2
        )

    generate_button = st.button("Generate Personalized Interview Guide", type="primary")

    st.markdown("---")

    if generate_button:
        if not st.session_state.get("llm_api_key"):
            st.error("🛑 Cannot run: Please provide your Google Gemini API Key in the sidebar.")
            return

        if not job_role or not experience_level:
            st.error("🛑 Please fill in both the job role and experience level.")
            return

        # Initialize the graph
        app = create_graph()
        
        # Initial state payload
        initial_state = {
            "job_role": job_role,
            "experience_level": experience_level,
            "identified_skills": [],
            "generated_questions": [],
            "final_guide_text": ""
        }

        st.subheader("Agent Execution Workflow")
        
        try:
            # Invoke the graph
            # Note: LangGraph's invoke is synchronous here, allowing Streamlit to track progress via st.status
            final_state = app.invoke(initial_state)

            # Display Results
            st.subheader("✅ Preparation Guide Complete!")
            st.markdown(final_state["final_guide_text"], unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An unrecoverable error occurred during graph execution. Please check the API key and ensure inputs are valid.")
            st.exception(e)
                
if __name__ == "__main__":
    main()