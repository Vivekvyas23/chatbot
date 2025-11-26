"""
REQUIRED INSTALLATION:
pip install streamlit langgraph langchain-google-genai sendgrid
"""

import streamlit as st
import os
from typing import TypedDict, Annotated
import operator
import re

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Attempt to import SendGrid (External Dependency Check)
SENDGRID_AVAILABLE = False
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except ImportError:
    pass # Will be handled in the Streamlit UI and the send_email node

# --- 4. What Information Needs to be Stored in the State ---
class AgentState(TypedDict):
    """
    Represents the state of our graph, carrying data between nodes.
    """
    recipient_email: str
    email_topic: str
    email_subject: str
    email_body: str
    # Use operator.add to append status messages throughout the workflow
    status_message: Annotated[list[str], operator.add]

# --- Helper Functions ---

def is_valid_email(email):
    """Simple regex for basic email validation."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# --- 2. Necessary Nodes (Functions) Required in the Graph ---

def write_email(state: AgentState) -> AgentState:
    """
    Node 1: Uses the LLM (Gemini) to draft a professional email.
    Inputs: email_topic
    Outputs: email_subject, email_body
    """
    st.info("🤖 Node: Generating email draft...")
    
    # Get LLM configuration from session state
    gemini_api_key = st.session_state.get("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API Key is missing.")
        return {"status_message": ["Error: Gemini API Key not set."]}

    try:
        # MANDATORY: Use ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=gemini_api_key,
            temperature=0.3
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return {"status_message": [f"Error initializing LLM: {e}"]}

    topic = state['email_topic']
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a professional email assistant. Write a short, formal email based on the topic. "
         "The output MUST strictly follow the format: 'SUBJECT: [Your Subject Line]\\n\\nBODY: [Your Email Body Content]'. "
         "Ensure the body content starts immediately after 'BODY:' without extra introductory text."),
        ("human", f"Topic: {topic}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({})
        content = response.content
        
        # Parse the structured output using regex
        subject_match = re.search(r"SUBJECT:\s*(.*)", content, re.IGNORECASE)
        body_match = re.search(r"BODY:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        
        subject = subject_match.group(1).strip() if subject_match else "Draft Subject Missing"
        # Clean up the body content
        body = body_match.group(1).strip() if body_match else "Draft body missing."
        
        st.success("✅ Email content generated successfully.")
        
        return {
            "email_subject": subject,
            "email_body": body,
            "status_message": ["Email draft created by LLM."]
        }
    except Exception as e:
        st.error(f"LLM generation failed: {e}")
        return {"status_message": [f"Error during email generation: {e}"]}


def send_email(state: AgentState) -> AgentState:
    """
    Node 2: Uses the SendGrid API to dispatch the email.
    Inputs: recipient_email, email_subject, email_body
    Outputs: status_message
    """
    st.info("📧 Node: Attempting to send email via SendGrid...")
    
    sg_key = st.session_state.get("SENDGRID_API_KEY")
    
    # Check if key is present AND the library is installed
    if not sg_key or not SENDGRID_AVAILABLE:
        msg = f"Error: SendGrid API Key not set or library missing. Email sending SIMULATED only."
        st.warning(msg)
        
        # Log the simulated action based on the draft content for transparency
        sim_msg = (
            f"--- SIMULATION DETAILS ---\n"
            f"To: {state['recipient_email']}\n"
            f"Subject: {state['email_subject']}\n"
            f"Body snippet: {state['email_body'][:50]}..."
        )
        st.code(sim_msg, language="text")
        
        return {"status_message": [msg]}

    recipient = state['recipient_email']
    subject = state['email_subject']
    body = state['email_body']
    
    # NOTE: SendGrid requires a verified sender email. 
    # This must be configured in the environment or assumed.
    SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "test@example.com") 

    try:
        message = Mail(
            from_email=SENDER_EMAIL,
            to_emails=recipient,
            subject=subject,
            # Convert newlines to HTML breaks for robustness
            html_content=body.replace('\n', '<br>') 
        )
        
        sg = SendGridAPIClient(sg_key)
        response = sg.send(message)
        
        if response.status_code == 202:
            final_msg = f"Email sent successfully! Status Code: {response.status_code}"
            st.success(final_msg)
        else:
            # SendGrid API returned non-202 status (e.g., 401 Unauthorized, 400 Bad Request)
            final_msg = f"SendGrid API Error. Status Code: {response.status_code}. Response Body: {response.body.decode()}"
            st.error(final_msg)
            
        return {"status_message": [final_msg]}

    except Exception as e:
        error_msg = f"Error during SendGrid API call: {e}"
        st.error(error_msg)
        return {"status_message": [error_msg]}


# --- LangGraph Definition ---

def create_graph():
    """Defines the sequential workflow of the Email Sender Agent."""
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("write_email", write_email)
    workflow.add_node("send_email", send_email)
    
    # 3. Flow of Data (Edges)
    # START -> write_email (Requirement 8 Check: Uses string node name)
    workflow.set_entry_point("write_email") 
    
    # write_email -> send_email (Default edge)
    workflow.add_edge("write_email", "send_email")
    
    # send_email -> END
    workflow.add_edge("send_email", END)
    
    return workflow.compile()

# --- Streamlit Application ---

def main():
    st.set_page_config(page_title="LangGraph Email Sender Agent", layout="wide")
    st.title("📧 LangGraph Email Sender Agent")
    st.markdown("Automates drafting and delivery using **Gemini** and **SendGrid**.")

    # --- B. Configuration Components (Sidebar) ---
    with st.sidebar:
        st.header("1. API Key Configuration")
        
        # Requirement 3 Check: Gemini Key in sidebar, type=password
        gemini_key = st.text_input(
            "Google Gemini API Key", 
            type="password", 
            key="GEMINI_API_KEY", 
            help="Used by the LLM drafting node."
        )
        
        # Requirement 2 Check: SendGrid Key
        sendgrid_key = st.text_input(
            "SendGrid API Key", 
            type="password", 
            key="SENDGRID_API_KEY", 
            help="Used by the message delivery node."
        )

        st.header("2. Dependencies")
        if not SENDGRID_AVAILABLE:
            st.error("🚨 `sendgrid` library not installed. Delivery will be simulated.")
        else:
            st.success("✅ `sendgrid` library detected.")

    # --- A. Input Components (Primary Execution Interface) ---
    st.header("Agent Inputs")
    col1, col2 = st.columns([2, 1])

    with col1:
        recipient_email = st.text_input(
            "Recipient Email Address",
            placeholder="recipient@example.com",
            key="recipient_email_input"
        )
    
    with col2:
        # Display validation status
        if recipient_email and not is_valid_email(recipient_email):
            st.error("Invalid email format.")
        elif recipient_email:
            st.success("Email format OK.")

    email_topic = st.text_area(
        "Email Topic/Prompt",
        placeholder="Draft a brief formal email requesting a follow-up meeting regarding the Q3 budget.",
        key="email_topic_input"
    )

    run_button = st.button("🚀 Run Email Sender Agent", use_container_width=True, type="primary")

    if run_button:
        # 1. Validation Checks
        if not gemini_key:
            st.error("Please enter the Google Gemini API Key in the sidebar.")
            return
        if not recipient_email or not is_valid_email(recipient_email):
            st.error("Execution stopped: Please enter a valid recipient email.")
            return
        if not email_topic:
            st.error("Execution stopped: Please enter an email topic.")
            return

        # 2. Initial State Setup
        initial_state = AgentState(
            recipient_email=recipient_email,
            email_topic=email_topic,
            email_subject="",
            email_body="",
            status_message=[]
        )

        # 3. Compile and Run Graph
        try:
            app = create_graph()
            
            st.subheader("Workflow Execution Log")
            
            # Invoke the graph to execute the sequential workflow
            final_state = app.invoke(initial_state)

            # --- C. Output Components ---
            st.subheader("Final Agent Results")
            st.markdown("---")

            # Display generated content
            st.markdown(f"**Recipient:** `{final_state['recipient_email']}`")
            st.subheader("Generated Email Content")
            st.code(f"Subject: {final_state['email_subject']}", language="text")
            
            st.markdown("### Email Body:")
            # Replace newlines with markdown line breaks (two spaces + newline)
            st.markdown(final_state['email_body'].replace('\n', '  \n')) 

            # Display final status
            st.subheader("Delivery Status")
            for msg in final_state['status_message']:
                if "Error" in msg or "failed" in msg or "SIMULATED" in msg:
                    st.error(msg)
                elif "sent successfully" in msg:
                    st.success(msg)
                else:
                    st.write(f"- {msg}")

        except Exception as e:
            st.exception(f"An unexpected error occurred during graph execution: {e}")


if __name__ == "__main__":
    main()