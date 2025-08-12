"""Healthcare Multi-Agent System Implementation using LangGraph."""

import json
from typing import Annotated, Literal, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph_supervisor import create_supervisor
import os
from dotenv import load_dotenv
import getpass
from datetime import datetime

load_dotenv()  # Load environment variables from .env file

GEMINI_API_KEY = "####"
GEMINI_MODEL = "gemini-1.5-flash"

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize OpenAI model (will fail gracefully if no API key)
try:
    # model = ChatGoogleGenerativeAI(
    #     model=GEMINI_MODEL, 
    #     google_api_key=GEMINI_API_KEY, 
    #     temperature=0.2
    # )
    model = ChatGroq(
        model="qwen/qwen3-32b",
        temperature = 0.2,
        max_retries=3
    )
except Exception as e:
    print(f"Warning: OpenAI model not initialized - {e}")
    print("Please set OPENAI_API_KEY environment variable")
    model = None

# Mock data for demonstration
MOCK_USER_PROFILE = {
    "age": 52,
    "gender": "Male", 
    "living_situation": "Living alone",
    "medical_history": ["high blood pressure", "diabetes"],
    "allergies": ["peanut", "sunflower"]
}

MOCK_MEDICATION_SCHEDULE = [
    {"medication": "paracetamol", "time": "3:00pm"},
    {"medication": "aspirin", "time": "7:00pm"},
    {"medication": "metformin", "time": "8:00am"},
    {"medication": "lisinopril", "time": "10:00pm"}
]

MOCK_FAMILY_CONTACTS = [
    {"name": "John Smith", "relation": "Son", "phone": "+1-555-0123"},
    {"name": "Mary Smith", "relation": "Daughter", "phone": "+1-555-0456"},
    {"name": "Emergency Contact", "relation": "Neighbor", "phone": "+1-555-0789"}
]

# Medication Reminder Agent Tools
@tool
def get_user_profile() -> Dict[str, Any]:
    """Get user profile including age, gender, medical history, and allergies."""
    return MOCK_USER_PROFILE

@tool  
def get_medication_schedule() -> List[Dict[str, str]]:
    """Get all medications and their scheduled intake times."""
    return MOCK_MEDICATION_SCHEDULE

@tool
def medicine_notification() -> str:
    """Check current time and notify about upcoming medicine (within 10 minutes)."""
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    
    # For demo purposes, let's assume it's 6:50 PM
    demo_time = "6:50pm"
    
    for med in MOCK_MEDICATION_SCHEDULE:
        med_time = med["time"]
        if "7:00pm" in med_time:  # Demo logic for aspirin
            return f"Hi, it's time to take {med['medication'].title()}!"
    
    return "No medications due in the next 10 minutes."

@tool
def medicine_intake_verification() -> str:
    """Verify if user took their medicine. Returns 'Yes' or 'No'."""
    # Mock response - in real implementation this would check with user
    return "No"  # Simulating user hasn't taken medicine

@tool
def health_escalation() -> str:
    """Reconfirm if user has taken medicine. Returns 'Yes' or 'No'."""
    # Mock response for escalation check
    return "No"  # Simulating user still hasn't taken medicine

@tool
def get_family_contacts() -> List[Dict[str, str]]:
    """Get family contacts for emergency notifications."""
    return MOCK_FAMILY_CONTACTS

# Emergency Agent Tools  
@tool
def get_action_plan(emergency_type: str) -> str:
    """Get action plan for emergency situations like 'gas leak', 'fire alarm', 'water burst'."""
    action_plans = {
        "gas leak": "EMERGENCY: Gas leak detected! Evacuate immediately. Do not use electrical switches. Call gas company at 911.",
        "fire alarm": "EMERGENCY: Fire alarm activated! Evacuate the building immediately. Call 911. Do not use elevators.",
        "water burst": "EMERGENCY: Water burst detected! Turn off main water supply. Move to safe area. Call emergency services."
    }
    
    plan = action_plans.get(emergency_type.lower(), f"Unknown emergency type: {emergency_type}")
    
    # Check if this is a critical emergency requiring immediate notification
    if emergency_type.lower() in ["gas leak", "fire alarm"]:
        return f"CRITICAL ALERT: {plan}"
    
    return plan

# Communication Agent Tool
@tool
def send_message(message: str) -> str:
    """Send message/notification to user or contacts."""
    print(f"MESSAGE SENT: {message}")
    return f"Message sent successfully: {message}"

# No custom handoff tools needed - langgraph_supervisor handles this automatically

# Create individual agents (only if model is available)
medication_reminder_agent = None
emergency_agent = None  
communication_agent = None

if model:
    medication_reminder_agent = create_react_agent(
        model=model,
        tools=[
            get_user_profile,
            get_medication_schedule, 
            medicine_notification,
            medicine_intake_verification,
            health_escalation,
            get_family_contacts
        ],
        prompt=(
            "You are a medication reminder agent helping users manage their medicine schedule.\n\n"
            "INSTRUCTIONS:\n"
            "- Help users track and take their medications on time\n"
            "- Check medication schedules and send timely reminders\n"
            "- Verify if user has taken their medicine\n" 
            "- If user hasn't taken medicine after verification and escalation, "
            "recommend contacting family or emergency contacts\n"
            "- Always be caring and supportive\n"
            "- Provide complete responses about medication management"
        ),
        name="medication_reminder_agent"
    )

    emergency_agent = create_react_agent(
        model=model,
        tools=[
            get_user_profile,
            get_action_plan,
            get_family_contacts
        ],
        prompt=(
            "You are an emergency response agent for healthcare safety.\n\n"
            "INSTRUCTIONS:\n"
            "- Respond to emergency situations like gas leaks, fire alarms, water bursts\n"
            "- Provide immediate action plans for emergency scenarios\n"
            "- For critical emergencies, provide clear step-by-step instructions\n"
            "- Always prioritize user safety and quick response\n"
            "- Include family contact information when needed for emergencies"
        ),
        name="emergency_agent"
    )

    communication_agent = create_react_agent(
        model=model,
        tools=[send_message],
        prompt=(
            "You are a communication agent responsible for sending messages and notifications.\n\n"
            "INSTRUCTIONS:\n"
            "- Send messages, alerts, and notifications as requested\n"
            "- Format messages clearly and include all relevant information\n" 
            "- For emergency situations, ensure urgent tone and clear instructions\n"
            "- For medication reminders, be supportive and helpful\n"
            "- Always confirm successful message delivery"
        ),
        name="communication_agent"
    )

def create_multi_agent_supervisor():
    """Create the multi-agent system using langgraph_supervisor."""
    
    if model is None:
        print("Warning: Cannot create agent system - model not initialized")
        return None
    
    # Create list of worker agents
    agents = [medication_reminder_agent, emergency_agent, communication_agent]
    
    # Create supervisor using langgraph_supervisor
    supervisor = create_supervisor(
        model=model,
        agents=agents,
        prompt=(
            "You are a healthcare supervisor managing specialized agents.\n\n"
            "INSTRUCTIONS:\n"
            "- Route user requests to the most appropriate specialized agent\n"
            "- Medication/medicine related queries -> medication_reminder_agent\n"
            "- Emergency/safety situations -> emergency_agent\n" 
            "- Direct communication/messaging needs -> communication_agent\n"
            "- Analyze user input carefully to determine the best agent\n"
            "- Only assign work to one agent at a time\n"
            "- Provide helpful context when delegating tasks"
        ),
        add_handoff_back_messages=True,
        output_mode="full_history"
    )
    
    return supervisor.compile()

# Create the main graph (will be None if no model)
healthcare_agent_system = create_multi_agent_supervisor() if model else None