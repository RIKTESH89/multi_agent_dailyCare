"""Streamlit Web Application for Healthcare Multi-Agent System."""

import streamlit as st
import asyncio
import time
from datetime import datetime
from agents import healthcare_agent_system, MOCK_USER_PROFILE, MOCK_MEDICATION_SCHEDULE
import os

# Set page config
st.set_page_config(
    page_title="DailyUX Elderly Care AI", 
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_status" not in st.session_state:
    st.session_state.agent_status = ""

def display_agent_thinking(agent_name, tool_name=None):
    """Display thinking status for agents."""
    if tool_name:
        return f"🤖 {agent_name} is thinking and using {tool_name}..."
    else:
        return f"🤖 {agent_name} is processing your request..."

def stream_agent_response(user_input):
    """Stream responses from the multi-agent system."""
    
    if healthcare_agent_system is None:
        st.error("Healthcare agent system not available. Please set API key environment variable and restart.")
        return
    
    # Initialize the conversation
    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    
    # Containers for displaying responses
    response_container = st.empty()
    
    all_responses = []
    processed_nodes = set()
    
    try:
        # Show initial status
        with st.status("Processing your request...", expanded=True) as status:
            
            # Stream through the multi-agent system
            for chunk_num, chunk in enumerate(healthcare_agent_system.stream(initial_state)):
                
                # Process each node in the chunk
                for node_name, node_data in chunk.items():
                    
                    # Show which node is active
                    if node_name not in processed_nodes:
                        if node_name == "supervisor":
                            st.write("**Supervisor** is routing your request...")
                        else:
                            agent_name = node_name.replace('_', ' ').title()
                            st.write(f"**{agent_name}** is working on your request...")
                        processed_nodes.add(node_name)
                    
                    # Check if there are messages in the node data
                    if "messages" in node_data and node_data["messages"]:
                        messages = node_data["messages"]
                        
                        # Look for meaningful AI messages with actual content
                        for msg in reversed(messages):
                            if (hasattr(msg, 'content') and msg.content and 
                                msg.content.strip() and 
                                len(msg.content.strip()) > 10 and  # Must have substantial content
                                not msg.content.startswith("Successfully transferred") and
                                not msg.content.startswith("Transferring") and
                                type(msg).__name__ == "AIMessage"):
                                
                                # This is a meaningful response
                                agent_display_name = node_name.replace('_', ' ').title()
                                response_text = f"**{agent_display_name}**: {msg.content}"
                                
                                # Only add if not already added
                                if response_text not in all_responses:
                                    all_responses.append(response_text)
                                    st.write(f"New response from **{agent_display_name}**")
                                break
                        
                        # Show tool usage
                        latest_msg = messages[-1] if messages else None
                        if latest_msg and hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                            for tool_call in latest_msg.tool_calls:
                                tool_name = tool_call.get('name', 'unknown_tool')
                                if not tool_name.startswith('transfer'):
                                    st.write(f"Using tool: **{tool_name}**")
            
            status.update(label="Request completed!", state="complete")
        
        # Display final response
        if all_responses:
            with response_container.container():
                st.markdown("### Complete Response:")
                for i, response in enumerate(all_responses):
                    st.markdown(response)
                    if i < len(all_responses) - 1:
                        st.markdown("---")
            
            # Add to session state for chat history
            final_response = "\n\n".join(all_responses)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response
            })
        else:
            st.warning("No meaningful response received from the system.")
    
    except Exception as e:
        st.error(f"Error in agent system: {str(e)}")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🏥 DailyUX Elderly Care AI")
    
    # Check if system is available
    if healthcare_agent_system is None:
        st.error("⚠️ Healthcare agent system not available. Please set OPENAI_API_KEY environment variable and restart the application.")
        st.info("To set the API key:\n- Windows: `set OPENAI_API_KEY=your_key_here`\n- Linux/Mac: `export OPENAI_API_KEY=your_key_here`")
        st.stop()
    
    # Sidebar with user info
    with st.sidebar:
        st.header("👤 User Profile")
        st.json(MOCK_USER_PROFILE)
        
        st.header("💊 Medication Schedule")  
        for med in MOCK_MEDICATION_SCHEDULE:
            st.write(f"• **{med['medication']}** at {med['time']}")
        
        st.header("🤖 Available Agents")
        st.write("• **Orchestrator**: Routes requests")
        st.write("• **Medication Reminder**: Manages medicines") 
        st.write("• **Emergency Agent**: Handles safety issues")
        st.write("• **Communication Agent**: Sends messages")
        
        # Quick Action Buttons
        st.header("⚡ Quick Actions")
        st.markdown("""
        <style>
        /* Global button styling for quick actions */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            border: none;
            padding: 14px 18px;
            font-weight: 700;
            font-size: 14px;
            transition: all 0.3s ease;
            cursor: pointer;
            box-shadow: 0 3px 6px rgba(0,0,0,0.15);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
            box-shadow: 0 3px 6px rgba(0,0,0,0.15);
        }
        
        /* Additional hover effect with subtle animation */
        .stButton > button:hover::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        
        /* Quick actions section styling */
        .quick-actions-container {
            margin: 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            border: 1px solid #e1e8ed;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create styled buttons with distinct visual identity
        
        # Medicine Reminder Button - Green theme
        if st.button("💊 Medicine Reminder", 
                    key="medicine_reminder_btn", 
                    help="Check what medications are due now and get reminders", 
                    use_container_width=True,
                    type="primary"):
            # Pre-filled prompt for medicine reminder
            st.session_state.quick_action_prompt = "It's 7:30 PM, John is sitting in his living room with his phone and watch. He is watching TV and needs to be reminded to take the heart medication."
            st.session_state.trigger_quick_action = True
            st.success("🎯 Starting medicine reminder check...")
        
        # Small spacing
        st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
        
        # Forgot Medicine Button - Orange/Red theme  
        if st.button("⚠️ Forgot to Take Medicine", 
                    key="forgot_medicine_btn", 
                    help="Report a missed medication dose and get compliance assistance", 
                    use_container_width=True,
                    type="secondary"):
            # Pre-filled prompt for forgot medicine
            st.session_state.quick_action_prompt = "It's 12:30 PM, John has turned on cooktop and is about to have lunch. He hasn't taken his gastro medicine that should be taken 30 minutes before meals. Please help remind him and check compliance."
            st.session_state.trigger_quick_action = True
            st.warning("⚡ Checking missed medication and compliance...")
        
        # Add information about what these buttons do
        with st.expander("ℹ️ About Quick Actions", expanded=False):
            st.markdown("""
            **💊 Medicine Reminder**: Simulates the evening medication scenario where John needs to take his heart medication while watching TV.
            
            **⚠️ Forgot to Take Medicine**: Simulates the pre-meal scenario where John is cooking but hasn't taken his gastro medicine that should be taken 30 minutes before meals.
            
            Both buttons will automatically start conversations with pre-defined scenarios to demonstrate the healthcare AI system.
            """)
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Main chat interface
    st.header("💬 Chat with DailyUX AI")
    
    # Display existing chat history first
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Handle quick action buttons
    if st.session_state.get("trigger_quick_action", False) and st.session_state.get("quick_action_prompt"):
        prompt = st.session_state.quick_action_prompt
        
        # Show processing indicator
        with st.status(f"🚀 Processing quick action...", expanded=True):
            st.write("📝 Adding your request to the conversation...")
            
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            st.write("🤖 Starting healthcare AI response...")
            
            # Stream agent response
            with st.chat_message("assistant"):
                stream_agent_response(prompt)
        
        # Reset quick action flags
        st.session_state.trigger_quick_action = False
        st.session_state.quick_action_prompt = None
    
    # Chat input
    if prompt := st.chat_input("Ask about medications, emergencies, or send messages..."):
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Stream agent response
        with st.chat_message("assistant"):
            stream_agent_response(prompt)
    

if __name__ == "__main__":
    main()