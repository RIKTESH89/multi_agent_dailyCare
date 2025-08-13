"""Streamlit Web Application for Healthcare Multi-Agent System."""

import streamlit as st
import asyncio
import time
import threading
from datetime import datetime, timedelta
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
if "scheduled_tasks" not in st.session_state:
    st.session_state.scheduled_tasks = []
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = datetime.now()

def display_agent_thinking(agent_name, tool_name=None):
    """Display thinking status for agents."""
    if tool_name:
        return f"🤖 {agent_name} is thinking and using {tool_name}..."
    else:
        return f"🤖 {agent_name} is processing your request..."

def schedule_delayed_prompt(prompt, delay_minutes=3):
    """Schedule a prompt to be executed after a delay."""
    execute_time = datetime.now() + timedelta(minutes=delay_minutes)
    task = {
        "prompt": prompt,
        "execute_time": execute_time,
        "scheduled_at": datetime.now(),
        "status": "pending"
    }
    st.session_state.scheduled_tasks.append(task)
    return task

def check_scheduled_tasks():
    """Check if any scheduled tasks are ready to execute."""
    current_time = datetime.now()
    pending_tasks = []
    ready_tasks = []
    
    for task in st.session_state.scheduled_tasks:
        if task["status"] == "pending":
            if current_time >= task["execute_time"]:
                ready_tasks.append(task)
                task["status"] = "ready"
            else:
                pending_tasks.append(task)
    
    return ready_tasks, pending_tasks

def execute_scheduled_task(task):
    """Execute a scheduled task."""
    task["status"] = "executing"
    task["executed_at"] = datetime.now()
    
    # Add the scheduled prompt to messages
    st.session_state.messages.append({
        "role": "user", 
        "content": f"[SCHEDULED FOLLOW-UP] {task['prompt']}"
    })
    
    # Trigger the agent response
    st.session_state.trigger_scheduled_action = True
    st.session_state.scheduled_action_prompt = task["prompt"]
    
    task["status"] = "completed"

def stream_agent_response(user_input):
    """Stream responses from the multi-agent system with detailed flow display."""
    
    if healthcare_agent_system is None:
        st.error("Healthcare agent system not available. Please set API key environment variable and restart.")
        return
    
    # Initialize the conversation
    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    
    # Containers for displaying live updates
    live_container = st.container()
    
    # Track all components of the conversation
    conversation_flow = []
    all_responses = []
    processed_nodes = set()
    processed_messages = set()
    
    try:
        # Show initial status
        with st.status("🔄 Processing your healthcare request...", expanded=True) as status:
            
            # Stream through the multi-agent system
            for chunk_num, chunk in enumerate(healthcare_agent_system.stream(initial_state)):
                
                # Process each node in the chunk
                for node_name, node_data in chunk.items():
                    
                    # Show which node is active
                    if node_name not in processed_nodes:
                        agent_display_name = node_name.replace('_', ' ').title()
                        if node_name == "supervisor":
                            status_msg = "🎯 **Supervisor** is analyzing your request and coordinating agents..."
                        else:
                            status_msg = f"🤖 **{agent_display_name}** is processing your healthcare needs..."
                        
                        st.write(status_msg)
                        conversation_flow.append(("status", status_msg))
                        processed_nodes.add(node_name)
                    
                    # Check if there are messages in the node data
                    if "messages" in node_data and node_data["messages"]:
                        messages = node_data["messages"]
                        
                        # Process each message
                        for msg in messages:
                            msg_id = f"{node_name}_{hash(str(msg.content))}"
                            
                            # Skip if we've already processed this exact message
                            if msg_id in processed_messages:
                                continue
                                
                            processed_messages.add(msg_id)
                            
                            # Show tool calls
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call.get('name', 'unknown_tool')
                                    tool_args = tool_call.get('args', {})
                                    
                                    if not tool_name.startswith('transfer'):
                                        tool_display = f"🔧 **{node_name.replace('_', ' ').title()}** is using tool: `{tool_name}`"
                                        if tool_args:
                                            # Show key arguments for context
                                            key_args = {}
                                            for key, value in tool_args.items():
                                                if len(str(value)) < 50:  # Only show short args
                                                    key_args[key] = value
                                            if key_args:
                                                tool_display += f" with parameters: {key_args}"
                                        
                                        st.write(tool_display)
                                        conversation_flow.append(("tool", tool_display))
                            
                            # Show tool results
                            if hasattr(msg, 'content') and msg.content and type(msg).__name__ == "ToolMessage":
                                tool_result = msg.content
                                if len(tool_result) > 200:
                                    tool_result = tool_result[:200] + "..."
                                
                                result_display = f"📋 **Tool Result**: {tool_result}"
                                st.write(result_display)
                                conversation_flow.append(("tool_result", result_display))
                            
                            # Show AI reasoning and responses
                            if (hasattr(msg, 'content') and msg.content and 
                                msg.content.strip() and 
                                len(msg.content.strip()) > 10 and 
                                not msg.content.startswith("Successfully transferred") and
                                not msg.content.startswith("Transferring") and
                                type(msg).__name__ == "AIMessage"):
                                
                                agent_display_name = node_name.replace('_', ' ').title()
                                response_text = f"💬 **{agent_display_name}**: {msg.content}"
                                
                                # Only add if not already added
                                if response_text not in all_responses:
                                    all_responses.append(response_text)
                                    st.write(f"✨ **New response from {agent_display_name}**")
                                    conversation_flow.append(("response", response_text))
                                    
                                    # Also show a preview of the response
                                    preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                                    st.markdown(f"> {preview}")
                                    conversation_flow.append(("preview", f"> {preview}"))
            
            status.update(label="✅ Healthcare analysis completed!", state="complete")
        
        # Display comprehensive conversation flow
        with st.expander("🔍 **Detailed Agent Flow & Reasoning**", expanded=False):
            st.markdown("### 📊 Complete Multi-Agent Conversation Flow")
            st.markdown("This shows the step-by-step reasoning and tool usage from each healthcare agent:")
            
            for i, (item_type, content) in enumerate(conversation_flow):
                if item_type == "status":
                    st.markdown(f"**Step {i+1}:** {content}")
                elif item_type == "tool":
                    st.markdown(f"**Step {i+1}:** {content}")
                elif item_type == "tool_result":
                    st.markdown(f"**Step {i+1}:** {content}")
                elif item_type == "response":
                    st.markdown(f"**Step {i+1}:** {content}")
                elif item_type == "preview":
                    continue  # Skip previews in detailed view
                
                st.markdown("---")
        
        # Display final summary response
        st.markdown("### 🎯 **Healthcare AI Summary**")
        if all_responses:
            for i, response in enumerate(all_responses):
                st.markdown(response)
                if i < len(all_responses) - 1:
                    st.markdown("---")
            
            # Add comprehensive response to session state for chat history
            # Include both the flow and the final responses
            flow_summary = "**Healthcare Agent Analysis Flow:**\n\n"
            for item_type, content in conversation_flow:
                if item_type in ["status", "tool", "response"]:
                    flow_summary += f"{content}\n\n"
            
            final_responses = "**Final Healthcare Recommendations:**\n\n" + "\n\n".join(all_responses)
            
            complete_response = flow_summary + "\n" + final_responses
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_responses  # Store only final responses for clean chat history
            })
        else:
            st.warning("No meaningful response received from the healthcare system.")
    
    except Exception as e:
        st.error(f"Error in healthcare agent system: {str(e)}")
        with st.expander("🔍 Error Details"):
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
            # Initial prompt for medicine reminder
            initial_prompt = "It's 7:30 PM, John is sitting in his living room with his phone and watch. He is watching TV and needs to be reminded to take the heart medication."
            
            # Follow-up prompt after 3 minutes
            followup_prompt = "It's now 8:00 PM and 30 minutes have passed since the heart medication reminder was sent to John. The system has not detected John taking the medication through contact sensors or visual tracking. Please send a follow-up alert asking John 'Have you had your heart medicine? Don't forget to have it'."
            
            # Execute initial prompt immediately
            st.session_state.quick_action_prompt = initial_prompt
            st.session_state.trigger_quick_action = True
            
            # Schedule the follow-up prompt for 3 minutes later
            task = schedule_delayed_prompt(followup_prompt, delay_minutes=3)
            
            st.success("🎯 Starting medicine reminder check...")
            st.info(f"⏰ Follow-up compliance check scheduled for {task['execute_time'].strftime('%H:%M:%S')} (3 minutes from now)")
        
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
            **💊 Medicine Reminder**: 
            - **Immediate**: Simulates 7:30 PM heart medication reminder while John is watching TV
            - **⏰ Scheduled**: Automatically triggers a follow-up compliance check after 3 minutes (8:00 PM scenario)
            - **Smart Flow**: Demonstrates progressive escalation from reminder → follow-up → potential family notification
            
            **⚠️ Forgot to Take Medicine**: 
            - Simulates the pre-meal scenario where John is cooking but hasn't taken his gastro medicine that should be taken 30 minutes before meals.
            
            **🔄 Scheduler System**: 
            The Medicine Reminder button demonstrates our intelligent scheduling system that automatically follows up on medication reminders to ensure patient compliance!
            """)
        
        # Scheduler Status Display
        ready_tasks, pending_tasks = check_scheduled_tasks()
        if pending_tasks or ready_tasks:
            st.header("⏰ Scheduled Tasks")
            
            if pending_tasks:
                st.subheader("🔄 Pending Tasks")
                for i, task in enumerate(pending_tasks):
                    time_left = task["execute_time"] - datetime.now()
                    minutes_left = max(0, int(time_left.total_seconds() / 60))
                    seconds_left = max(0, int(time_left.total_seconds() % 60))
                    
                    st.markdown(f"""
                    **Task {i+1}:** Follow-up compliance check  
                    ⏱️ **Time remaining:** {minutes_left}m {seconds_left}s  
                    🎯 **Scheduled for:** {task['execute_time'].strftime('%H:%M:%S')}  
                    📝 **Action:** Medication compliance follow-up
                    """)
                    
                    # Progress bar showing time until execution
                    total_wait = (task["execute_time"] - task["scheduled_at"]).total_seconds()
                    elapsed = (datetime.now() - task["scheduled_at"]).total_seconds()
                    progress = min(1.0, elapsed / total_wait) if total_wait > 0 else 1.0
                    st.progress(progress)
            
            if ready_tasks:
                st.subheader("✅ Ready to Execute")
                for task in ready_tasks:
                    st.success(f"Task ready: {task['prompt'][:50]}...")
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Main chat interface
    st.header("💬 Chat with DailyUX AI")
    
    # Auto-refresh mechanism for scheduled tasks
    current_time = datetime.now()
    if (current_time - st.session_state.last_check_time).total_seconds() >= 5:  # Check every 5 seconds
        st.session_state.last_check_time = current_time
        ready_tasks, pending_tasks = check_scheduled_tasks()
        
        # If we have pending tasks, auto-refresh the page
        if pending_tasks:
            time.sleep(1)  # Small delay
            st.rerun()
    
    # Check for scheduled tasks that are ready to execute
    ready_tasks, _ = check_scheduled_tasks()
    for task in ready_tasks:
        execute_scheduled_task(task)
    
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
    
    # Handle scheduled action execution
    if st.session_state.get("trigger_scheduled_action", False) and st.session_state.get("scheduled_action_prompt"):
        scheduled_prompt = st.session_state.scheduled_action_prompt
        
        # Show processing indicator for scheduled task
        with st.status("⏰ Executing scheduled follow-up...", expanded=True):
            st.write("🔄 Processing scheduled medication compliance check...")
            st.write("📅 This was automatically triggered based on your earlier medication reminder")
            
            # Display scheduled message
            with st.chat_message("user"):
                st.write(f"[SCHEDULED FOLLOW-UP] {scheduled_prompt}")
            
            st.write("🤖 Starting automated healthcare compliance check...")
            
            # Stream agent response for scheduled task
            with st.chat_message("assistant"):
                stream_agent_response(scheduled_prompt)
        
        # Reset scheduled action flags
        st.session_state.trigger_scheduled_action = False
        st.session_state.scheduled_action_prompt = None
    
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