"""Test script for the healthcare multi-agent system."""

import os
from agents import healthcare_agent_system

def test_medication_reminder():
    """Test medication reminder functionality."""
    print("🧪 Testing Medication Reminder Agent...")
    
    test_input = {"messages": [{"role": "user", "content": "Check my medication schedule and remind me about upcoming medicines"}]}
    
    try:
        for chunk in healthcare_agent_system.stream(test_input):
            for node_name, node_data in chunk.items():
                if "messages" in node_data and node_data["messages"]:
                    latest_message = node_data["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        print(f"[{node_name}]: {latest_message.content}")
        print("✅ Medication reminder test passed\n")
    except Exception as e:
        print(f"❌ Medication reminder test failed: {e}\n")

def test_emergency_agent():
    """Test emergency agent functionality."""
    print("🧪 Testing Emergency Agent...")
    
    test_input = {"messages": [{"role": "user", "content": "Fire alarm is going off in my house"}]}
    
    try:
        for chunk in healthcare_agent_system.stream(test_input):
            for node_name, node_data in chunk.items():
                if "messages" in node_data and node_data["messages"]:
                    latest_message = node_data["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        print(f"[{node_name}]: {latest_message.content}")
        print("✅ Emergency agent test passed\n")
    except Exception as e:
        print(f"❌ Emergency agent test failed: {e}\n")

def test_communication_agent():
    """Test communication agent functionality."""
    print("🧪 Testing Communication Agent...")
    
    test_input = {"messages": [{"role": "user", "content": "Send an urgent message to my family about my medication"}]}
    
    try:
        for chunk in healthcare_agent_system.stream(test_input):
            for node_name, node_data in chunk.items():
                if "messages" in node_data and node_data["messages"]:
                    latest_message = node_data["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        print(f"[{node_name}]: {latest_message.content}")
        print("✅ Communication agent test passed\n")
    except Exception as e:
        print(f"❌ Communication agent test failed: {e}\n")

def main():
    """Run all tests."""
    print("🏥 Healthcare Multi-Agent System Tests\n")
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️ Please set your OPENAI_API_KEY environment variable")
        return
    
    print("=" * 60)
    test_medication_reminder()
    test_emergency_agent() 
    test_communication_agent()
    print("=" * 60)
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()