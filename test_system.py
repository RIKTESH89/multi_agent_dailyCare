"""Test the complete multi-agent system."""

from agents import healthcare_agent_system

def test_medication_query():
    """Test a medication-related query."""
    print("Testing medication reminder system...")
    
    if healthcare_agent_system is None:
        print("Healthcare system not available - API key needed")
        return
    
    # Test input
    test_input = {
        "messages": [
            {"role": "user", "content": "Check my medication schedule and remind me about upcoming medicines"}
        ]
    }
    
    try:
        print("\nStreaming response from multi-agent system:")
        print("=" * 50)
        
        for chunk in healthcare_agent_system.stream(test_input):
            for node_name, node_data in chunk.items():
                if "messages" in node_data and node_data["messages"]:
                    latest_message = node_data["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        print(f"[{node_name}]: {latest_message.content}")
                        print("-" * 30)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

def test_emergency_query():
    """Test an emergency-related query."""
    print("\nTesting emergency response system...")
    
    if healthcare_agent_system is None:
        print("Healthcare system not available - API key needed")
        return
    
    # Test input
    test_input = {
        "messages": [
            {"role": "user", "content": "Fire alarm is going off in my house, what should I do?"}
        ]
    }
    
    try:
        print("\nStreaming response from multi-agent system:")
        print("=" * 50)
        
        for chunk in healthcare_agent_system.stream(test_input):
            for node_name, node_data in chunk.items():
                if "messages" in node_data and node_data["messages"]:
                    latest_message = node_data["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        print(f"[{node_name}]: {latest_message.content}")
                        print("-" * 30)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Healthcare Multi-Agent System Integration Test")
    print("=" * 60)
    
    test_medication_query()
    test_emergency_query()
    
    print("\n" + "=" * 60)
    print("Integration tests completed!")