"""Simple test for the supervisor system."""

from agents import healthcare_agent_system

def simple_test():
    """Test the supervisor system with a simple query."""
    print("Testing supervisor system...")
    
    if healthcare_agent_system is None:
        print("Healthcare system not available - model not initialized")
        return
    
    # Simple test input
    test_input = {
        "messages": [
            {"role": "user", "content": "What are my medications?"}
        ]
    }
    
    try:
        print("\nQuerying the system...")
        result = healthcare_agent_system.invoke(test_input)
        
        print("\nResult received:")
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, 'content') and msg.content:
                    try:
                        print(f"Content: {msg.content}")
                    except UnicodeEncodeError:
                        print(f"Content: {repr(msg.content)}")
        else:
            print("No messages in result")
            
        print("\nTest completed!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()