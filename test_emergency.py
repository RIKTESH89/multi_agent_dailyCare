"""Test emergency agent routing."""

from agents import healthcare_agent_system

def test_emergency():
    """Test emergency routing."""
    print("Testing emergency agent routing...")
    
    if healthcare_agent_system is None:
        print("Healthcare system not available")
        return
    
    # Emergency test input
    test_input = {
        "messages": [
            {"role": "user", "content": "Fire alarm is going off in my house!"}
        ]
    }
    
    try:
        print("\nQuerying emergency system...")
        result = healthcare_agent_system.invoke(test_input)
        
        print("\nResult received:")
        if "messages" in result:
            for i, msg in enumerate(result["messages"]):
                if hasattr(msg, 'content') and msg.content:
                    print(f"Message {i}: {type(msg).__name__}")
                    # Skip printing content to avoid Unicode issues, just show the structure
                    print(f"  Has content: {len(msg.content)} characters")
        
        print(f"\nTotal messages: {len(result.get('messages', []))}")
        print("Emergency test completed!")
        
    except Exception as e:
        print(f"Error during emergency test: {e}")

if __name__ == "__main__":
    test_emergency()