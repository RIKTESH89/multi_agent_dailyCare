"""Test the Streamlit streaming fix."""

from agents import healthcare_agent_system

def test_streamlit_streaming():
    """Test what the Streamlit app would capture."""
    print("Testing Streamlit streaming extraction...")
    
    if healthcare_agent_system is None:
        print("Healthcare system not available")
        return
    
    # Test input
    test_input = {
        "messages": [
            {"role": "user", "content": "What are my medications?"}
        ]
    }
    
    all_responses = []
    processed_nodes = set()
    
    try:
        print("\nProcessing chunks...")
        
        for chunk_num, chunk in enumerate(healthcare_agent_system.stream(test_input)):
            print(f"\nChunk {chunk_num + 1}:")
            
            for node_name, node_data in chunk.items():
                
                if node_name not in processed_nodes:
                    if node_name == "supervisor":
                        print("  [SUPERVISOR] is routing request...")
                    else:
                        agent_name = node_name.replace('_', ' ').title()
                        print(f"  [{agent_name.upper()}] is working...")
                    processed_nodes.add(node_name)
                
                if "messages" in node_data and node_data["messages"]:
                    messages = node_data["messages"]
                    
                    # Look for meaningful AI messages
                    for msg in reversed(messages):
                        if (hasattr(msg, 'content') and msg.content and 
                            msg.content.strip() and 
                            len(msg.content.strip()) > 10 and
                            not msg.content.startswith("Successfully transferred") and
                            not msg.content.startswith("Transferring") and
                            type(msg).__name__ == "AIMessage"):
                            
                            agent_display_name = node_name.replace('_', ' ').title()
                            response_text = f"**{agent_display_name}**: {msg.content[:100]}..."
                            
                            if response_text not in all_responses:
                                all_responses.append(response_text)
                                print(f"  [RESPONSE] Found response from {agent_display_name}")
                            break
                    
                    # Show tool usage
                    latest_msg = messages[-1] if messages else None
                    if latest_msg and hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                        for tool_call in latest_msg.tool_calls:
                            tool_name = tool_call.get('name', 'unknown_tool')
                            if not tool_name.startswith('transfer'):
                                print(f"  [TOOL] Using tool: {tool_name}")
        
        print(f"\nFinal captured responses: {len(all_responses)}")
        for i, response in enumerate(all_responses):
            print(f"{i+1}. {response}")
        
        print("\nTest completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streamlit_streaming()