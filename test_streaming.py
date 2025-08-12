"""Test the streaming functionality to debug the Streamlit issue."""

from agents import healthcare_agent_system

def test_streaming_debug():
    """Debug the streaming to see what's happening."""
    print("Testing streaming debug...")
    
    if healthcare_agent_system is None:
        print("Healthcare system not available")
        return
    
    # Test input
    test_input = {
        "messages": [
            {"role": "user", "content": "What are my medications?"}
        ]
    }
    
    try:
        print("\nStreaming through supervisor system...")
        print("=" * 60)
        
        chunk_count = 0
        for chunk in healthcare_agent_system.stream(test_input):
            chunk_count += 1
            print(f"\nChunk {chunk_count}:")
            print("-" * 30)
            
            for node_name, node_data in chunk.items():
                print(f"Node: {node_name}")
                
                if "messages" in node_data:
                    messages = node_data["messages"]
                    print(f"  Messages count: {len(messages)}")
                    
                    for i, msg in enumerate(messages):
                        msg_type = type(msg).__name__
                        print(f"  Message {i}: {msg_type}")
                        if hasattr(msg, 'content'):
                            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                            print(f"    Content preview: {repr(content_preview)}")
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"    Tool calls: {len(msg.tool_calls)}")
                            for tc in msg.tool_calls:
                                print(f"      - {tc.get('name', 'unknown')}")
                else:
                    print(f"  No messages in node data: {list(node_data.keys())}")
        
        print(f"\nTotal chunks processed: {chunk_count}")
        print("Streaming debug completed!")
        
    except Exception as e:
        print(f"Error during streaming debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streaming_debug()