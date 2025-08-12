"""Test script for individual tools without requiring OpenAI API key."""

from agents import (
    get_user_profile, 
    get_medication_schedule,
    medicine_notification,
    medicine_intake_verification,
    health_escalation,
    get_family_contacts,
    get_action_plan,
    send_message,
    MOCK_USER_PROFILE,
    MOCK_MEDICATION_SCHEDULE,
    MOCK_FAMILY_CONTACTS
)

def test_medication_tools():
    """Test medication-related tools."""
    print("Testing Medication Tools")
    print("=" * 40)
    
    # Test get_user_profile
    print("1. Testing get_user_profile()")
    profile = get_user_profile.invoke({})
    print(f"   Result: {profile}")
    assert profile == MOCK_USER_PROFILE, "User profile test failed"
    print("   PASSED\n")
    
    # Test get_medication_schedule  
    print("2. Testing get_medication_schedule()")
    schedule = get_medication_schedule.invoke({})
    print(f"   Result: {schedule}")
    assert schedule == MOCK_MEDICATION_SCHEDULE, "Medication schedule test failed"
    print("   PASSED\n")
    
    # Test medicine_notification
    print("3. Testing medicine_notification()")
    notification = medicine_notification.invoke({})
    print(f"   Result: {notification}")
    assert isinstance(notification, str), "Notification test failed"
    print("   PASSED\n")
    
    # Test medicine_intake_verification
    print("4. Testing medicine_intake_verification()")
    verification = medicine_intake_verification.invoke({})
    print(f"   Result: {verification}")
    assert verification in ["Yes", "No"], "Verification test failed"
    print("   PASSED\n")
    
    # Test health_escalation
    print("5. Testing health_escalation()")
    escalation = health_escalation.invoke({})
    print(f"   Result: {escalation}")
    assert escalation in ["Yes", "No"], "Escalation test failed"
    print("   PASSED\n")
    
    # Test get_family_contacts
    print("6. Testing get_family_contacts()")
    contacts = get_family_contacts.invoke({})
    print(f"   Result: {contacts}")
    assert contacts == MOCK_FAMILY_CONTACTS, "Family contacts test failed"
    print("   PASSED\n")

def test_emergency_tools():
    """Test emergency-related tools."""
    print("Testing Emergency Tools")
    print("=" * 40)
    
    # Test get_action_plan with different scenarios
    emergency_types = ["fire alarm", "gas leak", "water burst", "unknown emergency"]
    
    for i, emergency in enumerate(emergency_types, 1):
        print(f"{i}. Testing get_action_plan('{emergency}')")
        plan = get_action_plan.invoke({"emergency_type": emergency})
        print(f"   Result: {plan}")
        assert isinstance(plan, str) and len(plan) > 0, f"Action plan test failed for {emergency}"
        if emergency in ["fire alarm", "gas leak"]:
            assert "CRITICAL ALERT" in plan, f"Critical alert test failed for {emergency}"
        print("   PASSED\n")

def test_communication_tools():
    """Test communication-related tools."""
    print("Testing Communication Tools") 
    print("=" * 40)
    
    # Test send_message
    print("1. Testing send_message()")
    test_message = "Test emergency alert: User needs immediate assistance"
    result = send_message.invoke({"message": test_message})
    print(f"   Result: {result}")
    assert isinstance(result, str) and "Message sent successfully" in result, "Send message test failed"
    print("   PASSED\n")

def main():
    """Run all tool tests."""
    print("Healthcare Multi-Agent System - Tool Tests")
    print("=" * 60)
    print()
    
    try:
        test_medication_tools()
        test_emergency_tools()
        test_communication_tools()
        
        print("=" * 60)
        print("All tool tests PASSED!")
        print("The multi-agent system tools are working correctly")
        print("To test the full system, set OPENAI_API_KEY and run the agents")
        
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()