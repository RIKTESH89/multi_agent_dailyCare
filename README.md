# DailyUX Elderly Care AI - Healthcare Multi-Agent System

A sophisticated healthcare management system built with LangGraph that uses multiple specialized AI agents to handle medication reminders, emergency situations, and communications.

## 🏗️ Architecture

The system uses the **LangGraph Supervisor** pattern with:

- **Supervisor Agent**: Intelligent routing to specialized agents using `langgraph_supervisor`
- **Medication Reminder Agent**: Manages medication schedules and compliance
- **Emergency Agent**: Handles safety and emergency situations  
- **Communication Agent**: Sends messages and notifications

## 🤖 Agents & Tools

### Medication Reminder Agent
- `get_user_profile()`: Returns user medical profile and allergies
- `get_medication_schedule()`: Gets medication times and names
- `medicine_notification()`: Checks for upcoming medicines (10-min window)
- `medicine_intake_verification()`: Verifies if user took medicine
- `health_escalation()`: Re-confirms medicine intake
- `get_family_contacts()`: Gets emergency contact information

### Emergency Agent  
- `get_user_profile()`: Returns user medical profile
- `get_action_plan()`: Provides action plans for emergencies (gas leak, fire, water burst)
- `get_family_contacts()`: Gets emergency contacts for critical situations

### Communication Agent
- `send_message()`: Sends notifications and alerts

## 🚀 Setup & Installation

1. **Clone and Navigate**:
   ```bash
   cd multi_agent
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac  
   export OPENAI_API_KEY=your_api_key_here
   ```

4. **Test the System**:
   ```bash
   python test_agents.py
   ```

5. **Run Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 💻 Web Interface Features

- **Real-time Streaming**: See agents thinking and using tools
- **Interactive Chat**: Natural language interface for all requests
- **Quick Actions**: Pre-built buttons for common scenarios
- **Agent Status**: Visual indicators showing which agent is active
- **User Profile Display**: Shows current medication schedule and profile

## 🔧 Usage Examples

### Medication Management
```
"Check my medication schedule"
"Remind me about my evening medicines" 
"I haven't taken my aspirin yet"
```

### Emergency Situations
```
"Fire alarm is going off"
"I smell gas in my house"  
"Water pipe burst in basement"
```

### Communication
```
"Contact my family about medication compliance"
"Send urgent message to emergency contacts"
```

## 🎯 Key Features

- **LangGraph Supervisor** for intelligent agent routing and handoffs
- **Streaming responses** with real-time agent status
- **Emergency escalation** with family notifications
- **Medication compliance** tracking and reminders
- **Mock data** for demonstration purposes

## 🧪 Testing

The system includes comprehensive tests for each agent:

```bash
python test_agents.py
```

This will test:
- Medication reminder workflows
- Emergency response protocols  
- Communication agent functionality
- Agent-to-agent handoffs

## 🔒 Security Notes

- Uses mock data for demonstration
- In production, integrate with real medical databases
- Implement proper authentication and encryption
- Follow HIPAA compliance guidelines for medical data

## 🛠️ Customization

- Modify `MOCK_USER_PROFILE` for different user scenarios
- Update `MOCK_MEDICATION_SCHEDULE` for various medication regimens
- Extend emergency types in `get_action_plan()`
- Add new tools and agents as needed

## 📱 Running the App

Once you run `streamlit run streamlit_app.py`, the web interface will be available at `http://localhost:8501` with:

- Interactive chat interface
- Real-time agent status updates  
- Pre-built quick action buttons
- Streaming responses showing agent thinking processes

## 🔄 Workflow Example

1. User asks: "Fire alarm going off"
2. **Orchestrator** routes to **Emergency Agent**
3. **Emergency Agent** calls `get_action_plan("fire alarm")`
4. **Emergency Agent** transfers to **Communication Agent** with urgent message
5. **Communication Agent** calls `send_message()` with emergency details
6. System returns comprehensive emergency response

This creates a natural, conversational healthcare assistant that can handle both routine medication management and critical emergency situations.