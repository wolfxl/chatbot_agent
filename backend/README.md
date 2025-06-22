# Summer Camp Chatbot Backend

A FastAPI backend for the Summer Camp Chatbot with LangGraph integration, supporting multiple agents and both OpenAI and Gemini LLM providers.

## Features

- 🚀 **FastAPI Backend**: Modern, fast web framework with automatic API documentation
- 🤖 **Multi-Agent System**: Register Agent and Educator Agent with intelligent routing
- 🔄 **Centralized LLM Management**: Shared LLM instances across all agents
- 🌐 **Multi-LLM Support**: Switch between OpenAI and Gemini providers
- 💬 **Session Management**: Maintain conversation context across messages
- 🧠 **Intelligent Routing**: Automatic agent selection based on user intent
- 🌐 **CORS Support**: Ready for frontend integration
- 📊 **Health Monitoring**: Built-in health checks and provider status

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the backend directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Google Gemini Configuration
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-pro

# Supabase Configuration (for future use)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Run the Backend

```bash
# Start the FastAPI server
python main.py
```

The server will start on `http://localhost:8000`

### 4. Test the Backend

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_llm_manager.py -v
python -m pytest tests/test_register_agent.py -v
python -m pytest tests/test_educator_agent.py -v
python -m pytest tests/test_agent_router.py -v
```

## API Endpoints

### Health Check
- **GET** `/` or `/health` - Check backend status and available providers

### Chat
- **POST** `/chat` - Send messages to the chatbot
  ```json
  {
    "message": "Hello! Can you help me find summer camps?",
    "session_id": "optional-session-id",
    "provider": "openai",
    "agent": "register"  // "register", "educator", or null for auto-routing
  }
  ```

### Providers
- **GET** `/providers` - Get available LLM providers and their status

### Agent Suggestions
- **POST** `/suggest-agent` - Get agent suggestions for a message

### Test
- **POST** `/test` - Test the chatbot with a sample message

## Multi-Agent System

### Available Agents

1. **Register Agent** (`register`)
   - General conversation and camp registration assistance
   - Handles questions about camp signup, availability, and general inquiries
   - Keywords: "register", "sign up", "enroll", "application", "availability"

2. **Educator Agent** (`educator`)
   - Specialized in camp selection guidance and educational advice
   - Provides detailed information about choosing the right camp
   - Keywords: "choose", "select", "find", "recommend", "consider", "factors"

### Intelligent Routing

The system automatically routes messages to the most appropriate agent based on:
- **Keyword Detection**: Identifies intent from message content
- **Regex Patterns**: Matches specific phrases and questions
- **Fallback**: Defaults to Register Agent for general conversation

### Manual Agent Selection

You can also specify which agent to use:
```json
{
  "message": "What should I consider when choosing a summer camp?",
  "agent": "educator"
}
```

## Frontend Integration

### HTML Frontend Example

A simple HTML frontend is provided in `../frontend_example.html`. To use it:

1. Start the backend server
2. Open `frontend_example.html` in a web browser
3. The frontend will automatically connect to `http://localhost:8000`

### Frontend Features

- 🎨 **Modern UI**: Beautiful gradient design with responsive layout
- 🔄 **Provider Switching**: Toggle between OpenAI and Gemini
- 🤖 **Agent Selection**: Choose specific agents or use auto-routing
- 💬 **Real-time Chat**: Send messages and receive responses
- 📱 **Mobile Friendly**: Responsive design for all devices
- 🔗 **Connection Status**: Visual indicator of backend connectivity

### Custom Frontend Integration

To integrate with your own frontend:

```javascript
// Example API call with agent selection
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        message: 'What should I consider when choosing a summer camp?',
        provider: 'openai',
        agent: 'educator'  // Optional: let system auto-route if not specified
    })
});

const data = await response.json();
console.log(data.response); // Bot response
console.log(data.agent_type); // Which agent responded
console.log(data.routed_to); // Auto-routing information
```

## Project Structure

```
backend/
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (create this)
├── agents/
│   ├── __init__.py
│   ├── llm_manager.py        # Centralized LLM instance management
│   ├── register_agent.py     # General conversation agent
│   ├── educator_agent.py     # Camp selection guidance agent
│   ├── agent_router.py       # Intelligent agent routing
│   └── state.py              # LangGraph state management
├── tests/
│   ├── test_llm_manager.py   # LLM manager tests
│   ├── test_register_agent.py # Register agent tests
│   ├── test_educator_agent.py # Educator agent tests
│   ├── test_agent_router.py  # Agent router tests
│   └── test_api.py           # API integration tests
├── database/
│   └── supabase_client.py    # Supabase integration
└── README.md
```

## Centralized LLM Management

The system uses a centralized LLM manager that:

### Benefits
- **Shared Instances**: All agents use the same LLM instances
- **Efficient Resource Usage**: No duplicate LLM initialization
- **Provider Management**: Centralized provider status and testing
- **Easy Configuration**: Single point for LLM settings

### Features
- **Automatic Initialization**: LLMs are initialized on first use
- **Provider Testing**: Built-in health checks for each provider
- **Fallback Support**: Automatic fallback to available providers
- **Status Monitoring**: Real-time provider availability status

### Usage
```python
from agents.llm_manager import get_llm, get_available_providers

# Get LLM instance
llm = get_llm("openai")

# Check available providers
providers = get_available_providers()  # ["openai", "gemini"]
```

## LangGraph System

The backend uses LangGraph for stateful conversation management:

### State Management
- `ConversationState`: Holds conversation data, history, and context
- Session tracking across multiple messages
- Error handling and logging

### Agent Architecture
- **RegisterAgent**: General conversation and registration assistance
- **EducatorAgent**: Specialized camp selection guidance
- **AgentRouter**: Intelligent message routing between agents
- Support for multiple LLM providers (OpenAI, Gemini)

### Workflow
1. User sends message to `/chat` endpoint
2. AgentRouter analyzes message intent
3. Message is routed to appropriate agent (Register or Educator)
4. Selected agent processes message with LLM
5. Response and updated state are returned to frontend

## Development

### Adding New Agents

1. Create a new agent class in `agents/` (see `educator_agent.py` for example)
2. Add routing logic to `AgentRouter._detect_intent()`
3. Update the API endpoints to support the new agent
4. Add tests for the new agent

### Adding New LLM Providers

1. Add provider configuration to environment variables
2. Update `LLMManager.initialize()` method
3. Update the `LLMProvider` type definition
4. Add provider-specific tests

### Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific functionality
python -m pytest tests/test_llm_manager.py -v
python -m pytest tests/test_register_agent.py -v
python -m pytest tests/test_educator_agent.py -v
python -m pytest tests/test_agent_router.py -v
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your API keys are correctly set in `.env`
   - Check that the keys are valid and have sufficient credits
   - Use the `/providers` endpoint to check provider status

2. **CORS Errors**
   - The backend includes CORS middleware for common frontend ports
   - Add your frontend URL to `allow_origins` in `main.py` if needed

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're running from the correct directory

4. **Agent Routing Issues**
   - Check the `/suggest-agent` endpoint to see which agent would be selected
   - Use manual agent selection if auto-routing isn't working as expected

5. **LLM Provider Issues**
   - Use the `/providers` endpoint to check provider status
   - Ensure at least one provider is properly configured
   - Check logs for initialization errors

## API Response Format

### Chat Response
```json
{
  "response": "I'd be happy to help you find summer camps!",
  "session_id": "session-123",
  "conversation_history": [...],
  "llm_provider": "openai",
  "agent_type": "register",
  "routed_to": "register",
  "intent_detected": "general_inquiry"
}
```

### Provider Status
```json
{
  "providers": {
    "openai": {
      "available": true,
      "working": true,
      "model": "gpt-3.5-turbo"
    },
    "gemini": {
      "available": true,
      "working": true,
      "model": "gemini-pro"
    }
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.
