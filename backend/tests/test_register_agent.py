"""
Test script for Register Agent with OpenAI LLM
"""
import asyncio
import os
import sys

# Add parent directory to path so we can import from agents module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file with override
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from agents.register_agent import get_register_agent, create_register_agent

async def test_register_agent():
    """Test the Register Agent with OpenAI"""
    
    # Check if OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable is required")
        print("Please set this in your .env file")
        return
    
    print("✅ OpenAI API key found")
    
    # Test messages
    test_messages = [
        "Hello!",
        "Hi there, how are you?",
        "I need help finding summer camps",
        "What can you help me with?",
        "Tell me about yourself"
    ]
    
    print(f"\n🧪 Testing Register Agent with OpenAI")
    
    try:
        # Get the Register Agent for OpenAI
        print(f"🔌 Initializing Register Agent (OpenAI)...")
        agent = get_register_agent("openai")
        print(f"✅ Register Agent (OpenAI) initialized successfully!")
        
        # Test with messages
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test {i} (OpenAI) ---")
            print(f"User: {message}")
            
            # Process message
            result = await agent.process_message(message)
            
            print(f"Agent: {result['response']}")
            print(f"Session ID: {result['session_id']}")
            print(f"Provider: {result['llm_provider']}")
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Success!")
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        print(f"\n🎉 OpenAI tests completed!")
        
    except Exception as e:
        print(f"❌ Error testing OpenAI: {e}")
        import traceback
        traceback.print_exc()

async def test_provider_selection():
    """Test the OpenAI provider selection functionality"""
    
    print(f"\n{'='*50}")
    print("Testing OpenAI Provider Selection")
    print(f"{'='*50}")
    
    try:
        print(f"\n🔧 Creating new Register Agent with OpenAI...")
        agent = create_register_agent("openai")
        print(f"✅ Successfully created OpenAI agent")
        print(f"   Provider: {agent.llm_provider}")
        
        # Test a simple message
        result = await agent.process_message("Hello!")
        print(f"   Test response: {result['response'][:50]}...")
        
    except Exception as e:
        print(f"❌ Failed to create OpenAI agent: {e}")

if __name__ == "__main__":
    asyncio.run(test_register_agent())
    asyncio.run(test_provider_selection()) 