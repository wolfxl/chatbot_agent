"""
Test script for Educator Agent with OpenAI LLM
"""
import asyncio
import os
import sys

# Add parent directory to path so we can import from agents module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file with override
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from agents.educator_agent import get_educator_agent, create_educator_agent

async def test_educator_agent():
    """Test the Educator Agent with educational questions"""
    
    # Check if OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable is required")
        print("Please set this in your .env file")
        return
    
    print("✅ OpenAI API key found")
    
    # Test messages focused on educational guidance
    test_messages = [
        "What should I consider when choosing a summer camp for my 10-year-old?",
        "How do I know if a camp is safe and accredited?",
        "What questions should I ask when evaluating a camp?",
        "What are the most important factors for a child with special needs?",
        "How do I assess the value and cost of different camps?"
    ]
    
    print(f"\n🧪 Testing Educator Agent with OpenAI")
    
    try:
        # Get the Educator Agent for OpenAI
        print(f"🔌 Initializing Educator Agent (OpenAI)...")
        agent = get_educator_agent("openai")
        print(f"✅ Educator Agent (OpenAI) initialized successfully!")
        
        # Test with messages
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test {i} (OpenAI) ---")
            print(f"User: {message}")
            
            # Process message
            result = await agent.process_message(message)
            
            print(f"Agent: {result['response']}")
            print(f"Session ID: {result['session_id']}")
            print(f"Provider: {result['llm_provider']}")
            print(f"Agent Type: {result['agent_type']}")
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Success!")
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        print(f"\n🎉 Educator Agent tests completed!")
        
    except Exception as e:
        print(f"❌ Error testing Educator Agent: {e}")
        import traceback
        traceback.print_exc()

async def test_agent_router():
    """Test the agent router functionality"""
    
    print(f"\n{'='*50}")
    print("Testing Agent Router")
    print(f"{'='*50}")
    
    try:
        from agents.agent_router import AgentRouter
        
        print(f"\n🔧 Creating Agent Router...")
        router = AgentRouter()
        print(f"✅ Agent Router created successfully!")
        
        # Test messages that should route to different agents
        test_cases = [
            {
                "message": "Hello! Can you help me find summer camps?",
                "expected_agent": "register"
            },
            {
                "message": "What should I consider when choosing a camp?",
                "expected_agent": "educator"
            },
            {
                "message": "How do I evaluate camp safety?",
                "expected_agent": "educator"
            },
            {
                "message": "I'm looking for sports camps in my area",
                "expected_agent": "register"
            },
            {
                "message": "What questions should I ask about staff qualifications?",
                "expected_agent": "educator"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Router Test {i} ---")
            print(f"Message: {test_case['message']}")
            print(f"Expected: {test_case['expected_agent']}")
            
            # Test routing
            result = await router.route_message(test_case['message'])
            actual_agent = result.get('routed_to', 'unknown')
            
            print(f"Routed to: {actual_agent}")
            print(f"Response: {result['response'][:100]}...")
            
            if actual_agent == test_case['expected_agent']:
                print("✅ Correct routing!")
            else:
                print("⚠️  Unexpected routing")
        
        print(f"\n🎉 Agent Router tests completed!")
        
    except Exception as e:
        print(f"❌ Error testing Agent Router: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_educator_agent())
    asyncio.run(test_agent_router()) 