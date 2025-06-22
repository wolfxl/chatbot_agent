"""
Test script for the FastAPI backend endpoints
"""
import asyncio
import httpx
import json

# API base URL
BASE_URL = "http://localhost:8000"

async def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                print("✅ Health endpoint working!")
                return True
            else:
                print("❌ Health endpoint failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error testing health endpoint: {e}")
            return False

async def test_providers_endpoint():
    """Test the providers endpoint"""
    print("\n🔍 Testing providers endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/providers")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✅ Providers endpoint working!")
                return True
            else:
                print("❌ Providers endpoint failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error testing providers endpoint: {e}")
            return False

async def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n🔍 Testing chat endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test message
            chat_data = {
                "message": "Hello! Can you help me find summer camps?",
                "provider": "openai"
            }
            
            response = await client.post(
                f"{BASE_URL}/chat",
                json=chat_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✅ Chat endpoint working!")
                return True
            else:
                print("❌ Chat endpoint failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error testing chat endpoint: {e}")
            return False

async def test_educator_agent():
    """Test the Educator Agent specifically"""
    print("\n🔍 Testing Educator Agent...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test educational question
            chat_data = {
                "message": "What should I consider when choosing a summer camp for my child?",
                "provider": "openai",
                "agent": "educator"
            }
            
            response = await client.post(
                f"{BASE_URL}/chat",
                json=chat_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            print(f"Agent Type: {result.get('agent_type')}")
            print(f"Routed To: {result.get('routed_to')}")
            print(f"Response: {result.get('response', '')[:100]}...")
            
            if response.status_code == 200 and result.get('agent_type') == 'educator':
                print("✅ Educator Agent working!")
                return True
            else:
                print("❌ Educator Agent failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error testing Educator Agent: {e}")
            return False

async def test_agent_routing():
    """Test automatic agent routing"""
    print("\n🔍 Testing agent routing...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test messages that should route to different agents
            test_cases = [
                {
                    "message": "Hello! I need help finding camps",
                    "expected_agent": "register"
                },
                {
                    "message": "What safety factors should I consider?",
                    "expected_agent": "educator"
                }
            ]
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n--- Routing Test {i} ---")
                print(f"Message: {test_case['message']}")
                print(f"Expected: {test_case['expected_agent']}")
                
                chat_data = {
                    "message": test_case['message'],
                    "provider": "openai"
                    # No agent specified - should auto-route
                }
                
                response = await client.post(
                    f"{BASE_URL}/chat",
                    json=chat_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    actual_agent = result.get('routed_to', 'unknown')
                    print(f"Routed to: {actual_agent}")
                    
                    if actual_agent == test_case['expected_agent']:
                        print("✅ Correct routing!")
                    else:
                        print("⚠️  Unexpected routing")
                else:
                    print(f"❌ Request failed: {response.status_code}")
            
            return True
                
        except Exception as e:
            print(f"❌ Error testing agent routing: {e}")
            return False

async def test_suggest_agent():
    """Test the suggest-agent endpoint"""
    print("\n🔍 Testing suggest-agent endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test message
            suggest_data = {
                "message": "What should I look for in camp safety?",
                "provider": "openai"
            }
            
            response = await client.post(
                f"{BASE_URL}/suggest-agent",
                json=suggest_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✅ Suggest-agent endpoint working!")
                return True
            else:
                print("❌ Suggest-agent endpoint failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error testing suggest-agent endpoint: {e}")
            return False

async def test_test_endpoint():
    """Test the test endpoint"""
    print("\n🔍 Testing test endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/test")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✅ Test endpoint working!")
                return True
            else:
                print("❌ Test endpoint failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error testing test endpoint: {e}")
            return False

async def test_conversation_flow():
    """Test a full conversation flow"""
    print("\n🔍 Testing conversation flow...")
    
    async with httpx.AsyncClient() as client:
        try:
            # First message
            chat_data1 = {
                "message": "Hi there! I'm looking for summer camps for my 12-year-old.",
                "provider": "openai"
            }
            
            response1 = await client.post(
                f"{BASE_URL}/chat",
                json=chat_data1,
                headers={"Content-Type": "application/json"}
            )
            
            if response1.status_code != 200:
                print("❌ First message failed!")
                return False
            
            result1 = response1.json()
            session_id = result1["session_id"]
            print(f"✅ First message successful! Session ID: {session_id}")
            print(f"Response: {result1['response']}")
            
            # Second message (same session)
            chat_data2 = {
                "message": "What types of camps do you recommend?",
                "session_id": session_id,
                "provider": "openai"
            }
            
            response2 = await client.post(
                f"{BASE_URL}/chat",
                json=chat_data2,
                headers={"Content-Type": "application/json"}
            )
            
            if response2.status_code != 200:
                print("❌ Second message failed!")
                return False
            
            result2 = response2.json()
            print(f"✅ Second message successful!")
            print(f"Response: {result2['response']}")
            print(f"Conversation history length: {len(result2['conversation_history'])}")
            
            return True
                
        except Exception as e:
            print(f"❌ Error testing conversation flow: {e}")
            return False

async def main():
    """Run all tests"""
    print("🧪 Testing FastAPI Backend Endpoints")
    print("=" * 50)
    
    # Test all endpoints
    tests = [
        test_health_endpoint,
        test_providers_endpoint,
        test_chat_endpoint,
        test_educator_agent,
        test_agent_routing,
        test_suggest_agent,
        test_test_endpoint,
        test_conversation_flow
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Backend is ready for frontend connection.")
    else:
        print("⚠️  Some tests failed. Please check the backend setup.")

if __name__ == "__main__":
    asyncio.run(main()) 