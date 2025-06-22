"""
Main FastAPI application entry point for the Summer Camp Chatbot
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import logging
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Import our agent
from agents.register_agent import get_register_agent, create_register_agent
from agents.educator_agent import get_educator_agent, create_educator_agent
from agents.agent_router import AgentRouter
from agents.llm_manager import initialize_llm_manager, get_provider_status, get_available_providers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Summer Camp Chatbot API",
    description="API for the Summer Camp Chatbot with LangGraph and LLM integration",
    version="1.0.0"
)

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    provider: Optional[str] = "openai"  # Default to OpenAI
    agent: Optional[str] = None  # "register", "educator", or None for auto-routing

class ChatResponse(BaseModel):
    response: str
    formatted_response: str = None
    session_id: str
    conversation_history: list
    llm_provider: str
    agent_type: str
    routed_to: Optional[str] = None
    intent_detected: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    llm_providers: Dict[str, bool]

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check and provider status"""
    try:
        # Initialize LLM manager
        initialize_llm_manager()
        
        # Get provider status from centralized manager
        provider_status = get_provider_status()
        
        # Convert to simple boolean format for backward compatibility
        providers = {}
        for provider, status in provider_status.items():
            providers[provider] = status["available"] and status["working"]
        
        return HealthResponse(
            status="healthy",
            message="Summer Camp Chatbot API is running",
            llm_providers=providers
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - process user messages through the appropriate agent
    
    Args:
        request: ChatRequest containing message and optional session_id
        
    Returns:
        ChatResponse with agent response and session info
    """
    try:
        # Validate provider
        if request.provider not in ["openai", "gemini"]:
            raise HTTPException(status_code=400, detail="Invalid provider. Use 'openai' or 'gemini'")
        
        # Validate agent if specified
        if request.agent and request.agent not in ["register", "educator"]:
            raise HTTPException(status_code=400, detail="Invalid agent. Use 'register', 'educator', or omit for auto-routing")
        
        # Initialize agent router
        try:
            router = AgentRouter()
        except Exception as e:
            logger.error(f"Failed to initialize agent router: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Failed to initialize agents. Please check your API keys."
            )
        
        # Process the message
        logger.info(f"Processing message with provider {request.provider}, agent {request.agent or 'auto'}: {request.message[:50]}...")
        
        # Route message to appropriate agent
        result = await router.route_message(
            message=request.message,
            session_id=request.session_id,
            preferred_agent=request.agent
        )
        
        # Check for errors
        if result.get("error"):
            logger.error(f"Agent error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return the response
        return ChatResponse(
            response=result["response"],
            formatted_response=result.get("formatted_response", None),
            session_id=result["session_id"],
            conversation_history=result["conversation_history"],
            llm_provider=result["llm_provider"],
            agent_type=result["agent_type"],
            routed_to=result.get("routed_to"),
            intent_detected=result.get("intent_detected"),
            error=result.get("error")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/providers")
async def get_providers():
    """Get available LLM providers and their status"""
    try:
        # Initialize LLM manager
        initialize_llm_manager()
        
        # Get detailed provider status from centralized manager
        provider_status = get_provider_status()
        
        return {"providers": provider_status}
        
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test")
async def test_agent():
    """Test endpoint to verify agent functionality"""
    try:
        # Initialize LLM manager and agent router
        initialize_llm_manager()
        router = AgentRouter()
        
        # Test with a simple message
        test_message = "Hello! Can you help me find summer camps?"
        
        # Use the agent router to process the message
        result = await router.route_message(
            message=test_message,
            session_id=None,
            preferred_agent=None  # Let it auto-route
        )
        
        return {
            "status": "success",
            "provider": result["llm_provider"],
            "agent_type": result["agent_type"],
            "response": result["response"],
            "session_id": result["session_id"],
            "routed_to": result.get("routed_to"),
            "intent_detected": result.get("intent_detected")
        }
                
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest-agent")
async def suggest_agent(request: ChatRequest):
    """Get agent suggestions for a message"""
    try:
        # Initialize agent router
        try:
            router = AgentRouter()
        except Exception as e:
            logger.error(f"Failed to initialize agent router: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Failed to initialize agents. Please check your API keys."
            )
        
        # Get suggestions
        suggestions = await router.get_agent_suggestions(request.message)
        
        return {
            "message": request.message,
            "suggestions": suggestions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 