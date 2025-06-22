"""
Agent Router - Routes messages between different agents based on user intent
"""
import re
from typing import Dict, Any, Literal
from agents.register_agent import get_register_agent
from agents.educator_agent import get_educator_agent
from agents.state import ConversationState
from agents.llm_manager import initialize_llm_manager, get_available_providers
import logging

logger = logging.getLogger(__name__)

# Type for agent selection
AgentType = Literal["register", "educator"]


class AgentRouter:
    """Routes messages to appropriate agents based on user intent"""
    
    def __init__(self):
        """Initialize the agent router"""
        # Initialize the LLM manager first
        initialize_llm_manager()
        
        # Get available providers
        available_providers = get_available_providers()
        if not available_providers:
            raise ValueError("No LLM providers available")
        
        # Use the first available provider (prefer OpenAI)
        default_provider = "openai" if "openai" in available_providers else available_providers[0]
        
        self.register_agent = get_register_agent(default_provider)
        self.educator_agent = get_educator_agent(default_provider)
        
        logger.info(f"Agent Router initialized with provider: {default_provider}")
        logger.info(f"Available providers: {available_providers}")
    
    def _detect_intent(self, message: str) -> AgentType:
        """
        Detect user intent to determine which agent should handle the message
        
        Args:
            message: User's message
            
        Returns:
            AgentType: Which agent should handle the message
        """
        message_lower = message.lower()
        
        # Keywords that indicate educational/guidance needs
        educator_keywords = [
            'what to consider', 'how to choose', 'what should i look for',
            'safety', 'accreditation', 'staff', 'qualifications', 'cost',
            'budget', 'value', 'special needs', 'medical', 'health',
            'communication', 'reviews', 'references', 'checklist',
            'tips', 'advice', 'guidance', 'considerations', 'factors',
            'questions to ask', 'evaluate', 'assess', 'compare',
            'age appropriate', 'developmental', 'program types',
            'location', 'logistics', 'transportation', 'duration'
        ]
        
        # Check if message contains educator-related keywords
        for keyword in educator_keywords:
            if keyword in message_lower:
                logger.info(f"Detected educator intent: '{keyword}' in message")
                return "educator"
        
        # Check for specific question patterns
        educator_patterns = [
            r'what.*consider.*camp',
            r'how.*choose.*camp',
            r'what.*look.*for',
            r'should.*ask',
            r'important.*factors',
            r'key.*considerations',
            r'what.*know.*before',
            r'questions.*ask',
            r'checklist',
            r'tips.*advice'
        ]
        
        for pattern in educator_patterns:
            if re.search(pattern, message_lower):
                logger.info(f"Detected educator intent via pattern: '{pattern}'")
                return "educator"
        
        # Default to register agent for general conversation
        logger.info("Defaulting to register agent for general conversation")
        return "register"
    
    async def route_message(
        self, 
        message: str, 
        session_id: str = None,
        preferred_agent: AgentType = None
    ) -> Dict[str, Any]:
        """
        Route a message to the appropriate agent
        
        Args:
            message: User's message
            session_id: Optional session ID
            preferred_agent: Optional preferred agent override
            
        Returns:
            Dict containing agent response and metadata
        """
        try:
            # Determine which agent should handle the message
            if preferred_agent:
                agent_type = preferred_agent
                logger.info(f"Using preferred agent: {agent_type}")
            else:
                agent_type = self._detect_intent(message)
                logger.info(f"Detected agent type: {agent_type}")
            
            # Route to appropriate agent
            if agent_type == "educator":
                logger.info("Routing to Educator Agent")
                result = await self.educator_agent.process_message(message, session_id)
            else:
                logger.info("Routing to Register Agent")
                result = await self.register_agent.process_message(message, session_id)
            
            # Add routing metadata
            result["routed_to"] = agent_type
            result["intent_detected"] = agent_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error in agent routing: {e}")
            return {
                "response": "I apologize, but I encountered an error routing your message. Please try again.",
                "session_id": session_id or "error",
                "llm_provider": "unknown",
                "agent_type": "error",
                "routed_to": "error",
                "intent_detected": "error",
                "error": str(e)
            }
    
    async def get_agent_suggestions(self, message: str) -> Dict[str, Any]:
        """
        Get suggestions for which agent might be best for a message
        
        Args:
            message: User's message
            
        Returns:
            Dict with agent suggestions and confidence scores
        """
        detected_agent = self._detect_intent(message)
        
        suggestions = {
            "primary_suggestion": detected_agent,
            "confidence": "high" if detected_agent == "educator" else "medium",
            "reasoning": self._get_reasoning(message, detected_agent),
            "available_agents": ["register", "educator"]
        }
        
        return suggestions
    
    def _get_reasoning(self, message: str, agent_type: AgentType) -> str:
        """Get reasoning for agent selection"""
        if agent_type == "educator":
            return "Message contains keywords related to camp selection criteria, safety, or guidance needs"
        else:
            return "Message appears to be general conversation or initial inquiry" 