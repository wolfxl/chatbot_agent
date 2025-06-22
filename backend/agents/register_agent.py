"""
Register Agent - A simple LangGraph agent using Gemini or OpenAI LLM
"""
import uuid
from typing import Dict, Any, Literal
from langchain.schema import HumanMessage, SystemMessage
from agents.state import ConversationState, create_state_graph
from agents.llm_manager import get_llm, LLMProvider
from agents.format_agent import get_format_agent
from langgraph.graph import StateGraph
import logging

logger = logging.getLogger(__name__)

# Type for LLM provider selection
LLMProvider = Literal["gemini", "openai"]


class RegisterAgent:
    """Register Agent for handling user communication with Gemini or OpenAI LLM"""
    
    def __init__(self, llm_provider: LLMProvider = "openai", format_style: str = "html"):
        """
        Initialize the Register Agent with specified LLM provider
        
        Args:
            llm_provider: Choice of LLM provider ("gemini" or "openai")
            format_style: Formatting style for responses ("markdown", "html", "plain")
        """
        self.llm_provider = llm_provider
        self.format_style = format_style
        self.llm = self._setup_llm()
        self.format_agent = get_format_agent(format_style)
        self.workflow = self._setup_workflow()
    
    def _setup_llm(self):
        """
        Setup LLM based on provider selection using centralized LLM manager
        
        Returns:
            Configured LLM instance (Gemini or OpenAI)
        """
        try:
            # Use centralized LLM manager
            llm = get_llm(self.llm_provider)
            logger.info(f"Register Agent using {self.llm_provider} LLM from centralized manager")
            return llm
                
        except Exception as e:
            logger.error(f"Failed to get {self.llm_provider} LLM from manager: {e}")
            raise
    
    def _setup_workflow(self) -> StateGraph:
        """
        Setup the LangGraph workflow
        
        Returns:
            StateGraph: Configured workflow
        """
        workflow = create_state_graph()
        
        # Add the register_agent node with our implementation
        workflow.add_node("register_agent", self._register_agent_node)
        
        # Add the format_agent node for post-processing
        workflow.add_node("format_agent", self._format_agent_node)
        
        # Set entry point
        workflow.set_entry_point("register_agent")
        
        # Connect register_agent to format_agent
        workflow.add_edge("register_agent", "format_agent")
        
        # Set finish point
        workflow.set_finish_point("format_agent")
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        
        logger.info(f"LangGraph workflow compiled successfully with {self.llm_provider} and {self.format_style} formatting")
        return compiled_workflow
    
    def _register_agent_node(self, state: ConversationState) -> ConversationState:
        """
        Register Agent node implementation
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        try:
            # Create system message
            system_message = SystemMessage(content="""
            You are a helpful Personal Assistant that assists users in starting conversations.
            Your role is to:
            1. Greet users warmly and introduce yourself
            2. Ask how you can help them today
            3. Provide a friendly, welcoming response
            4. Keep responses concise but helpful
            5. say "Goodbye" when the conversation is over
            
            Be conversational, friendly, and encouraging. Make users feel welcome!
            """)
            
            # Create user message
            user_message = HumanMessage(content=state.user_message)
            
            # Get response from LLM
            messages = [system_message, user_message]
            response = self.llm.invoke(messages)
            
            # Update state
            state.agent_response = response.content
            
            # Add to conversation history
            state.conversation_history.append({
                "user": state.user_message,
                "agent": state.agent_response,
                "timestamp": str(uuid.uuid4()),
                "llm_provider": self.llm_provider,
                "agent_type": "register"
            })
            
            logger.info(f"Register Agent (Personal Assistant) ({self.llm_provider}) processed message for session: {state.session_id}")
            
        except Exception as e:
            logger.error(f"Error in Register Agent ({self.llm_provider}): {e}")
            state.error = str(e)
            state.agent_response = f"I apologize, but I encountered an error with {self.llm_provider}. Please try again."
        
        return state
    
    def _format_agent_node(self, state: ConversationState) -> ConversationState:
        """
        Format Agent node implementation for post-processing
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state with formatted response
        """
        try:
            # Use the format agent to format the response
            state = self.format_agent.format_response(state)
            logger.info(f"Format Agent ({self.format_style}) processed response for session: {state.session_id}")
        except Exception as e:
            logger.error(f"Error in Format Agent ({self.format_style}): {e}")
            # Fallback to original response if formatting fails
            state.context["formatted_response"] = state.agent_response
        
        return state
    
    async def process_message(
        self, 
        message: str, 
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the Register Agent
        
        Args:
            message: User's message
            session_id: Optional session ID for conversation tracking
            
        Returns:
            Dict containing agent response and session info
        """
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Create initial state
            initial_state = ConversationState(
                user_message=message,
                session_id=session_id
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)

            # If it's a dict-like object, convert to ConversationState
            if hasattr(final_state, 'get') and 'agent_response' in final_state:
                # Try to convert to ConversationState
                try:
                    final_state_obj = ConversationState(**final_state)
                except Exception as e:
                    logger.error(f"Failed to convert to ConversationState: {e}")
                    final_state_obj = final_state
            else:
                final_state_obj = final_state

            # Get formatted response from context, fallback to original response
            formatted_response = getattr(final_state_obj, 'context', {}).get('formatted_response', getattr(final_state_obj, 'agent_response', None))

            return {
                "response": getattr(final_state_obj, 'agent_response', None),
                "formatted_response": formatted_response,
                "session_id": getattr(final_state_obj, 'session_id', None),
                "conversation_history": getattr(final_state_obj, 'conversation_history', None),
                "llm_provider": self.llm_provider,
                "agent_type": "register",
                "format_style": self.format_style,
                "error": getattr(final_state_obj, 'error', None)
            }
            
        except Exception as e:
            logger.error(f"Error processing message with {self.llm_provider}: {e}")
            return {
                "response": f"I apologize, but I encountered an error with {self.llm_provider}. Please try again.",
                "session_id": session_id or str(uuid.uuid4()),
                "llm_provider": self.llm_provider,
                "agent_type": "register",
                "format_style": self.format_style,
                "error": str(e)
            }


# Global Register Agent instances
register_agent_gemini = RegisterAgent("gemini")
register_agent_openai = RegisterAgent("openai")


def get_register_agent(provider: LLMProvider = "gemini") -> RegisterAgent:
    """
    Get Register Agent instance for specified provider
    
    Args:
        provider: LLM provider ("gemini" or "openai")
        
    Returns:
        RegisterAgent instance
    """
    if provider == "gemini":
        return register_agent_gemini
    elif provider == "openai":
        return register_agent_openai
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_register_agent(provider: LLMProvider = "gemini") -> RegisterAgent:
    """
    Create a new Register Agent instance for specified provider
    
    Args:
        provider: LLM provider ("gemini" or "openai")
        
    Returns:
        New RegisterAgent instance
    """
    return RegisterAgent(provider) 