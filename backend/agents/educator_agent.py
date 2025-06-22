"""
Educator Agent - Specializes in providing guidance on summer camp selection criteria
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


class EducatorAgent:
    """Educator Agent for providing guidance on summer camp selection criteria"""
    
    def __init__(self, llm_provider: LLMProvider = "openai", format_style: str = "html"):
        """
        Initialize the Educator Agent with specified LLM provider
        
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
            logger.info(f"Educator Agent using {self.llm_provider} LLM from centralized manager")
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
        
        # Add the educator_agent node with our implementation
        workflow.add_node("educator_agent", self._educator_agent_node)
        
        # Add the format_agent node for post-processing
        workflow.add_node("format_agent", self._format_agent_node)
        
        # Set entry point
        workflow.set_entry_point("educator_agent")
        
        # Connect educator_agent to format_agent
        workflow.add_edge("educator_agent", "format_agent")
        
        # Set finish point
        workflow.set_finish_point("format_agent")
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        
        logger.info(f"LangGraph workflow compiled successfully with {self.llm_provider} and {self.format_style} formatting")
        return compiled_workflow
    
    def _educator_agent_node(self, state: ConversationState) -> ConversationState:
        """
        Educator Agent node implementation
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        try:
            # Create system message with educator expertise
            system_message = SystemMessage(content="""
            You are an expert Educator Agent specializing in summer camp guidance and selection criteria.
            
            Your expertise includes:
            1. **Age-Appropriate Considerations**: Understanding developmental stages and appropriate activities for different age groups
            2. **Safety and Accreditation**: Knowledge of safety standards, certifications, and what to look for in camp safety
            3. **Program Types**: Different types of camps (sports, arts, STEM, traditional, specialty, etc.)
            4. **Location and Logistics**: Transportation, distance, duration, and practical considerations
            5. **Cost and Value**: Budget considerations, what's included, and value assessment
            6. **Staff Qualifications**: Understanding staff training, ratios, and qualifications
            7. **Special Needs Accommodations**: Support for children with different abilities or requirements
            8. **Health and Medical**: Medical staff, medication policies, and health consider
            9. **Communication**: How camps communicate with parents and handle emergencies
            10. **Reviews and References**: How to evaluate camp reputation and get references
            
            Your role is to:
            - Provide comprehensive, well-structured advice on camp selection criteria
            - Ask clarifying questions to better understand the family's specific needs
            - Offer practical tips and checklists for parents
            - Address concerns about safety, cost, and child development
            - Provide balanced, informed perspectives on different camp options
            - Help parents understand what questions to ask when evaluating camps
            
            Be thorough, educational, and supportive. Use clear language and provide actionable advice.
            Structure your responses with clear sections when appropriate.
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
                "agent_type": "educator"
            })
            
            logger.info(f"Educator Agent ({self.llm_provider}) processed message for session: {state.session_id}")
            
        except Exception as e:
            logger.error(f"Error in Educator Agent ({self.llm_provider}): {e}")
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
        Process a user message through the Educator Agent
        
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
                "agent_type": "educator",
                "format_style": self.format_style,
                "error": getattr(final_state_obj, 'error', None)
            }
            
        except Exception as e:
            logger.error(f"Error processing message with {self.llm_provider}: {e}")
            return {
                "response": f"I apologize, but I encountered an error with {self.llm_provider}. Please try again.",
                "session_id": session_id or str(uuid.uuid4()),
                "llm_provider": self.llm_provider,
                "agent_type": "educator",
                "format_style": self.format_style,
                "error": str(e)
            }


# Global Educator Agent instances
educator_agent_gemini = EducatorAgent("gemini")
educator_agent_openai = EducatorAgent("openai")


def get_educator_agent(provider: LLMProvider = "openai") -> EducatorAgent:
    """
    Get Educator Agent instance for specified provider
    
    Args:
        provider: LLM provider ("gemini" or "openai")
        
    Returns:
        EducatorAgent instance
    """
    if provider == "gemini":
        return educator_agent_gemini
    elif provider == "openai":
        return educator_agent_openai
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_educator_agent(provider: LLMProvider = "openai") -> EducatorAgent:
    """
    Create a new Educator Agent instance for specified provider
    
    Args:
        provider: LLM provider ("gemini" or "openai")
        
    Returns:
        New EducatorAgent instance
    """
    return EducatorAgent(provider) 