"""
State management for LangGraph conversation system
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from langgraph.graph import StateGraph


@dataclass
class ConversationState:
    """State for managing conversation flow"""
    
    # User input
    user_message: str = ""
    
    # Agent responses
    agent_response: str = ""
    
    # Conversation history
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Session information
    session_id: str = ""
    
    # Agent context
    current_agent: str = "register_agent"
    agent_type: str = "register"  # register, educator, etc.
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    error: Optional[str] = None


def create_state_graph() -> StateGraph:
    """
    Create the state graph for the conversation system
    
    Returns:
        StateGraph: The conversation state graph
    """
    # Create the state graph
    workflow = StateGraph(ConversationState)
    
    # Note: Agent nodes will be added by individual agent classes
    # This function just creates the basic workflow structure
    
    return workflow


def register_agent_node(state: ConversationState) -> ConversationState:
    """
    Register Agent node - handles initial user communication
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated conversation state
    """
    # This will be implemented in the main agent file
    # For now, just return the state as-is
    return state


def educator_agent_node(state: ConversationState) -> ConversationState:
    """
    Educator Agent node - handles educational guidance
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated conversation state
    """
    # This will be implemented in the educator agent file
    # For now, just return the state as-is
    return state 