"""
Tests for Format Agent - Response formatting functionality
"""
import pytest
from agents.format_agent import FormatAgent, get_format_agent
from agents.state import ConversationState


class TestFormatAgent:
    """Test cases for FormatAgent class"""
    
    def test_init(self):
        """Test FormatAgent initialization"""
        agent = FormatAgent("markdown")
        assert agent.style == "markdown"
        
        agent = FormatAgent("html")
        assert agent.style == "html"
        
        agent = FormatAgent("plain")
        assert agent.style == "plain"
    
    def test_format_markdown(self):
        """Test markdown formatting"""
        agent = FormatAgent("markdown")
        
        # Test numbered list formatting
        text = "1. First item\n2. Second item"
        formatted = agent._format_markdown(text)
        assert "**1. First item**" in formatted
        assert "**2. Second item**" in formatted
        
        # Test section header formatting
        text = "Safety: Important considerations\nCost: Budget planning"
        formatted = agent._format_markdown(text)
        assert "**Safety:**" in formatted
        assert "**Cost:**" in formatted
        
        # Test bullet point formatting
        text = "Features:\n- Feature 1\n- Feature 2"
        formatted = agent._format_markdown(text)
        assert "• Feature 1" in formatted
        assert "• Feature 2" in formatted
    
    def test_format_html(self):
        """Test HTML formatting"""
        agent = FormatAgent("html")
        
        # Test numbered list formatting
        text = "1. First item\n2. Second item"
        formatted = agent._format_html(text)
        assert "<b>1. First item</b>" in formatted
        assert "<b>2. Second item</b>" in formatted
        
        # Test section header formatting
        text = "Safety: Important considerations"
        formatted = agent._format_html(text)
        assert "<b>Safety:</b>" in formatted
        
        # Test newline conversion
        text = "Line 1\nLine 2"
        formatted = agent._format_html(text)
        assert "<br>" in formatted
    
    def test_format_plain(self):
        """Test plain text formatting"""
        agent = FormatAgent("plain")
        
        text = "  Some text with extra spaces  \n\n"
        formatted = agent._format_plain(text)
        assert formatted == "Some text with extra spaces"
    
    def test_format_response_markdown(self):
        """Test format_response with markdown style"""
        agent = FormatAgent("markdown")
        
        # Create test state
        state = ConversationState(
            user_message="Test message",
            agent_response="1. First point\nSafety: Important\n- Feature 1"
        )
        
        # Format the response
        result_state = agent.format_response(state)
        
        # Check that formatted response is in context
        assert "formatted_response" in result_state.context
        formatted = result_state.context["formatted_response"]
        
        # Check formatting was applied
        assert "**1. First point**" in formatted
        assert "**Safety:**" in formatted
        assert "• Feature 1" in formatted
    
    def test_format_response_html(self):
        """Test format_response with HTML style"""
        agent = FormatAgent("html")
        
        # Create test state
        state = ConversationState(
            user_message="Test message",
            agent_response="1. First point\nSafety: Important"
        )
        
        # Format the response
        result_state = agent.format_response(state)
        
        # Check that formatted response is in context
        assert "formatted_response" in result_state.context
        formatted = result_state.context["formatted_response"]
        
        # Check formatting was applied
        assert "<b>1. First point</b>" in formatted
        assert "<b>Safety:</b>" in formatted
        assert "<br>" in formatted
    
    def test_format_response_plain(self):
        """Test format_response with plain style"""
        agent = FormatAgent("plain")
        
        # Create test state
        state = ConversationState(
            user_message="Test message",
            agent_response="  Some text with spaces  \n\n"
        )
        
        # Format the response
        result_state = agent.format_response(state)
        
        # Check that formatted response is in context
        assert "formatted_response" in result_state.context
        formatted = result_state.context["formatted_response"]
        
        # Check formatting was applied
        assert formatted == "Some text with spaces"
    
    def test_format_response_empty(self):
        """Test format_response with empty response"""
        agent = FormatAgent("markdown")
        
        # Create test state with empty response
        state = ConversationState(
            user_message="Test message",
            agent_response=""
        )
        
        # Format the response
        result_state = agent.format_response(state)
        
        # Check that formatted response is in context
        assert "formatted_response" in result_state.context
        assert result_state.context["formatted_response"] == ""
    
    def test_format_response_error_handling(self):
        """Test format_response error handling"""
        agent = FormatAgent("markdown")
        
        # Create test state with None response
        state = ConversationState(
            user_message="Test message",
            agent_response=None
        )
        
        # Format the response
        result_state = agent.format_response(state)
        
        # Check that formatted response is in context and falls back to original
        assert "formatted_response" in result_state.context
        assert result_state.context["formatted_response"] is None


class TestGlobalFunctions:
    """Test cases for global functions"""
    
    def test_get_format_agent_markdown(self):
        """Test get_format_agent with markdown style"""
        agent = get_format_agent("markdown")
        assert isinstance(agent, FormatAgent)
        assert agent.style == "markdown"
    
    def test_get_format_agent_html(self):
        """Test get_format_agent with html style"""
        agent = get_format_agent("html")
        assert isinstance(agent, FormatAgent)
        assert agent.style == "html"
    
    def test_get_format_agent_plain(self):
        """Test get_format_agent with plain style"""
        agent = get_format_agent("plain")
        assert isinstance(agent, FormatAgent)
        assert agent.style == "plain"
    
    def test_get_format_agent_custom(self):
        """Test get_format_agent with custom style"""
        agent = get_format_agent("custom_style")
        assert isinstance(agent, FormatAgent)
        assert agent.style == "custom_style"
    
    def test_get_format_agent_singleton(self):
        """Test that get_format_agent returns singleton instances"""
        agent1 = get_format_agent("markdown")
        agent2 = get_format_agent("markdown")
        assert agent1 is agent2
        
        agent3 = get_format_agent("html")
        agent4 = get_format_agent("html")
        assert agent3 is agent4
        
        # Different styles should be different instances
        assert agent1 is not agent3


class TestIntegration:
    """Integration tests for Format Agent"""
    
    def test_format_agent_with_real_content(self):
        """Test format agent with realistic content"""
        agent = FormatAgent("markdown")
        
        # Simulate a realistic educator response
        realistic_response = """
        Here are the key factors to consider when choosing a summer camp:

        1. Safety and Accreditation
        - Check if the camp is accredited by the American Camp Association
        - Verify staff background checks and training

        2. Age-Appropriate Activities
        - Ensure activities match your child's developmental stage
        - Look for age-specific programs

        3. Cost and Value
        - Compare what's included in the price
        - Ask about additional fees
        """
        
        state = ConversationState(
            user_message="What should I consider when choosing a camp?",
            agent_response=realistic_response
        )
        
        result_state = agent.format_response(state)
        formatted = result_state.context["formatted_response"]
        
        # Check that formatting was applied
        assert "**1. Safety and Accreditation**" in formatted  # Numbered lists
        assert "**2. Age-Appropriate Activities**" in formatted
        assert "**3. Cost and Value**" in formatted
        assert "• Check if the camp" in formatted  # Bullet points
    
    def test_format_agent_with_html_content(self):
        """Test format agent with HTML formatting"""
        agent = FormatAgent("html")
        
        response = """
        Important considerations:
        1. Safety first
        2. Cost analysis
        - Point A
        - Point B
        """
        
        state = ConversationState(
            user_message="What should I consider?",
            agent_response=response
        )
        
        result_state = agent.format_response(state)
        formatted = result_state.context["formatted_response"]
        
        # Check that HTML formatting was applied
        assert "<b>1. Safety first</b>" in formatted
        assert "<b>2. Cost analysis</b>" in formatted
        assert "<br>" in formatted 