"""
Format Agent - Formats the agent response before displaying it on the screen
"""
from agents.state import ConversationState
import logging
import re

logger = logging.getLogger(__name__)

class FormatAgent:
    """Agent for formatting responses for display"""
    def __init__(self, style: str = "markdown"):
        """
        Initialize the Format Agent
        Args:
            style: Formatting style ("markdown", "html", or "plain")
        """
        self.style = style

    def _format_markdown(self, text: str) -> str:
        # Bold entire numbered section header: '1. Safety and Accreditation' -> '**1. Safety and Accreditation**'
        text = re.sub(r"^\s*([0-9]+\.[^\n]+)", lambda m: f"**{m.group(1).strip()}**", text, flags=re.MULTILINE)
        # Bold section headers with or without colons: 'Safety:' or 'Safety and Accreditation' -> '**Safety:**' or '**Safety and Accreditation**'
        text = re.sub(r"^\s*([A-Z][^:\n]+)(:)?", r"**\1\2**", text, flags=re.MULTILINE)
        # Replace dash with bullet (handle leading spaces)
        text = re.sub(r"\n\s*- ", r"\n• ", text)
        return text

    def _format_html(self, text: str) -> str:
        """
        Enhanced HTML formatting with comprehensive support for various HTML elements
        """
        # Convert numbered lists to proper HTML (remove the number from <li> content)
        text = re.sub(r"^\s*[0-9]+\.\s+(.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
        
        # Convert bullet points to proper HTML
        text = re.sub(r"^\s*[-•]\s+(.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
        
        # Wrap consecutive list items in <ol> or <ul> tags
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        list_type = None
        list_items = []
        
        for line in lines:
            if line.strip().startswith('<li>'):
                if not in_list:
                    # Determine list type based on first item
                    if re.match(r'<li>.*', line.strip()):
                        # If the original line was a numbered list, use <ol>
                        if re.match(r'<li>[^<]*</li>', line.strip()):
                            # Check if the original line started with a number
                            orig_line = line.strip()[4:-5]
                            if re.match(r'^[0-9]+\.', orig_line):
                                list_type = 'ol'
                            else:
                                list_type = 'ul'
                        else:
                            list_type = 'ul'
                    else:
                        list_type = 'ul'
                    in_list = True
                list_items.append(line)
            else:
                if in_list:
                    # Close the list
                    if list_items:
                        formatted_lines.append(f"<{list_type}>")
                        formatted_lines.extend(list_items)
                        formatted_lines.append(f"</{list_type}>")
                    list_items = []
                    in_list = False
                    list_type = None
                
                # Process non-list content
                if line.strip():
                    # Bold section headers (numbered or text)
                    if re.match(r'^\s*([0-9]+\.[^\n]+|[A-Z][^:\n]+)(:)?', line.strip()):
                        line = re.sub(r'^\s*([0-9]+\.[^\n]+|[A-Z][^:\n]+)(:)?', r'<h3>\1\2</h3>', line.strip())
                    # Bold important terms
                    elif ':' in line and not line.strip().startswith('<'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            line = f"<strong>{parts[0].strip()}:</strong>{parts[1].strip()}"
                        else:
                            line = f"<p>{line.strip()}</p>"
                    else:
                        line = f"<p>{line.strip()}</p>"
                
                formatted_lines.append(line)
        
        # Handle any remaining list items
        if in_list and list_items:
            formatted_lines.append(f"<{list_type}>")
            formatted_lines.extend(list_items)
            formatted_lines.append(f"</{list_type}>")
        
        # Join lines and clean up
        result = '\n'.join(formatted_lines)
        
        # Remove empty paragraphs
        result = re.sub(r'<p>\s*</p>', '', result)
        
        # Add emphasis to important words
        result = re.sub(r'\b(important|key|essential|critical|vital)\b', r'<em>\1</em>', result, flags=re.IGNORECASE)
        
        # Add strong emphasis to very important terms
        result = re.sub(r'\b(must|should|always|never|required|recommended)\b', r'<strong>\1</strong>', result, flags=re.IGNORECASE)
        
        # Clean up multiple newlines
        result = re.sub(r'\n\s*\n', '\n', result)
        
        return result.strip()

    def _format_plain(self, text: str) -> str:
        return text.strip()

    def format_response(self, state: ConversationState) -> ConversationState:
        """
        Format the agent response in the conversation state
        Args:
            state: ConversationState with agent_response
        Returns:
            Updated ConversationState with formatted_response in context
        """
        try:
            response = state.agent_response
            if response is None:
                state.context["formatted_response"] = None
                return state
                
            if self.style == "markdown":
                formatted = self._format_markdown(response)
            elif self.style == "html":
                formatted = self._format_html(response)
            else:
                formatted = self._format_plain(response)
            state.context["formatted_response"] = formatted
            logger.info(f"Formatted response with style '{self.style}'")
        except Exception as e:
            logger.error(f"FormatAgent error: {e}")
            state.context["formatted_response"] = state.agent_response
        return state

# Singleton instance for default usage
format_agent_markdown = FormatAgent("markdown")
format_agent_html = FormatAgent("html")
format_agent_plain = FormatAgent("plain")

def get_format_agent(style: str = "markdown") -> FormatAgent:
    if style == "markdown":
        return format_agent_markdown
    elif style == "html":
        return format_agent_html
    elif style == "plain":
        return format_agent_plain
    else:
        return FormatAgent(style) 