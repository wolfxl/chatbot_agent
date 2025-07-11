from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import settings
import json

def extract_sender_info_with_llm(messages):
    """
    Use Gemini LLM to extract sender info (name, age group, interests) from chat history.
    messages: list of message objects (dicts or objects) with a content field
    Returns a dict with sender_name, sender_age_group, sender_interests.
    """
    user_history = "\n".join(
        f"User: {msg['content']}" if isinstance(msg, dict) and 'content' in msg else f"User: {getattr(msg, 'content', '')}"
        for msg in messages
        if (isinstance(msg, dict) and msg.get('content')) or (hasattr(msg, 'content') and getattr(msg, 'content'))
    )

    prompt = f"""
You are an information extraction assistant. Given the following chat history, extract the user's:
- name (if provided)
- age or age group (if provided, or infer from age)
- interests (as a list, if provided, or infer from context)

Return a JSON object with keys: sender_name, sender_age_group, sender_interests.

Chat history:
{user_history}

Examples:
- "i'm sam, find camps for 9 yrs old boy" -> {{"sender_name": "Sam", "sender_age_group": "9-11", "sender_interests": ["summer camps"]}}
- "Sam here, looking for camps for my 9 year old" -> {{"sender_name": "Sam", "sender_age_group": "9-11", "sender_interests": ["summer camps"]}}
- "My son Sam is 9, can you help?" -> {{"sender_name": "Sam", "sender_age_group": "9-11", "sender_interests": ["summer camps"]}}
- "I'm Alice, 13, love soccer and art" -> {{"sender_name": "Alice", "sender_age_group": "13-15", "sender_interests": ["soccer", "art"]}}
- "Looking for camps for my 10 year old who likes science." -> {{"sender_name": "campbuddy.com", "sender_age_group": "10-12", "sender_interests": ["science"]}}
- "My name is Bob." -> {{"sender_name": "Bob", "sender_age_group": "10-12", "sender_interests": ["summer camps"]}}
- "I'm 8 and like basketball." -> {{"sender_name": "campbuddy.com", "sender_age_group": "8-10", "sender_interests": ["basketball"]}}

Always extract the name and age if they are present, even if the message is short or informal.
If the user's age is given as a number, convert it to an age group of the form "X-Y" where X is the age and Y is X+2.
If any information is missing, use:
- sender_name: "campbuddy.com"
- sender_age_group: "10-12"
- sender_interests: ["summer camps"]
"""

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.4
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        info = json.loads(content)
    except Exception:
        info = {
            "sender_name": "campbuddy.com",
            "sender_age_group": "10-12",
            "sender_interests": ["summer camps"]
        }
    return info 