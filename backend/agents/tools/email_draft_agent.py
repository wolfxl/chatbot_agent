from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import settings
from typing import List

class EmailDraftAgent:
    def __init__(
        self,
        sender_name: str,
        sender_age_group: str,
        sender_interests: List[str],
        receiver_name: str,
        receiver_email: str,
        topic: str,
        prompt_template: str = None
    ):
        self.sender_name = sender_name
        self.sender_age_group = sender_age_group
        self.sender_interests = sender_interests
        self.receiver_name = receiver_name
        self.receiver_email = receiver_email
        self.topic = topic
        # Explicitly use the provided prompt_template or a default that includes all fields
        self.prompt_template = prompt_template or (
            "Write a short, concise, and business-like email from {sender_name} "
            "(age group: {sender_age_group}, interests: {sender_interests}) to {receiver_name}. "
            "The topic of the email is: '{topic}'. "
            "Make it professional and to the point. "
            "Explicitly mention the sender's age group in the email body."
        )

    def draft_email(self) -> str:
        # Explicitly format the prompt with sender_age_group, sender_interests, and template
        prompt = self.prompt_template.format(
            sender_name=self.sender_name,
            sender_age_group=self.sender_age_group,
            sender_interests=', '.join(self.sender_interests),
            receiver_name=self.receiver_name,
            receiver_email=self.receiver_email,
            topic=self.topic
        )
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.7
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        email_text = getattr(response, 'content', str(response))
        return email_text
