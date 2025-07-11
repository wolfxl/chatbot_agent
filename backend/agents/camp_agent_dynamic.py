from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable
from backend.agents.state import CampChatState
from backend.database.supabase_client import supabase_client
from backend.models.schemas import CampSearchFilters
from backend.config import settings
from typing import Dict, Any
import logging
import json
from backend.agents.tools.email_draft_agent import EmailDraftAgent
from backend.agents.extract_info import extract_sender_info_with_llm
import re

logger = logging.getLogger(__name__)

@traceable(name="classify_intent")
async def classify_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:

    try:

        if isinstance(state, dict):
            chat_state = CampChatState(**state)
        else:
            chat_state = state
            

        last_message = chat_state.messages[-1].content if chat_state.messages else ""
        has_cached_results = bool(chat_state.last_search_results)
        
        logger.info(f"🎯 CLASSIFY - Message: '{last_message}' | Cached results: {len(chat_state.last_search_results) if has_cached_results else 0}")
        

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0
        )
        

        system_prompt = f"""You are an intent classifier for a summer camp search system.

The user currently has {len(chat_state.last_search_results) if has_cached_results else 0} cached search results from previous queries.

Analyze the user's message and classify into one of these intents:

1. **"search"** - Get NEW camps from database (requires database query)
   - Finding camps with new/different criteria
   - Asking for camp count without existing results
   - Completely new search requests

2. **"filter"** - Filter/analyze EXISTING cached results (no database needed)
   - Questions about already-found camps
   - Filtering cached results by criteria
   - Asking about specific camps from previous search
   - Only use if user has cached results ({has_cached_results})

3. **"send_availability_email"** - User wants to contact a camp or request information via email
   - Requests to email a camp, contact for more info, or send an inquiry
   - Phrases like "Can you email this camp for me?", "Contact the camp about availability", etc.

4. **"general"** - General conversation (no camp data needed)
   - Greetings, how-to questions, general chat

Return JSON format:
{{
    "intent": "search|filter|send_availability_email|general",
    "search_criteria": {{
        "activity": "activity type from user request",
        "location": "city, state, or general area",
        "age": "age mentioned",
        "budget": "price range if mentioned"
    }}
}}

Examples:
- "how many camps do you have" -> {{"intent": "search", "search_criteria": {{}}}}
- "find soccer camps" -> {{"intent": "search", "search_criteria": {{"activity": "soccer"}}}}
- "are there camps related to minecraft" (with cached results) -> {{"intent": "filter", "search_criteria": {{"activity": "minecraft"}}}}
- "show me the first 3" (with cached results) -> {{"intent": "filter", "search_criteria": {{}}}}
- "what about art camps instead" -> {{"intent": "search", "search_criteria": {{"activity": "art"}}}}
- "Can you email this camp for me?" -> {{"intent": "send_availability_email", "search_criteria": {{}}}}
- "hello" -> {{"intent": "general"}}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User message: {last_message}")
        ]
        

        response = llm.invoke(messages)
        
        try:
            # Clean the response content - remove markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            intent_data = json.loads(content)
        except Exception as json_error:
            logger.error(f"❌ JSON parsing failed: {json_error}")
            intent_data = {"intent": "general", "search_criteria": {}}
        

        extracted_intent = intent_data.get("intent", "general")
        

        if extracted_intent == "filter" and not has_cached_results:
            extracted_intent = "search"
            logger.info(f"🔄 CLASSIFY - Converted filter→search (no cached results)")
        
        chat_state.current_intent = extracted_intent
        logger.info(f"✅ CLASSIFY - Intent: {extracted_intent}")
        

        if extracted_intent in ["search", "filter"]:
            criteria = intent_data.get("search_criteria", {})
            chat_state.search_filters = {}
            

            if criteria.get("activity"):
                chat_state.search_filters["category"] = criteria.get("activity")
            

            location = criteria.get("location", "")
            if location:
                if "," in location:
                    parts = location.split(",")
                    chat_state.search_filters["city"] = parts[0].strip()
                    if len(parts) > 1:
                        chat_state.search_filters["state"] = parts[1].strip()
                else:
                    chat_state.search_filters["location"] = location.strip()
            

            if criteria.get("age"):
                try:
                    age = int(criteria.get("age"))
                    grade = max(0, age - 5)
                    chat_state.search_filters["min_grade"] = grade
                    chat_state.search_filters["max_grade"] = grade + 2
                except:
                    pass
                    
            logger.info(f"🔍 CLASSIFY - Filters: {chat_state.search_filters}")
        
        return chat_state.dict()
        
    except Exception as e:
        logger.error(f"❌ CLASSIFY - Error: {e}")
        chat_state = CampChatState(**state)
        chat_state.current_intent = "general"
        return chat_state.dict()

@traceable(name="search_camps")
async def search_camps_node(state: Dict[str, Any]) -> Dict[str, Any]:

    logger.info(f"🔍 SEARCH - Database query starting")
    
    try:

        chat_state = CampChatState(**state)
        

        filters = CampSearchFilters()
        if chat_state.search_filters:
            for key, value in chat_state.search_filters.items():
                if value and hasattr(filters, key):
                    setattr(filters, key, value)
        

        camps = await supabase_client.search_camps(filters)
        

        chat_state.last_search_results = camps
        logger.info(f"✅ SEARCH - Found {len(camps)} camps from database")
        
        return chat_state.dict()
        
    except Exception as e:
        logger.error(f"❌ SEARCH - Database error: {e}")
        chat_state = CampChatState(**state)
        chat_state.last_search_results = []
        return chat_state.dict()

@traceable(name="filter_cached")
async def filter_cached_results_node(state: Dict[str, Any]) -> Dict[str, Any]:

    logger.info(f"🔎 FILTER - Processing cached results")
    
    try:

        chat_state = CampChatState(**state)
        
        cached_camps = chat_state.last_search_results or []
        logger.info(f"🔎 FILTER - Starting with {len(cached_camps)} cached camps")
        

        if not chat_state.search_filters:
            logger.info(f"✅ FILTER - No filters, returning all {len(cached_camps)} cached camps")
            return chat_state.dict()
        

        filtered_camps = []
        search_filters = chat_state.search_filters
        
        for camp in cached_camps:
            match = True
            

            if search_filters.get("category"):
                category_filter = search_filters["category"].lower()
                

                camp_name = (camp.get("camp_name") or "").lower()
                camp_description = (camp.get("description") or "").lower()
                

                if category_filter not in camp_name and category_filter not in camp_description:

                    camp_categories = camp.get("camp_categories") or []
                    category_match = False
                    for cat_rel in camp_categories:
                        if cat_rel and isinstance(cat_rel, dict):
                            categories = cat_rel.get("categories") or {}
                            cat_name = (categories.get("name") or "").lower()
                            if category_filter in cat_name:
                                category_match = True
                                break
                    if not category_match:
                        match = False
            

            if search_filters.get("city") or search_filters.get("state") or search_filters.get("location"):
                location_filter = (
                    (search_filters.get("city") or "") + " " + 
                    (search_filters.get("state") or "") + " " + 
                    (search_filters.get("location") or "")
                ).strip().lower()
                
                camp_sessions = camp.get("camp_sessions") or []
                location_match = False
                for session in camp_sessions:
                    if session and isinstance(session, dict):
                        location = session.get("locations") or {}
                        session_location = (
                            (location.get("city") or "") + " " + 
                            (location.get("state") or "")
                        ).strip().lower()
                        if location_filter in session_location:
                            location_match = True
                            break
                if not location_match:
                    match = False
            

            if search_filters.get("min_grade") is not None or search_filters.get("max_grade") is not None:
                camp_min = camp.get("min_grade")
                camp_max = camp.get("max_grade")
                filter_min = search_filters.get("min_grade")
                filter_max = search_filters.get("max_grade")
                
                if camp_min is not None and camp_max is not None:
    
                    if filter_min is not None and camp_max < filter_min:
                        match = False
                    if filter_max is not None and camp_min > filter_max:
                        match = False
            
            if match:
                filtered_camps.append(camp)
        

        chat_state.last_search_results = filtered_camps
        logger.info(f"✅ FILTER - Filtered to {len(filtered_camps)} camps (no database query)")
        
        return chat_state.dict()
        
    except Exception as e:
        logger.error(f"❌ FILTER - Error: {e}")

        chat_state = CampChatState(**state)
        chat_state.last_search_results = []
        logger.info(f"🔄 FILTER - Error fallback: returning empty results")
        return chat_state.dict()

@traceable(name="generate_response")
async def generate_response_node(state: Dict[str, Any]) -> Dict[str, Any]:

    # Prevent overwriting final_response if already set (e.g., by send_availability_email_node)
    if state.get("final_response"):
        return state

    try:

        chat_state = CampChatState(**state)
        results_count = len(chat_state.last_search_results) if chat_state.last_search_results else 0
        
        logger.info(f"🗨️ RESPONSE - Intent: {chat_state.current_intent} | Results: {results_count}")
        

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.7
        )
        

        last_message = chat_state.messages[-1].content if chat_state.messages else ""
        

        if chat_state.current_intent in ["search", "filter"]:
            if chat_state.last_search_results:
                camps_info = []
                for camp in chat_state.last_search_results[:5]:
                    camp_text = f"**{camp.get('camp_name', 'Unknown Camp')}**"
                    

                    org = camp.get('organizations')
                    if org:
                        camp_text += f" by {org.get('name', 'Unknown Org')}"
                    camp_text += "\n"
                    

                    if camp.get('price'):
                        camp_text += f"Price: ${camp['price']}/week\n"
                    

                    if camp.get('min_grade') is not None and camp.get('max_grade') is not None:
                        camp_text += f"Grades: {camp['min_grade']}-{camp['max_grade']}\n"
                    

                    sessions = camp.get('camp_sessions', [])
                    if sessions:
                        session = sessions[0]
                        location = session.get('locations')
                        if location:
                            city = location.get('city', '')
                            state = location.get('state', '')
                            if city or state:
                                camp_text += f"Location: {city}, {state}\n"
                    

                    if camp.get('description'):
                        desc = camp['description'][:150] + "..." if len(camp['description']) > 150 else camp['description']
                        camp_text += f"Description: {desc}\n"
                    
                    camps_info.append(camp_text)
                

                camps_text = "\n\n".join(camps_info)
                
                if chat_state.current_intent == "filter":
                    action_text = "filtered your previous search results"
                else:
                    action_text = "searched our database"
                

                if "how many" in last_message.lower() or "count" in last_message.lower():
                    prompt = f"The user asked: '{last_message}'\n\nI {action_text} and found {len(chat_state.last_search_results)} camps matching the criteria.\n\nGenerate a helpful response about the total count and offer to show some examples or help them search for specific types."
                else:
                    prompt = f"The user asked: '{last_message}'\n\nI {action_text} and found {len(chat_state.last_search_results)} camps. Here are the top results:\n\n{camps_text}\n\nGenerate an enthusiastic, helpful response about these camps. Be conversational and highlight key details."
                
            else:

                filters_applied = chat_state.search_filters or {}
                filter_text = ", ".join([f"{k}: {v}" for k, v in filters_applied.items() if v])
                
                action_text = "filtered your previous results" if chat_state.current_intent == "filter" else "searched our database"
                prompt = f"The user asked: '{last_message}'\nFilters applied: {filter_text}\n\nI {action_text} but couldn't find any camps matching these criteria. Generate a helpful response suggesting they try different search terms or ask about available camp types. Be encouraging and helpful."
        
        else:
            # General response
            prompt = f"The user said: '{last_message}'\n\nYou are a helpful summer camp assistant. Respond naturally and helpfully. You can mention that users can search for camps by activity, location, or age group."
        

        response = llm.invoke([HumanMessage(content=prompt)])
        ai_response = response.content
        

        chat_state.final_response = ai_response
        
        return chat_state.dict()
        
    except Exception as e:
        logger.error(f"❌ RESPONSE - Error: {e}")
        chat_state = CampChatState(**state)
        chat_state.final_response = "I'm here to help you find great summer camps! What are you looking for?"
        return chat_state.dict()

def send_availability_email_node(state: dict) -> dict:
    # Use Gemini LLM to extract sender info from chat history
    sender_info = extract_sender_info_with_llm(state.get("messages", []))
    # Extract all camps from last search results
    last_search_results = state.get("last_search_results", [])
    topic = state.get("topic", "camp availability")
    prompt_template = state.get("prompt_template")

    drafted_emails = []
    # For debugging: loop exactly twice
    for i in range(2):
        if i < len(last_search_results):
            camp = last_search_results[i]
            receiver_name = camp.get("camp_name", "Camp Team")
            receiver_email = camp.get("contact_email", "test@gmail.com")
        else:
            receiver_name = f"Camp Team {i+1}"
            receiver_email = f"test{i+1}@gmail.com"
        agent = EmailDraftAgent(
            sender_name=sender_info["sender_name"],
            sender_age_group=sender_info["sender_age_group"],
            sender_interests=sender_info["sender_interests"],
            receiver_name=receiver_name,
            receiver_email=receiver_email,
            topic=topic,
            prompt_template=prompt_template
        )
        email_text = agent.draft_email()
        drafted_emails.append({
            "receiver_name": receiver_name,
            "receiver_email": receiver_email,
            "email_text": email_text
        })
        # Print each drafted email for backend visibility
        print("\n--- Email Drafted ---\n")
        print(f"To: {receiver_name} <{receiver_email}>")
        print(f"Subject: {topic}")
        print(email_text)
        print("\n--- End Email ---\n")

    state["drafted_emails"] = drafted_emails
    state["email_status"] = "sent"
    # Set a user-facing response summarizing the action
    if drafted_emails:
        state["final_response"] = (
            "An enquiry email about availability has been sent to the camp(s). Would you like to start a new search?"
        )
    else:
        state["final_response"] = "No camps found to send emails to."
    return state

class CampAgent:
    def __init__(self):
        self.graph = self._create_graph()

        self.session_store: Dict[str, CampChatState] = {}
    
    def _get_or_create_session_state(self, message: str, session_id: str) -> CampChatState:
        if session_id in self.session_store:

            existing_state = self.session_store[session_id]
            existing_state.messages.append(HumanMessage(content=message))
            cached_count = len(existing_state.last_search_results) if existing_state.last_search_results else 0
            logger.info(f"📚 SESSION - Loaded session {session_id[-8:]} | Messages: {len(existing_state.messages)} | Cached: {cached_count}")
            return existing_state
        else:

            new_state = CampChatState(
                messages=[HumanMessage(content=message)],
                session_id=session_id
            )
            self.session_store[session_id] = new_state
            logger.info(f"🆕 SESSION - Created new session {session_id[-8:]}")
            return new_state
    
    def _save_session_state(self, session_id: str, state: CampChatState) -> None:
        self.session_store[session_id] = state
        cached_count = len(state.last_search_results) if state.last_search_results else 0
        logger.info(f"💾 SESSION - Saved session {session_id[-8:]} | Messages: {len(state.messages)} | Cached: {cached_count}")
    
    def _create_graph(self) -> StateGraph:
        

        workflow = StateGraph(dict)
        

        workflow.add_node("classify", classify_intent_node)
        workflow.add_node("search", search_camps_node)
        workflow.add_node("filter", filter_cached_results_node)
        workflow.add_node("respond", generate_response_node)
        workflow.add_node("send_availability_email", send_availability_email_node)
        

        def route_after_classify(state: Dict[str, Any]) -> str:
            current_intent = state.get("current_intent", "general")
            logger.info(f"🚏 ROUTER - Intent: {current_intent}")
            
            if current_intent == "search":
                logger.info(f"🚏 ROUTER - Route: search (database query)")
                return "search"
            elif current_intent == "filter":
                logger.info(f"🚏 ROUTER - Route: filter (cached results)")
                return "filter"
            elif current_intent == "send_availability_email":
                logger.info(f"🚏 ROUTER - Route: send_availability_email (email draft)")
                return "send_availability_email"
            else:
                logger.info(f"🚏 ROUTER - Route: respond (general)")
                return "respond"
        

        workflow.set_entry_point("classify")
        workflow.add_conditional_edges(
            "classify",
            route_after_classify,
            {"search": "search", "filter": "filter", "respond": "respond", "send_availability_email": "send_availability_email"}
        )
        workflow.add_edge("search", "respond")
        workflow.add_edge("filter", "respond")
        workflow.add_edge("respond", END)
        workflow.add_edge("send_availability_email", "respond")
        

        return workflow.compile()
    
    async def process_message(self, message: str, session_id: str) -> Dict[str, Any]:
        
        try:

            initial_state = self._get_or_create_session_state(message, session_id)
            

            initial_state.current_intent = None
            initial_state.final_response = None
            
            logger.info(f"🚀 WORKFLOW - Processing message {len(initial_state.messages)} for session {session_id[-8:]}")
            

            final_state = await self.graph.ainvoke(initial_state.dict())
            

            if isinstance(final_state, dict):
                final_state = CampChatState(**final_state)
            

            self._save_session_state(session_id, final_state)
            

            response_text = final_state.final_response or "I'm here to help you find great summer camps! What are you looking for?"
            
            return {
                "response": response_text,
                "session_id": session_id,
                "context": {
                    "intent": final_state.current_intent,
                    "search_count": len(final_state.last_search_results) if final_state.last_search_results else 0,
                    "filters_used": final_state.search_filters,
                    "conversation_length": len(final_state.messages),
                    "has_cached_results": bool(final_state.last_search_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I'm here to help you find great summer camps! What are you looking for?",
                "session_id": session_id,
                "context": {"error": str(e)}
            }


camp_agent = CampAgent()
