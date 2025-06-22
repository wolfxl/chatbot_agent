"""
LLM Manager - Centralized LLM instance management for all agents
"""
import os
from typing import Dict, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

# Type for LLM provider selection
LLMProvider = Literal["gemini", "openai"]


class LLMManager:
    """Centralized manager for LLM instances"""
    
    def __init__(self):
        """Initialize the LLM manager"""
        self._llm_instances: Dict[str, ChatOpenAI | ChatGoogleGenerativeAI] = {}
        self._initialized = False
        
    def initialize(self):
        """Initialize all available LLM instances"""
        if self._initialized:
            logger.info("LLM Manager already initialized")
            return
        
        logger.info("Initializing LLM Manager...")
        
        # Initialize OpenAI
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                self._llm_instances["openai"] = ChatOpenAI(
                    model=openai_model,
                    openai_api_key=openai_api_key,
                    temperature=0,
                    max_tokens=1500
                )
                logger.info(f"✅ OpenAI LLM initialized with model: {openai_model}")
            else:
                logger.warning("❌ OPENAI_API_KEY not found, skipping OpenAI initialization")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI LLM: {e}")
        
        # Initialize Gemini
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                gemini_model = os.getenv("GEMINI_MODEL", "gemini-pro")
                self._llm_instances["gemini"] = ChatGoogleGenerativeAI(
                    model=gemini_model,
                    google_api_key=google_api_key,
                    temperature=0,
                    max_output_tokens=1500
                )
                logger.info(f"✅ Gemini LLM initialized with model: {gemini_model}")
            else:
                logger.warning("❌ GOOGLE_API_KEY not found, skipping Gemini initialization")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini LLM: {e}")
        
        if not self._llm_instances:
            raise ValueError("No LLM providers could be initialized. Please check your API keys.")
        
        self._initialized = True
        logger.info(f"🎉 LLM Manager initialized with {len(self._llm_instances)} providers: {list(self._llm_instances.keys())}")
    
    def get_llm(self, provider: LLMProvider) -> ChatOpenAI | ChatGoogleGenerativeAI:
        """
        Get LLM instance for specified provider
        
        Args:
            provider: LLM provider ("openai" or "gemini")
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not available
        """
        if not self._initialized:
            self.initialize()
        
        if provider not in self._llm_instances:
            available = list(self._llm_instances.keys())
            raise ValueError(f"Provider '{provider}' not available. Available providers: {available}")
        
        return self._llm_instances[provider]
    
    def get_available_providers(self) -> list[str]:
        """
        Get list of available LLM providers
        
        Returns:
            List of available provider names
        """
        if not self._initialized:
            self.initialize()
        
        return list(self._llm_instances.keys())
    
    def is_provider_available(self, provider: LLMProvider) -> bool:
        """
        Check if a provider is available
        
        Args:
            provider: LLM provider to check
            
        Returns:
            True if provider is available, False otherwise
        """
        if not self._initialized:
            self.initialize()
        
        return provider in self._llm_instances
    
    def get_default_provider(self) -> str:
        """
        Get the default provider (prefer OpenAI, fallback to Gemini)
        
        Returns:
            Default provider name
            
        Raises:
            ValueError: If no providers are available
        """
        if not self._initialized:
            self.initialize()
        
        if "openai" in self._llm_instances:
            return "openai"
        elif "gemini" in self._llm_instances:
            return "gemini"
        else:
            raise ValueError("No LLM providers available")
    
    def test_provider(self, provider: LLMProvider) -> bool:
        """
        Test if a provider is working with a simple call
        
        Args:
            provider: LLM provider to test
            
        Returns:
            True if provider is working, False otherwise
        """
        try:
            llm = self.get_llm(provider)
            # Make a simple test call
            response = llm.invoke("Hello")
            return True
        except Exception as e:
            logger.error(f"Provider {provider} test failed: {e}")
            return False
    
    def get_provider_status(self) -> Dict[str, Dict[str, any]]:
        """
        Get status of all providers
        
        Returns:
            Dictionary with provider status information
        """
        if not self._initialized:
            self.initialize()
        
        status = {}
        
        for provider in ["openai", "gemini"]:
            if provider in self._llm_instances:
                status[provider] = {
                    "available": True,
                    "working": self.test_provider(provider),
                    "model": self._get_model_name(provider)
                }
            else:
                status[provider] = {
                    "available": False,
                    "working": False,
                    "error": "Not initialized"
                }
        
        return status
    
    def _get_model_name(self, provider: LLMProvider) -> str:
        """Get the model name for a provider"""
        if provider == "openai":
            return os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        elif provider == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-pro")
        else:
            return "unknown"
    
    def reset(self):
        """Reset the manager for testing purposes"""
        self._llm_instances = {}
        self._initialized = False


# Global LLM Manager instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """
    Get the global LLM Manager instance
    
    Returns:
        LLMManager instance
    """
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def reset_llm_manager():
    """Reset the global LLM manager for testing purposes"""
    global _llm_manager
    if _llm_manager is not None:
        _llm_manager.reset()
    _llm_manager = None


def initialize_llm_manager():
    """Initialize the global LLM Manager"""
    manager = get_llm_manager()
    manager.initialize()


def get_llm(provider: LLMProvider) -> ChatOpenAI | ChatGoogleGenerativeAI:
    """
    Get LLM instance for specified provider
    
    Args:
        provider: LLM provider ("openai" or "gemini")
        
    Returns:
        LLM instance
    """
    manager = get_llm_manager()
    return manager.get_llm(provider)


def get_available_providers() -> list[str]:
    """
    Get list of available LLM providers
    
    Returns:
        List of available provider names
    """
    manager = get_llm_manager()
    return manager.get_available_providers()


def is_provider_available(provider: LLMProvider) -> bool:
    """
    Check if a provider is available
    
    Args:
        provider: LLM provider to check
        
    Returns:
        True if provider is available, False otherwise
    """
    manager = get_llm_manager()
    return manager.is_provider_available(provider)


def get_default_provider() -> str:
    """
    Get the default provider (prefer OpenAI, fallback to Gemini)
    
    Returns:
        Default provider name
    """
    manager = get_llm_manager()
    return manager.get_default_provider()


def get_provider_status() -> Dict[str, Dict[str, any]]:
    """
    Get status of all providers
    
    Returns:
        Dictionary with provider status information
    """
    manager = get_llm_manager()
    return manager.get_provider_status() 