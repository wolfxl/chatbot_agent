"""
Tests for LLM Manager - Centralized LLM instance management
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from agents.llm_manager import (
    LLMManager, 
    get_llm_manager, 
    initialize_llm_manager,
    get_llm,
    get_available_providers,
    is_provider_available,
    get_default_provider,
    get_provider_status,
    reset_llm_manager
)


@pytest.fixture(autouse=True)
def reset_manager():
    """Reset the LLM manager before each test"""
    reset_llm_manager()
    yield
    reset_llm_manager()


@pytest.mark.usefixtures("reset_manager")
@patch.dict(os.environ, {}, clear=True)
class TestLLMManager:
    """Test cases for LLMManager class"""
    
    def test_init(self):
        """Test LLMManager initialization"""
        manager = LLMManager()
        assert manager._llm_instances == {}
        assert manager._initialized is False
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_initialize_success(self, mock_gemini, mock_openai):
        """Test successful initialization with both providers"""
        manager = LLMManager()
        
        # Mock the LLM instances
        mock_openai_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_gemini.return_value = mock_gemini_instance
        
        manager.initialize()
        
        assert manager._initialized is True
        assert "openai" in manager._llm_instances
        assert "gemini" in manager._llm_instances
        assert len(manager._llm_instances) == 2
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_no_keys(self):
        """Test initialization with no API keys"""
        manager = LLMManager()
        
        with pytest.raises(ValueError, match="No LLM providers could be initialized"):
            manager.initialize()
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_initialize_openai_only(self, mock_openai):
        """Test initialization with only OpenAI"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        manager.initialize()
        
        assert manager._initialized is True
        assert "openai" in manager._llm_instances
        assert "gemini" not in manager._llm_instances
        assert len(manager._llm_instances) == 1
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-google-key'}, clear=True)
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_initialize_gemini_only(self, mock_gemini):
        """Test initialization with only Gemini"""
        manager = LLMManager()
        
        mock_gemini_instance = MagicMock()
        mock_gemini.return_value = mock_gemini_instance
        
        manager.initialize()
        
        assert manager._initialized is True
        assert "gemini" in manager._llm_instances
        assert "openai" not in manager._llm_instances
        assert len(manager._llm_instances) == 1
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_get_llm_success(self, mock_gemini, mock_openai):
        """Test getting LLM instance successfully"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_gemini.return_value = mock_gemini_instance
        
        manager.initialize()
        
        # Test getting OpenAI
        openai_llm = manager.get_llm("openai")
        assert openai_llm == mock_openai_instance
        
        # Test getting Gemini
        gemini_llm = manager.get_llm("gemini")
        assert gemini_llm == mock_gemini_instance
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_llm_not_initialized(self):
        """Test getting LLM when not initialized"""
        manager = LLMManager()
        
        with pytest.raises(ValueError, match="No LLM providers could be initialized"):
            manager.get_llm("openai")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_get_llm_provider_not_available(self, mock_openai):
        """Test getting LLM for unavailable provider"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        manager.initialize()
        
        with pytest.raises(ValueError, match="Provider 'gemini' not available"):
            manager.get_llm("gemini")
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_get_available_providers(self, mock_gemini, mock_openai):
        """Test getting available providers"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_gemini.return_value = mock_gemini_instance
        
        manager.initialize()
        
        providers = manager.get_available_providers()
        assert "openai" in providers
        assert "gemini" in providers
        assert len(providers) == 2
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_is_provider_available(self, mock_openai):
        """Test checking if provider is available"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        manager.initialize()
        
        assert manager.is_provider_available("openai") is True
        assert manager.is_provider_available("gemini") is False
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_get_default_provider_openai_preferred(self, mock_gemini, mock_openai):
        """Test getting default provider with OpenAI available"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_gemini.return_value = mock_gemini_instance
        
        manager.initialize()
        
        default = manager.get_default_provider()
        assert default == "openai"
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-google-key'}, clear=True)
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_get_default_provider_gemini_fallback(self, mock_gemini):
        """Test getting default provider with only Gemini available"""
        manager = LLMManager()
        
        mock_gemini_instance = MagicMock()
        mock_gemini.return_value = mock_gemini_instance
        
        manager.initialize()
        
        default = manager.get_default_provider()
        assert default == "gemini"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_default_provider_none_available(self):
        """Test getting default provider with no providers available"""
        manager = LLMManager()
        with pytest.raises(ValueError, match="No LLM providers could be initialized"):
            manager.get_default_provider()
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_test_provider_success(self, mock_gemini, mock_openai):
        """Test provider testing with successful call"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_openai_instance.invoke.return_value = "Hello response"
        mock_openai.return_value = mock_openai_instance
        
        manager.initialize()
        
        result = manager.test_provider("openai")
        assert result is True
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_test_provider_failure(self, mock_openai):
        """Test provider testing with failed call"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_openai_instance.invoke.side_effect = Exception("API Error")
        mock_openai.return_value = mock_openai_instance
        
        manager.initialize()
        
        result = manager.test_provider("openai")
        assert result is False
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key',
        'OPENAI_MODEL': 'gpt-4',
        'GEMINI_MODEL': 'gemini-pro-1.5'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_get_provider_status(self, mock_gemini, mock_openai):
        """Test getting provider status"""
        manager = LLMManager()
        
        mock_openai_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_openai_instance.invoke.return_value = "Hello response"
        mock_gemini_instance.invoke.return_value = "Hello response"
        mock_openai.return_value = mock_openai_instance
        mock_gemini.return_value = mock_gemini_instance
        
        manager.initialize()
        
        status = manager.get_provider_status()
        
        assert "openai" in status
        assert "gemini" in status
        assert status["openai"]["available"] is True
        assert status["openai"]["working"] is True
        assert status["openai"]["model"] == "gpt-4"
        assert status["gemini"]["available"] is True
        assert status["gemini"]["working"] is True
        assert status["gemini"]["model"] == "gemini-pro-1.5"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_provider_status_no_providers(self):
        """Test getting provider status with no providers available"""
        manager = LLMManager()
        
        with pytest.raises(ValueError):
            manager.get_provider_status()
    
    def test_reset(self):
        """Test resetting the manager"""
        manager = LLMManager()
        manager._llm_instances = {"test": "instance"}
        manager._initialized = True
        
        manager.reset()
        
        assert manager._llm_instances == {}
        assert manager._initialized is False


class TestGlobalFunctions:
    """Test cases for global functions"""
    
    def test_get_llm_manager_singleton(self):
        """Test that get_llm_manager returns singleton instance"""
        manager1 = get_llm_manager()
        manager2 = get_llm_manager()
        assert manager1 is manager2
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_initialize_llm_manager(self, mock_openai):
        """Test global initialize_llm_manager function"""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        initialize_llm_manager()
        
        manager = get_llm_manager()
        assert manager._initialized is True
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_get_llm_global(self, mock_openai):
        """Test global get_llm function"""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        llm = get_llm("openai")
        assert llm == mock_openai_instance
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_get_available_providers_global(self, mock_gemini, mock_openai):
        """Test global get_available_providers function"""
        mock_openai_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_gemini.return_value = mock_gemini_instance
        
        providers = get_available_providers()
        assert "openai" in providers
        assert "gemini" in providers
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_is_provider_available_global(self, mock_openai):
        """Test global is_provider_available function"""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        assert is_provider_available("openai") is True
        assert is_provider_available("gemini") is False
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    @patch('agents.llm_manager.ChatOpenAI')
    def test_get_default_provider_global(self, mock_openai):
        """Test global get_default_provider function"""
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        default = get_default_provider()
        assert default == "openai"
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key'
    })
    @patch('agents.llm_manager.ChatOpenAI')
    @patch('agents.llm_manager.ChatGoogleGenerativeAI')
    def test_get_provider_status_global(self, mock_gemini, mock_openai):
        """Test global get_provider_status function"""
        mock_openai_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_openai_instance.invoke.return_value = "Hello response"
        mock_gemini_instance.invoke.return_value = "Hello response"
        mock_openai.return_value = mock_openai_instance
        mock_gemini.return_value = mock_gemini_instance
        
        status = get_provider_status()
        
        assert "openai" in status
        assert "gemini" in status
        assert status["openai"]["available"] is True
        assert status["gemini"]["available"] is True
    
    def test_reset_llm_manager(self):
        """Test resetting the global LLM manager"""
        # Get initial manager
        manager1 = get_llm_manager()
        
        # Reset
        reset_llm_manager()
        
        # Get new manager
        manager2 = get_llm_manager()
        
        # Should be different instances
        assert manager1 is not manager2 