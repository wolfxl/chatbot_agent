'use client'

import { useState, useEffect, useRef } from 'react'

export default function ChatInterface() {
  const [messages, setMessages] = useState([
    { type: 'bot', text: 'Hi! I can help you find the perfect summer camp for your child. What are you looking for?' }
  ])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [selectedProvider, setSelectedProvider] = useState('openai')
  const [selectedAgent, setSelectedAgent] = useState('auto') // 'auto', 'register', 'educator'

  // Refs for scrolling and input focus
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-focus input when component mounts and after messages are added
  useEffect(() => {
    inputRef.current?.focus()
  }, [messages])

  const sendMessage = async () => {
    if (!inputText.trim()) return

    const userMessage = { type: 'user', text: inputText }
    setMessages(prev => [...prev, userMessage])
    
    const currentInput = inputText
    setInputText('')
    setIsLoading(true)

    try {
      const requestBody = {
        message: currentInput,
        session_id: sessionId,
        provider: selectedProvider,
        agent: selectedAgent === 'auto' ? null : selectedAgent
      }

      console.log('🚀 Sending request:', { 
        message: currentInput, 
        session_id: sessionId ? sessionId.slice(-8) : 'new',
        provider: selectedProvider,
        agent: selectedAgent
      })

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      })

      if (response.ok) {
        const data = await response.json()
        
        console.log('📨 Response received:', data)

        if (data.session_id && data.session_id !== sessionId) {
          setSessionId(data.session_id)
          console.log('💾 Session ID stored:', data.session_id.slice(-8))
        }
        
        const botMessage = { 
          type: 'bot', 
          text: data.response || 'Sorry, I encountered an error.',
          formattedText: data.formatted_response || null,
          agentType: data.agent_type,
          routedTo: data.routed_to
        }
        setMessages(prev => [...prev, botMessage])
      } else {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }
    } catch (error) {
      console.error('❌ Error:', error)
      const errorMessage = { type: 'bot', text: `Sorry, I'm having trouble connecting to the server: ${error.message}` }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isLoading) {
      sendMessage()
    }
  }

  const handleProviderChange = (provider) => {
    setSelectedProvider(provider)
    console.log('🔄 Provider changed to:', provider)
  }

  const handleAgentChange = (agent) => {
    setSelectedAgent(agent)
    console.log('🔄 Agent changed to:', agent)
  }

  const getAgentDisplayName = (agentType) => {
    switch (agentType) {
      case 'register':
        return 'Receptionist'
      case 'educator':
        return 'Professor'
      default:
        return 'Assistant'
    }
  }

  const renderMessageContent = (message) => {
    // If we have formatted text (HTML), render it
    if (message.formattedText) {
      return (
        <div 
          className="formatted-content"
          dangerouslySetInnerHTML={{ __html: message.formattedText }}
        />
      )
    }
    
    // Otherwise, render plain text
    return <div className="plain-content">{message.text}</div>
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Summer Camp Assistant</h1>
        <div className="provider-selector">
          <button 
            className={`provider-btn ${selectedProvider === 'openai' ? 'active' : ''}`}
            onClick={() => handleProviderChange('openai')}
            disabled={isLoading}
          >
            OpenAI
          </button>
          <button 
            className={`provider-btn ${selectedProvider === 'gemini' ? 'active' : ''}`}
            onClick={() => handleProviderChange('gemini')}
            disabled={isLoading}
          >
            Gemini
          </button>
        </div>
        <div className="agent-selector">
          <button 
            className={`agent-btn ${selectedAgent === 'auto' ? 'active' : ''}`}
            onClick={() => handleAgentChange('auto')}
            disabled={isLoading}
          >
            Auto Route
          </button>
          <button 
            className={`agent-btn ${selectedAgent === 'register' ? 'active' : ''}`}
            onClick={() => handleAgentChange('register')}
            disabled={isLoading}
          >
            Receptionist
          </button>
          <button 
            className={`agent-btn ${selectedAgent === 'educator' ? 'active' : ''}`}
            onClick={() => handleAgentChange('educator')}
            disabled={isLoading}
          >
            Professor
          </button>
        </div>
        {sessionId && (
          <div className="session-info">
            Session: {sessionId.slice(-8)} | Provider: {selectedProvider} | Agent: {selectedAgent === 'auto' ? 'Auto' : getAgentDisplayName(selectedAgent)}
          </div>
        )}
      </div>
      
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`}>
            <div className="message-content">
              {renderMessageContent(message)}
              {message.agentType && (
                <div className="agent-indicator">
                  {getAgentDisplayName(message.agentType)}
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message bot">
            <div className="message-content">
              Thinking...
            </div>
          </div>
        )}
        {/* Invisible element for scrolling to bottom */}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <input
          ref={inputRef}
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask me about summer camps..."
          disabled={isLoading}
        />
        <button 
          onClick={sendMessage}
          disabled={isLoading || !inputText.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}
