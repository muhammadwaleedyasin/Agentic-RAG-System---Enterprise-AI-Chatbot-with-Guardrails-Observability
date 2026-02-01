'use client';

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { apiClient } from '@/lib/api';
import { Conversation, Message, ChatMessage } from '@/types/api';

interface UseChatReturn {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  messages: ChatMessage[];
  isLoading: boolean;
  isConnected: boolean;
  error: string | null;
  createConversation: () => Promise<void>;
  selectConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  refreshConversations: () => Promise<void>;
}

export function useChat(): UseChatReturn {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { socket, isConnected, send } = useWebSocket();

  // Set up WebSocket message handlers
  useEffect(() => {
    if (!socket) return;

    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'message':
            // Add new message to current conversation
            if (data.conversation_id === currentConversation?.id) {
              setMessages(prev => [...prev, {
                id: data.message_id,
                role: data.role || 'assistant',
                content: data.content,
                timestamp: new Date().toISOString(),
                metadata: data.metadata,
              }]);
            }
            break;
            
          case 'token':
            // Update streaming message
            if (data.conversation_id === currentConversation?.id) {
              setMessages(prev => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                
                if (lastIndex >= 0 && updated[lastIndex].role === 'assistant' && !updated[lastIndex].id) {
                  // Update streaming message
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    content: updated[lastIndex].content + data.data,
                  };
                } else {
                  // Start new streaming message
                  updated.push({
                    role: 'assistant',
                    content: data.data,
                    timestamp: new Date().toISOString(),
                  });
                }
                
                return updated;
              });
            }
            break;
            
          case 'message_complete':
            // Finalize streaming message
            if (data.conversation_id === currentConversation?.id) {
              setMessages(prev => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                
                if (lastIndex >= 0 && updated[lastIndex].role === 'assistant') {
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    id: data.message_id,
                    metadata: data.metadata,
                  };
                }
                
                return updated;
              });
            }
            break;
            
          case 'error':
            setError(data.message || 'An error occurred');
            break;
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    socket.addEventListener('message', handleMessage);
    return () => socket.removeEventListener('message', handleMessage);
  }, [socket, currentConversation?.id]);

  const refreshConversations = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await apiClient.getConversations({ sort_by: 'updated_at', sort_order: 'desc' });
      const conversationsArray = (data as any)?.data || data as Conversation[];
      setConversations(conversationsArray);
    } catch (err) {
      console.error('Failed to load conversations:', err);
      setError(err instanceof Error ? err.message : 'Failed to load conversations');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load conversations on mount
  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  const createConversation = useCallback(async () => {
    try {
      setError(null);
      const conversation = await apiClient.createConversation({
        title: 'New Conversation',
      }) as Conversation;
      
      setConversations(prev => [conversation, ...prev]);
      setCurrentConversation(conversation);
      setMessages([]);
    } catch (err) {
      console.error('Failed to create conversation:', err);
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
    }
  }, []);

  const selectConversation = useCallback(async (conversationId: string) => {
    try {
      setError(null);
      const conversation = conversations.find(c => c.id === conversationId);
      if (!conversation) return;

      setCurrentConversation(conversation);
      
      // Load messages for this conversation
      const messagesData = await apiClient.getMessages(conversationId);
      const messagesArray = (messagesData as any)?.data || messagesData as Message[];
      const chatMessages: ChatMessage[] = messagesArray.map((msg: Message) => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: msg.created_at,
        metadata: msg.metadata,
      }));
      
      setMessages(chatMessages);
    } catch (err) {
      console.error('Failed to load conversation:', err);
      setError(err instanceof Error ? err.message : 'Failed to load conversation');
    }
  }, [conversations]);

  const deleteConversation = useCallback(async (conversationId: string) => {
    try {
      setError(null);
      await apiClient.deleteConversation(conversationId);
      
      setConversations(prev => prev.filter(c => c.id !== conversationId));
      
      // If we deleted the current conversation, clear it
      if (currentConversation?.id === conversationId) {
        setCurrentConversation(null);
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete conversation');
    }
  }, [currentConversation?.id]);

  const sendMessage = useCallback(async (content: string) => {
    if (!currentConversation || !content.trim()) return;

    try {
      setError(null);
      
      // Add user message immediately
      const userMessage: ChatMessage = {
        role: 'user',
        content: content.trim(),
        timestamp: new Date().toISOString(),
      };
      
      setMessages(prev => [...prev, userMessage]);

      // Send via WebSocket for real-time response
      if (isConnected && socket) {
        send({
          type: 'chat_message',
          conversation_id: currentConversation.id,
          content: content.trim(),
        });
      } else {
        // Fallback to API if WebSocket not available
        await apiClient.sendMessage(currentConversation.id, content.trim());
        
        // Reload messages to get the response
        const messagesData = await apiClient.getMessages(currentConversation.id);
        const messagesArray = (messagesData as any)?.data || messagesData as Message[];
        const chatMessages: ChatMessage[] = messagesArray.map((msg: Message) => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: msg.created_at,
          metadata: msg.metadata,
        }));
        
        setMessages(chatMessages);
      }
    } catch (err) {
      console.error('Failed to send message:', err);
      setError(err instanceof Error ? err.message : 'Failed to send message');
    }
  }, [currentConversation, isConnected, socket, send]);

  return {
    conversations,
    currentConversation,
    messages,
    isLoading,
    isConnected,
    error,
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
    refreshConversations,
  };
}