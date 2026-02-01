'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { useChat } from '@/hooks/useChat';
import { ConversationSidebar } from '@/components/chat/conversation-sidebar';
import { MessageList } from '@/components/chat/message-list';
import { MessageInput } from '@/components/chat/message-input';
import { Card } from '@/components/ui/card';
import { AuthGuard } from '@/components/auth/auth-guard';
import { Loader2 } from 'lucide-react';

function ChatPageContent() {
  const { isAuthenticated } = useAuth();
  const {
    conversations,
    currentConversation,
    messages,
    isLoading,
    isConnected,
    createConversation,
    selectConversation,
    sendMessage,
  } = useChat();

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    if (isAuthenticated && conversations.length === 0) {
      // Create initial conversation if none exists
      createConversation();
    }
  }, [isAuthenticated, conversations.length, createConversation]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary-600" />
          <h2 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-2">
            Loading Chat...
          </h2>
          <p className="text-secondary-600 dark:text-secondary-400">
            Setting up your conversation
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      {/* Mobile sidebar overlay */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed md:relative z-50 md:z-0
        w-80 h-full bg-white dark:bg-secondary-900 
        border-r border-secondary-200 dark:border-secondary-700
        transform transition-transform duration-200 ease-in-out
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        <ConversationSidebar
          conversations={conversations}
          currentConversation={currentConversation}
          onSelectConversation={selectConversation}
          onCreateConversation={createConversation}
          onCloseSidebar={() => setIsSidebarOpen(false)}
        />
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat header */}
        <div className="bg-white dark:bg-secondary-900 border-b border-secondary-200 dark:border-secondary-700 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="md:hidden p-2 rounded-md hover:bg-secondary-100 dark:hover:bg-secondary-800 focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-label="Open conversation sidebar"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            
            <div>
              <h1 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                {currentConversation?.title || 'New Conversation'}
              </h1>
              <div className="flex items-center gap-2 text-sm text-secondary-600 dark:text-secondary-400">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success-500' : 'bg-error-500'}`} />
                {isConnected ? 'Connected' : 'Disconnected'}
              </div>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-hidden">
          {currentConversation ? (
            <MessageList
              messages={messages}
              isLoading={isLoading}
              className="h-full"
            />
          ) : (
            <div className="h-full flex items-center justify-center">
              <Card className="p-8 text-center max-w-md">
                <h3 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                  Welcome to Agentic RAG
                </h3>
                <p className="text-secondary-600 dark:text-secondary-400 mb-4">
                  Start a new conversation to begin chatting with the AI assistant.
                </p>
                <button
                  onClick={createConversation}
                  className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
                  data-testid="conversation-new"
                >
                  Start New Conversation
                </button>
              </Card>
            </div>
          )}
        </div>

        {/* Message input */}
        {currentConversation && (
          <div className="border-t border-secondary-200 dark:border-secondary-700">
            <MessageInput
              onSendMessage={sendMessage}
              disabled={!isConnected}
              placeholder={isConnected ? "Type your message..." : "Connecting..."}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <AuthGuard>
      <ChatPageContent />
    </AuthGuard>
  );
}