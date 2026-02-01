'use client';

import { useEffect, useRef } from 'react';
import { ChatMessage } from '@/types/api';
import { User, Bot, Loader2 } from 'lucide-react';
import clsx from 'clsx';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading?: boolean;
  className?: string;
}

export function MessageList({ messages, isLoading = false, className }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const renderMessageContent = (content: string) => {
    // Simple markdown-like rendering for basic formatting
    const lines = content.split('\n');
    return lines.map((line, index) => {
      if (line.startsWith('```')) {
        return null; // Handle code blocks separately if needed
      }
      
      // Handle bullet points
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        return (
          <li key={index} className="ml-4">
            {line.trim().substring(2)}
          </li>
        );
      }
      
      // Handle bold text **text**
      const boldPattern = /\*\*(.*?)\*\*/g;
      const parts = line.split(boldPattern);
      
      return (
        <p key={index} className={index === 0 ? '' : 'mt-2'}>
          {parts.map((part, partIndex) => {
            if (partIndex % 2 === 1) {
              return <strong key={partIndex}>{part}</strong>;
            }
            return part;
          })}
        </p>
      );
    });
  };

  return (
    <div
      ref={containerRef}
      className={clsx('flex flex-col h-full overflow-y-auto p-4 space-y-4', className)}
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
    >
      {messages.length === 0 && !isLoading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <Bot className="h-12 w-12 mx-auto mb-4 text-secondary-400" />
            <h3 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-2">
              Start a conversation
            </h3>
            <p className="text-secondary-600 dark:text-secondary-400">
              Ask me anything! I can help you with questions, analysis, and more.
            </p>
          </div>
        </div>
      ) : (
        <>
          {messages.map((message, index) => (
            <div
              key={message.id || index}
              className={clsx(
                'flex gap-3 max-w-4xl',
                message.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto'
              )}
            >
              {/* Avatar */}
              <div
                className={clsx(
                  'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
                  message.role === 'user'
                    ? 'bg-primary-600 text-white'
                    : 'bg-secondary-200 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-300'
                )}
              >
                {message.role === 'user' ? (
                  <User className="h-4 w-4" />
                ) : (
                  <Bot className="h-4 w-4" />
                )}
              </div>

              {/* Message content */}
              <div
                className={clsx(
                  'flex-1 min-w-0',
                  message.role === 'user' ? 'text-right' : 'text-left'
                )}
              >
                <div
                  className={clsx(
                    'inline-block px-4 py-3 rounded-lg max-w-full',
                    message.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-secondary-100 dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 border border-secondary-200 dark:border-secondary-700'
                  )}
                >
                  <div className="prose prose-sm max-w-none">
                    {renderMessageContent(message.content)}
                  </div>

                  {/* Sources/metadata */}
                  {message.metadata?.sources && message.metadata.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-secondary-200 dark:border-secondary-600">
                      <p className="text-xs font-medium mb-2 opacity-70">Sources:</p>
                      <div className="space-y-1">
                        {message.metadata.sources.map((source: any, sourceIndex: number) => (
                          <div
                            key={sourceIndex}
                            className="text-xs p-2 bg-secondary-50 dark:bg-secondary-700/50 rounded border"
                          >
                            <div className="font-medium">{source.filename}</div>
                            {source.snippet && (
                              <div className="mt-1 opacity-70 line-clamp-2">
                                {source.snippet}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Timestamp */}
                <div
                  className={clsx(
                    'text-xs text-secondary-500 dark:text-secondary-400 mt-1',
                    message.role === 'user' ? 'text-right' : 'text-left'
                  )}
                >
                  {message.timestamp && formatTime(message.timestamp)}
                </div>
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {isLoading && (
            <div className="flex gap-3 max-w-4xl mr-auto">
              <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-secondary-200 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-300">
                <Bot className="h-4 w-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="inline-block px-4 py-3 rounded-lg bg-secondary-100 dark:bg-secondary-800 border border-secondary-200 dark:border-secondary-700">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm text-secondary-600 dark:text-secondary-400">
                      Thinking...
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
}