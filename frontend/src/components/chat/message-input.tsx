'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip } from 'lucide-react';
import { Button } from '@/components/ui/button';
import clsx from 'clsx';

interface MessageInputProps {
  onSendMessage: (content: string) => void;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
}

export function MessageInput({
  onSendMessage,
  disabled = false,
  placeholder = 'Type your message...',
  maxLength = 4000,
}: MessageInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedMessage = message.trim();
    if (!trimmedMessage || disabled) return;

    onSendMessage(trimmedMessage);
    setMessage('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const characterCount = message.length;
  const isNearLimit = characterCount > maxLength * 0.8;
  const isOverLimit = characterCount > maxLength;

  return (
    <div className="p-4 bg-white dark:bg-secondary-900 border-t border-secondary-200 dark:border-secondary-700">
      <form onSubmit={handleSubmit} className="flex flex-col gap-2">
        <div className="flex gap-2 items-end">
          {/* Attachment button (placeholder for future file upload) */}
          <button
            type="button"
            disabled={disabled}
            className="p-2 rounded-md text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300 hover:bg-secondary-100 dark:hover:bg-secondary-800 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label="Attach file"
            title="Attach file (coming soon)"
          >
            <Paperclip className="h-5 w-5" />
          </button>

          {/* Message input */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled}
              maxLength={maxLength}
              className={clsx(
                'w-full resize-none rounded-md border border-secondary-300 dark:border-secondary-600 bg-white dark:bg-secondary-800 px-3 py-2 text-sm text-secondary-900 dark:text-secondary-100 placeholder:text-secondary-500 dark:placeholder:text-secondary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:cursor-not-allowed disabled:opacity-50 transition-colors',
                'min-h-[44px] max-h-[120px]',
                isOverLimit && 'border-error-500 focus:ring-error-500'
              )}
              rows={1}
              data-testid="chat-input"
              aria-label="Message input"
              aria-describedby="character-count"
            />
          </div>

          {/* Send button */}
          <Button
            type="submit"
            disabled={disabled || !message.trim() || isOverLimit}
            size="icon"
            className="h-11 w-11 flex-shrink-0"
            data-testid="chat-send"
            aria-label="Send message"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>

        {/* Character count and hints */}
        <div className="flex items-center justify-between text-xs text-secondary-500 dark:text-secondary-400">
          <div className="flex items-center gap-4">
            <span>Press Enter to send, Shift+Enter for new line</span>
          </div>
          
          {(isNearLimit || isOverLimit) && (
            <span
              id="character-count"
              className={clsx(
                'font-medium',
                isOverLimit ? 'text-error-600 dark:text-error-400' : 'text-warning-600 dark:text-warning-400'
              )}
            >
              {characterCount}/{maxLength}
            </span>
          )}
        </div>
      </form>
    </div>
  );
}