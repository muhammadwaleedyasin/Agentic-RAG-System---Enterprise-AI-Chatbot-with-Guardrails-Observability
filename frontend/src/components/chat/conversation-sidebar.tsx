'use client';

import { useState } from 'react';
import { Conversation } from '@/types/api';
import { Button } from '@/components/ui/button';
import { Modal } from '@/components/ui/modal';
import { Plus, MessageSquare, Trash2, Search, X } from 'lucide-react';
import clsx from 'clsx';

interface ConversationSidebarProps {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  onSelectConversation: (conversationId: string) => void;
  onCreateConversation: () => void;
  onDeleteConversation?: (conversationId: string) => void;
  onCloseSidebar?: () => void;
}

export function ConversationSidebar({
  conversations,
  currentConversation,
  onSelectConversation,
  onCreateConversation,
  onDeleteConversation,
  onCloseSidebar,
}: ConversationSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [conversationToDelete, setConversationToDelete] = useState<string | null>(null);

  const filteredConversations = conversations.filter(conversation =>
    conversation.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleDeleteClick = (conversationId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    setConversationToDelete(conversationId);
    setDeleteModalOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (conversationToDelete && onDeleteConversation) {
      await onDeleteConversation(conversationToDelete);
      setDeleteModalOpen(false);
      setConversationToDelete(null);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return `${days} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <>
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-secondary-200 dark:border-secondary-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
              Conversations
            </h2>
            {onCloseSidebar && (
              <button
                onClick={onCloseSidebar}
                className="md:hidden p-1 rounded-md hover:bg-secondary-100 dark:hover:bg-secondary-800 focus:outline-none focus:ring-2 focus:ring-primary-500"
                aria-label="Close sidebar"
              >
                <X className="h-5 w-5" />
              </button>
            )}
          </div>

          {/* New conversation button */}
          <Button
            onClick={onCreateConversation}
            className="w-full mb-4"
            data-testid="conversation-new"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Conversation
          </Button>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 placeholder:text-secondary-500 dark:placeholder:text-secondary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Conversations list */}
        <div className="flex-1 overflow-y-auto">
          {filteredConversations.length === 0 ? (
            <div className="p-4 text-center">
              <MessageSquare className="h-8 w-8 mx-auto mb-2 text-secondary-400" />
              <p className="text-sm text-secondary-600 dark:text-secondary-400">
                {searchQuery ? 'No conversations found' : 'No conversations yet'}
              </p>
            </div>
          ) : (
            <div className="p-2 space-y-1">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  onClick={() => onSelectConversation(conversation.id)}
                  className={clsx(
                    'group relative p-3 rounded-md cursor-pointer transition-colors',
                    'hover:bg-secondary-100 dark:hover:bg-secondary-800',
                    currentConversation?.id === conversation.id
                      ? 'bg-primary-100 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800'
                      : 'border border-transparent'
                  )}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelectConversation(conversation.id);
                    }
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="text-sm font-medium text-secondary-900 dark:text-secondary-100 truncate">
                        {conversation.title}
                      </h3>
                      <div className="flex items-center gap-2 mt-1 text-xs text-secondary-500 dark:text-secondary-400">
                        <span>{conversation.message_count} messages</span>
                        <span>•</span>
                        <span>{formatDate(conversation.updated_at)}</span>
                      </div>
                    </div>

                    {/* Delete button */}
                    {onDeleteConversation && (
                      <button
                        onClick={(e) => handleDeleteClick(conversation.id, e)}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-error-100 dark:hover:bg-error-900/20 text-error-600 dark:text-error-400 transition-opacity focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-error-500"
                        aria-label={`Delete conversation: ${conversation.title}`}
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Delete confirmation modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Delete Conversation"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-secondary-700 dark:text-secondary-300">
            Are you sure you want to delete this conversation? This action cannot be undone.
          </p>
          
          <div className="flex gap-3 justify-end">
            <Button
              variant="outline"
              onClick={() => setDeleteModalOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDelete}
            >
              Delete
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
}