'use client';

import { useState } from 'react';
import { Document, QueryParams } from '@/types/api';
import { Button } from '@/components/ui/button';
import { Modal } from '@/components/ui/modal';
import { Input } from '@/components/ui/input';
import { 
  FileText, 
  Download, 
  Trash2, 
  Filter, 
  Search, 
  ChevronLeft, 
  ChevronRight,
  MoreHorizontal,
  RefreshCw
} from 'lucide-react';
import clsx from 'clsx';

interface DocumentTableProps {
  documents: Document[];
  totalCount: number;
  currentPage: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onFilterChange: (filters: Partial<QueryParams>) => void;
  onDeleteDocument: (documentId: string) => void;
  onRefresh: () => void;
}

export function DocumentTable({
  documents,
  totalCount,
  currentPage,
  pageSize,
  onPageChange,
  onFilterChange,
  onDeleteDocument,
  onRefresh,
}: DocumentTableProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<Document | null>(null);
  const [sortField, setSortField] = useState<string>('uploaded_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const totalPages = Math.ceil(totalCount / pageSize);

  const handleSearch = () => {
    onFilterChange({
      search: searchQuery || undefined,
      page: 1,
    });
  };

  const handleStatusFilter = (status: string) => {
    setSelectedStatus(status);
    onFilterChange({
      filters: status === 'all' ? undefined : { status },
      page: 1,
    });
  };

  const handleSort = (field: string) => {
    const newOrder = sortField === field && sortOrder === 'asc' ? 'desc' : 'asc';
    setSortField(field);
    setSortOrder(newOrder);
    onFilterChange({
      sort_by: field,
      sort_order: newOrder,
    });
  };

  const handleDeleteClick = (document: Document) => {
    setDocumentToDelete(document);
    setDeleteModalOpen(true);
  };

  const handleConfirmDelete = () => {
    if (documentToDelete) {
      onDeleteDocument(documentToDelete.id);
      setDeleteModalOpen(false);
      setDocumentToDelete(null);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusBadge = (status: Document['upload_status']) => {
    const baseClasses = 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium';
    
    switch (status) {
      case 'completed':
        return <span className={clsx(baseClasses, 'bg-success-100 text-success-800 dark:bg-success-900/20 dark:text-success-400')}>Completed</span>;
      case 'processing':
        return <span className={clsx(baseClasses, 'bg-warning-100 text-warning-800 dark:bg-warning-900/20 dark:text-warning-400')}>Processing</span>;
      case 'failed':
        return <span className={clsx(baseClasses, 'bg-error-100 text-error-800 dark:bg-error-900/20 dark:text-error-400')}>Failed</span>;
      default:
        return <span className={clsx(baseClasses, 'bg-secondary-100 text-secondary-800 dark:bg-secondary-900/20 dark:text-secondary-400')}>Pending</span>;
    }
  };

  const getSortIcon = (field: string) => {
    if (sortField !== field) return null;
    return sortOrder === 'asc' ? '↑' : '↓';
  };

  return (
    <>
      <div className="space-y-4">
        {/* Filters and Search */}
        <div className="flex flex-col sm:flex-row gap-4 p-4 border-b border-secondary-200 dark:border-secondary-700">
          <div className="flex-1 flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search documents..."
                className="pl-10"
              />
            </div>
            <Button onClick={handleSearch} variant="outline" size="icon">
              <Search className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex gap-2">
            {/* Status Filter */}
            <select
              value={selectedStatus}
              onChange={(e) => handleStatusFilter(e.target.value)}
              className="px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-md bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="processing">Processing</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>

            <Button onClick={onRefresh} variant="outline" size="icon">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          {documents.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="h-12 w-12 mx-auto mb-4 text-secondary-400" />
              <h3 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-2">
                No documents found
              </h3>
              <p className="text-secondary-600 dark:text-secondary-400">
                Upload some documents to get started
              </p>
            </div>
          ) : (
            <table className="min-w-full divide-y divide-secondary-200 dark:divide-secondary-700">
              <thead className="bg-secondary-50 dark:bg-secondary-800">
                <tr>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider cursor-pointer hover:text-secondary-700 dark:hover:text-secondary-300"
                    onClick={() => handleSort('filename')}
                  >
                    Name {getSortIcon('filename')}
                  </th>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider cursor-pointer hover:text-secondary-700 dark:hover:text-secondary-300"
                    onClick={() => handleSort('file_size')}
                  >
                    Size {getSortIcon('file_size')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th
                    className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider cursor-pointer hover:text-secondary-700 dark:hover:text-secondary-300"
                    onClick={() => handleSort('uploaded_at')}
                  >
                    Uploaded {getSortIcon('uploaded_at')}
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 dark:text-secondary-400 uppercase tracking-wider">
                    Tags
                  </th>
                  <th className="relative px-6 py-3">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-secondary-900 divide-y divide-secondary-200 dark:divide-secondary-700">
                {documents.map((document) => (
                  <tr key={document.id} className="hover:bg-secondary-50 dark:hover:bg-secondary-800">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="h-5 w-5 text-secondary-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
                            {document.original_filename}
                          </div>
                          <div className="text-sm text-secondary-500 dark:text-secondary-400">
                            {document.mime_type}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-500 dark:text-secondary-400">
                      {formatFileSize(document.file_size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {getStatusBadge(document.upload_status)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-500 dark:text-secondary-400">
                      {formatDate(document.uploaded_at)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex flex-wrap gap-1">
                        {document.tags.slice(0, 2).map((tag, index) => (
                          <span
                            key={index}
                            className="inline-flex items-center px-2 py-1 rounded-md text-xs bg-primary-100 text-primary-800 dark:bg-primary-900/20 dark:text-primary-400"
                          >
                            {tag}
                          </span>
                        ))}
                        {document.tags.length > 2 && (
                          <span className="text-xs text-secondary-500 dark:text-secondary-400">
                            +{document.tags.length - 2} more
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={document.upload_status !== 'completed'}
                          title={document.upload_status !== 'completed' ? 'Document not ready for download' : 'Download document'}
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteClick(document)}
                          className="text-error-600 hover:text-error-700 dark:text-error-400 dark:hover:text-error-300"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-secondary-200 dark:border-secondary-700">
            <div className="flex-1 flex justify-between sm:hidden">
              <Button
                onClick={() => onPageChange(currentPage - 1)}
                disabled={currentPage === 1}
                variant="outline"
              >
                Previous
              </Button>
              <Button
                onClick={() => onPageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                variant="outline"
              >
                Next
              </Button>
            </div>

            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-secondary-700 dark:text-secondary-300">
                  Showing{' '}
                  <span className="font-medium">{(currentPage - 1) * pageSize + 1}</span>
                  {' '}to{' '}
                  <span className="font-medium">
                    {Math.min(currentPage * pageSize, totalCount)}
                  </span>
                  {' '}of{' '}
                  <span className="font-medium">{totalCount}</span>
                  {' '}results
                </p>
              </div>
              <div>
                <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                  <Button
                    onClick={() => onPageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    variant="outline"
                    className="rounded-r-none"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  
                  {/* Page numbers */}
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    const page = i + Math.max(1, currentPage - 2);
                    if (page > totalPages) return null;
                    
                    return (
                      <Button
                        key={page}
                        onClick={() => onPageChange(page)}
                        variant={currentPage === page ? "primary" : "outline"}
                        className="rounded-none"
                      >
                        {page}
                      </Button>
                    );
                  })}
                  
                  <Button
                    onClick={() => onPageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    variant="outline"
                    className="rounded-l-none"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </nav>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Delete Document"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-secondary-700 dark:text-secondary-300">
            Are you sure you want to delete &quot;{documentToDelete?.original_filename}&quot;? This action cannot be undone.
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