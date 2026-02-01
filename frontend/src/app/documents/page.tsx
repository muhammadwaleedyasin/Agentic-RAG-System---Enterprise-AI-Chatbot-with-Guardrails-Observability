'use client';

import { useState, useEffect, useCallback } from 'react';
import { AuthGuard } from '@/components/auth/auth-guard';
import { FileUpload } from '@/components/documents/file-upload';
import { DocumentTable } from '@/components/documents/document-table';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Modal } from '@/components/ui/modal';
import { apiClient } from '@/lib/api';
import { Document, QueryParams } from '@/types/api';
import { Upload, FileText, Loader2 } from 'lucide-react';

function DocumentsPageContent() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [filters, setFilters] = useState<QueryParams>({
    page: 1,
    per_page: 20,
    sort_by: 'uploaded_at',
    sort_order: 'desc',
  });

  const loadDocuments = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await apiClient.getDocuments(filters);
      const data = response as any;
      
      if (data.data) {
        setDocuments(data.data);
        setTotalCount(data.total || 0);
      } else {
        setDocuments(data);
        setTotalCount(data.length || 0);
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setIsLoading(false);
    }
  }, [filters]);

  // Load documents
  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleUploadSuccess = (document: Document) => {
    setDocuments(prev => [document, ...prev]);
    setTotalCount(prev => prev + 1);
    setUploadModalOpen(false);
  };

  const handleDeleteDocument = async (documentId: string) => {
    try {
      await apiClient.deleteDocument(documentId);
      setDocuments(prev => prev.filter(doc => doc.id !== documentId));
      setTotalCount(prev => prev - 1);
    } catch (err) {
      console.error('Failed to delete document:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete document');
    }
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    setFilters(prev => ({ ...prev, page }));
  };

  const handleFilterChange = (newFilters: Partial<QueryParams>) => {
    setCurrentPage(1);
    setFilters(prev => ({ ...prev, ...newFilters, page: 1 }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-100">
            Documents
          </h1>
          <p className="text-secondary-600 dark:text-secondary-400 mt-1">
            Upload and manage your documents for the knowledge base
          </p>
        </div>
        <Button
          onClick={() => setUploadModalOpen(true)}
          className="flex items-center gap-2"
        >
          <Upload className="h-4 w-4" />
          Upload Documents
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
            <FileText className="h-4 w-4 text-secondary-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalCount}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing</CardTitle>
            <Loader2 className="h-4 w-4 text-warning-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {documents.filter(doc => doc.upload_status === 'processing').length}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Ready</CardTitle>
            <div className="h-4 w-4 rounded-full bg-success-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {documents.filter(doc => doc.upload_status === 'completed').length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-md p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-error-800 dark:text-error-200">
                Error loading documents
              </h3>
              <div className="mt-2 text-sm text-error-700 dark:text-error-300">
                {error}
              </div>
              <div className="mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={loadDocuments}
                >
                  Try Again
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Documents Table */}
      <Card>
        <CardContent className="p-0">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary-600" />
                <p className="text-secondary-600 dark:text-secondary-400">
                  Loading documents...
                </p>
              </div>
            </div>
          ) : (
            <DocumentTable
              documents={documents}
              totalCount={totalCount}
              currentPage={currentPage}
              pageSize={filters.per_page || 20}
              onPageChange={handlePageChange}
              onFilterChange={handleFilterChange}
              onDeleteDocument={handleDeleteDocument}
              onRefresh={loadDocuments}
            />
          )}
        </CardContent>
      </Card>

      {/* Upload Modal */}
      <Modal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        title="Upload Documents"
        size="lg"
      >
        <FileUpload
          onUploadSuccess={handleUploadSuccess}
          onCancel={() => setUploadModalOpen(false)}
        />
      </Modal>
    </div>
  );
}

export default function DocumentsPage() {
  return (
    <AuthGuard requireAuth={true}>
      <DocumentsPageContent />
    </AuthGuard>
  );
}