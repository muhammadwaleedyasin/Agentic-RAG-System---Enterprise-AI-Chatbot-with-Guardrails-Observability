'use client';

import { useState, useRef, useCallback } from 'react';
import { Document } from '@/types/api';
import { apiClient } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Upload, File, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import clsx from 'clsx';

interface FileUploadProps {
  onUploadSuccess?: (document: Document) => void;
  onCancel?: () => void;
  maxFiles?: number;
  maxFileSize?: number; // in MB
  acceptedTypes?: string[];
}

interface UploadFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress?: number;
  error?: string;
  document?: Document;
}

export function FileUpload({
  onUploadSuccess,
  onCancel,
  maxFiles = 10,
  maxFileSize = 50, // 50MB default
  acceptedTypes = ['.pdf', '.txt', '.md', '.docx', '.doc'],
}: FileUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [tags, setTags] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptString = acceptedTypes.join(',');

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxFileSize * 1024 * 1024) {
      return `File size must be less than ${maxFileSize}MB`;
    }

    // Check file type
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(extension)) {
      return `File type not supported. Accepted types: ${acceptedTypes.join(', ')}`;
    }

    return null;
  };

  const addFiles = (newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles);
    
    if (files.length + fileArray.length > maxFiles) {
      alert(`Maximum ${maxFiles} files allowed`);
      return;
    }

    const validFiles: UploadFile[] = [];
    
    fileArray.forEach(file => {
      const error = validateFile(file);
      validFiles.push({
        file,
        id: Math.random().toString(36).substr(2, 9),
        status: error ? 'error' : 'pending',
        error: error || undefined,
      });
    });

    setFiles(prev => [...prev, ...validFiles]);
  };

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  const uploadFile = async (uploadFile: UploadFile) => {
    setFiles(prev => prev.map(f => 
      f.id === uploadFile.id 
        ? { ...f, status: 'uploading', progress: 0 }
        : f
    ));

    try {
      const tagArray = tags.split(',').map(tag => tag.trim()).filter(Boolean);
      const document = await apiClient.uploadDocument(
        uploadFile.file,
        { uploadedBy: 'user' },
        tagArray
      ) as Document;

      setFiles(prev => prev.map(f => 
        f.id === uploadFile.id 
          ? { ...f, status: 'success', progress: 100, document }
          : f
      ));

      if (onUploadSuccess) {
        onUploadSuccess(document);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setFiles(prev => prev.map(f => 
        f.id === uploadFile.id 
          ? { ...f, status: 'error', error: errorMessage }
          : f
      ));
    }
  };

  const uploadAllFiles = async () => {
    const pendingFiles = files.filter(f => f.status === 'pending');
    
    // Upload files sequentially to avoid overwhelming the server
    for (const file of pendingFiles) {
      await uploadFile(file);
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      addFiles(droppedFiles);
    }
  }, [addFiles]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(e.target.files);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusIcon = (status: UploadFile['status']) => {
    switch (status) {
      case 'uploading':
        return <Loader2 className="h-4 w-4 animate-spin text-primary-600" />;
      case 'success':
        return <CheckCircle className="h-4 w-4 text-success-600" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-error-600" />;
      default:
        return <File className="h-4 w-4 text-secondary-400" />;
    }
  };

  const canUpload = files.some(f => f.status === 'pending');
  const hasSuccessfulUploads = files.some(f => f.status === 'success');

  return (
    <div className="space-y-6">
      {/* Drop zone */}
      <Card
        className={clsx(
          'border-2 border-dashed p-8 text-center transition-colors',
          isDragOver
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/10'
            : 'border-secondary-300 dark:border-secondary-600 hover:border-secondary-400 dark:hover:border-secondary-500'
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <Upload className="h-12 w-12 mx-auto mb-4 text-secondary-400" />
        <h3 className="text-lg font-medium text-secondary-900 dark:text-secondary-100 mb-2">
          Upload Documents
        </h3>
        <p className="text-secondary-600 dark:text-secondary-400 mb-4">
          Drag and drop files here, or click to select files
        </p>
        <Button
          onClick={() => fileInputRef.current?.click()}
          variant="outline"
        >
          Select Files
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={acceptString}
          onChange={handleFileSelect}
          className="hidden"
        />
        <p className="text-xs text-secondary-500 dark:text-secondary-400 mt-4">
          Supported formats: {acceptedTypes.join(', ')} • Max size: {maxFileSize}MB • Max files: {maxFiles}
        </p>
      </Card>

      {/* Tags input */}
      <div>
        <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
          Tags (optional)
        </label>
        <Input
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="Enter tags separated by commas"
          className="w-full"
        />
        <p className="text-xs text-secondary-500 dark:text-secondary-400 mt-1">
          Add tags to help organize and find your documents
        </p>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-medium text-secondary-900 dark:text-secondary-100">
            Files ({files.length})
          </h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {files.map((uploadFile) => (
              <div
                key={uploadFile.id}
                className="flex items-center gap-3 p-3 bg-secondary-50 dark:bg-secondary-800 rounded-md"
              >
                {getStatusIcon(uploadFile.status)}
                
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-secondary-900 dark:text-secondary-100 truncate">
                    {uploadFile.file.name}
                  </p>
                  <p className="text-xs text-secondary-500 dark:text-secondary-400">
                    {formatFileSize(uploadFile.file.size)}
                  </p>
                  {uploadFile.error && (
                    <p className="text-xs text-error-600 dark:text-error-400 mt-1">
                      {uploadFile.error}
                    </p>
                  )}
                </div>

                {uploadFile.status === 'uploading' && uploadFile.progress !== undefined && (
                  <div className="w-16 text-xs text-secondary-600 dark:text-secondary-400">
                    {uploadFile.progress}%
                  </div>
                )}

                <button
                  onClick={() => removeFile(uploadFile.id)}
                  disabled={uploadFile.status === 'uploading'}
                  className="p-1 rounded hover:bg-secondary-200 dark:hover:bg-secondary-700 text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  aria-label="Remove file"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3 justify-end">
        {onCancel && (
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        )}
        
        {canUpload && (
          <Button onClick={uploadAllFiles}>
            Upload {files.filter(f => f.status === 'pending').length} Files
          </Button>
        )}

        {hasSuccessfulUploads && !canUpload && onCancel && (
          <Button onClick={onCancel}>
            Done
          </Button>
        )}
      </div>
    </div>
  );
}