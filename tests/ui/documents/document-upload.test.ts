import { test, expect } from '@playwright/test';
import { test as authTest } from '../fixtures/auth-fixtures';
import path from 'path';

/**
 * Document Upload Tests
 * Tests file upload functionality, progress tracking, and validation
 */

test.describe('Document Upload', () => {
  test.beforeEach(async ({ page }) => {
    // Login as user with upload permissions
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
    await page.goto('/documents');
  });

  test('should display upload interface correctly', async ({ page }) => {
    // Verify upload components
    await expect(page.locator('[data-testid="upload-area"]')).toBeVisible();
    await expect(page.locator('[data-testid="file-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="upload-button"]')).toBeVisible();
    
    // Verify drag and drop area
    await expect(page.locator('[data-testid="drop-zone"]')).toBeVisible();
    await expect(page.locator('[data-testid="drop-zone"]'))
      .toContainText('Drag and drop files here or click to select');
    
    // Verify supported formats
    await expect(page.locator('[data-testid="supported-formats"]')).toBeVisible();
    await expect(page.locator('[data-testid="supported-formats"]'))
      .toContainText('PDF, DOC, DOCX, TXT, MD');
  });

  test('should upload single PDF file successfully', async ({ page }) => {
    // Create test file
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    
    // Upload file
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    
    // Verify file appears in upload queue
    await expect(page.locator('[data-testid="upload-queue"]')).toBeVisible();
    await expect(page.locator('[data-testid="file-item"]')).toHaveCount(1);
    await expect(page.locator('[data-testid="file-name"]')).toContainText('test-document.pdf');
    
    // Start upload
    await page.click('[data-testid="start-upload"]');
    
    // Verify upload progress
    await expect(page.locator('[data-testid="upload-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();
    
    // Wait for upload completion
    await expect(page.locator('[data-testid="upload-success"]')).toBeVisible();
    await expect(page.locator('[data-testid="upload-complete"]'))
      .toContainText('Upload completed successfully');
  });

  test('should upload multiple files simultaneously', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/document1.pdf'),
      path.join(__dirname, '../fixtures/document2.txt'),
      path.join(__dirname, '../fixtures/document3.docx')
    ];
    
    // Upload multiple files
    await page.setInputFiles('[data-testid="file-input"]', files);
    
    // Verify all files in queue
    await expect(page.locator('[data-testid="file-item"]')).toHaveCount(3);
    
    // Start batch upload
    await page.click('[data-testid="upload-all"]');
    
    // Verify individual progress bars
    await expect(page.locator('[data-testid="file-progress"]')).toHaveCount(3);
    
    // Verify batch progress
    await expect(page.locator('[data-testid="batch-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="batch-progress-text"]'))
      .toContainText('Uploading 3 files');
    
    // Wait for all uploads to complete
    await expect(page.locator('[data-testid="all-uploads-complete"]')).toBeVisible();
  });

  test('should handle drag and drop upload', async ({ page }) => {
    // Simulate drag and drop
    const dataTransfer = await page.evaluateHandle(() => new DataTransfer());
    
    // Create mock file
    await page.evaluate(() => {
      const file = new File(['test content'], 'dropped-file.txt', { type: 'text/plain' });
      const dt = new DataTransfer();
      dt.items.add(file);
      
      const dropZone = document.querySelector('[data-testid="drop-zone"]');
      const dragEvent = new DragEvent('drop', {
        dataTransfer: dt,
        bubbles: true
      });
      
      dropZone.dispatchEvent(dragEvent);
    });
    
    // Verify file was added
    await expect(page.locator('[data-testid="file-item"]')).toHaveCount(1);
    await expect(page.locator('[data-testid="file-name"]')).toContainText('dropped-file.txt');
  });

  test('should show file validation errors', async ({ page }) => {
    // Try to upload unsupported file type
    const invalidFile = path.join(__dirname, '../fixtures/invalid-file.exe');
    
    await page.setInputFiles('[data-testid="file-input"]', invalidFile);
    
    // Should show validation error
    await expect(page.locator('[data-testid="file-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="file-error"]'))
      .toContainText('Unsupported file type');
    
    // File should not be added to queue
    await expect(page.locator('[data-testid="file-item"]')).toHaveCount(0);
  });

  test('should handle file size validation', async ({ page }) => {
    // Mock large file
    await page.evaluate(() => {
      const largeFile = new File(['x'.repeat(100 * 1024 * 1024)], 'large-file.pdf', { 
        type: 'application/pdf' 
      });
      
      const input = document.querySelector('[data-testid="file-input"]');
      const dt = new DataTransfer();
      dt.items.add(largeFile);
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    
    // Should show size error
    await expect(page.locator('[data-testid="file-size-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="file-size-error"]'))
      .toContainText('File size exceeds limit');
  });

  test('should display upload progress accurately', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/medium-file.pdf');
    
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    
    // Mock slow upload for progress testing
    await page.route('/api/v1/documents/upload', async route => {
      // Simulate chunked upload responses
      const chunks = 10;
      for (let i = 1; i <= chunks; i++) {
        await new Promise(resolve => setTimeout(resolve, 200));
        // Would send progress updates via WebSocket in real implementation
      }
      route.fulfill({ status: 200, body: JSON.stringify({ success: true }) });
    });
    
    await page.click('[data-testid="start-upload"]');
    
    // Verify progress updates
    await expect(page.locator('[data-testid="progress-percentage"]')).toBeVisible();
    
    // Check that progress increases
    let previousProgress = 0;
    for (let i = 0; i < 5; i++) {
      await page.waitForTimeout(500);
      const progressText = await page.locator('[data-testid="progress-percentage"]').textContent();
      const currentProgress = parseInt(progressText.replace('%', ''));
      
      expect(currentProgress).toBeGreaterThanOrEqual(previousProgress);
      previousProgress = currentProgress;
    }
  });

  test('should handle upload cancellation', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/large-file.pdf');
    
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    await page.click('[data-testid="start-upload"]');
    
    // Cancel upload
    await page.click('[data-testid="cancel-upload"]');
    
    // Should show cancellation confirmation
    await expect(page.locator('[data-testid="upload-cancelled"]')).toBeVisible();
    
    // Progress should stop
    await expect(page.locator('[data-testid="upload-progress"]')).not.toBeVisible();
    
    // File should be removed from queue
    await expect(page.locator('[data-testid="file-item"]')).toHaveCount(0);
  });

  test('should retry failed uploads', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    
    // Simulate upload failure
    await page.route('/api/v1/documents/upload', route => route.abort());
    
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    await page.click('[data-testid="start-upload"]');
    
    // Should show error
    await expect(page.locator('[data-testid="upload-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="retry-upload"]')).toBeVisible();
    
    // Remove route block and retry
    await page.unroute('/api/v1/documents/upload');
    await page.click('[data-testid="retry-upload"]');
    
    // Should succeed
    await expect(page.locator('[data-testid="upload-success"]')).toBeVisible();
  });

  test('should show file metadata input', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    
    // Should show metadata form
    await expect(page.locator('[data-testid="file-metadata"]')).toBeVisible();
    
    // Fill metadata
    await page.fill('[data-testid="document-title"]', 'Custom Document Title');
    await page.fill('[data-testid="document-description"]', 'Test document description');
    await page.selectOption('[data-testid="document-category"]', 'technical');
    await page.fill('[data-testid="document-tags"]', 'test, document, pdf');
    
    // Start upload with metadata
    await page.click('[data-testid="start-upload"]');
    
    // Metadata should be included in upload
    const uploadRequest = page.waitForRequest('/api/v1/documents/upload');
    await uploadRequest;
    
    await expect(page.locator('[data-testid="upload-success"]')).toBeVisible();
  });

  test('should handle duplicate file detection', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    
    // Upload file first time
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    await page.click('[data-testid="start-upload"]');
    await expect(page.locator('[data-testid="upload-success"]')).toBeVisible();
    
    // Try to upload same file again
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    
    // Should show duplicate warning
    await expect(page.locator('[data-testid="duplicate-warning"]')).toBeVisible();
    await expect(page.locator('[data-testid="duplicate-warning"]'))
      .toContainText('This file already exists');
    
    // Should offer options
    await expect(page.locator('[data-testid="replace-file"]')).toBeVisible();
    await expect(page.locator('[data-testid="keep-both"]')).toBeVisible();
    await expect(page.locator('[data-testid="skip-upload"]')).toBeVisible();
  });

  test('should handle upload queue management', async ({ page }) => {
    const files = [
      path.join(__dirname, '../fixtures/file1.pdf'),
      path.join(__dirname, '../fixtures/file2.txt'),
      path.join(__dirname, '../fixtures/file3.docx')
    ];
    
    await page.setInputFiles('[data-testid="file-input"]', files);
    
    // Should show queue controls
    await expect(page.locator('[data-testid="queue-controls"]')).toBeVisible();
    await expect(page.locator('[data-testid="clear-queue"]')).toBeVisible();
    await expect(page.locator('[data-testid="remove-selected"]')).toBeVisible();
    
    // Select files to remove
    await page.check('[data-testid="file-checkbox"]:nth-child(1)');
    await page.check('[data-testid="file-checkbox"]:nth-child(2)');
    
    // Remove selected
    await page.click('[data-testid="remove-selected"]');
    
    // Should have only one file left
    await expect(page.locator('[data-testid="file-item"]')).toHaveCount(1);
    
    // Clear entire queue
    await page.click('[data-testid="clear-queue"]');
    await expect(page.locator('[data-testid="file-item"]')).toHaveCount(0);
  });

  test('should handle upload permissions', async ({ page }) => {
    // Login as viewer (read-only)
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'viewer_test');
    await page.fill('[data-testid="password-input"]', 'ViewerTest123!');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
    await page.goto('/documents');
    
    // Upload should be disabled
    await expect(page.locator('[data-testid="upload-area"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="upload-disabled-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="upload-disabled-message"]'))
      .toContainText('You do not have permission to upload documents');
  });

  test('should handle network errors during upload', async ({ page }) => {
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    
    // Simulate network error
    await page.route('/api/v1/documents/upload', route => {
      route.fulfill({ status: 500, body: 'Internal Server Error' });
    });
    
    await page.click('[data-testid="start-upload"]');
    
    // Should show network error
    await expect(page.locator('[data-testid="network-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="network-error"]'))
      .toContainText('Network error occurred');
    
    // Should offer retry
    await expect(page.locator('[data-testid="retry-upload"]')).toBeVisible();
  });

  test('should handle chunked upload for large files', async ({ page }) => {
    // Mock large file
    await page.evaluate(() => {
      const largeFile = new File(['x'.repeat(50 * 1024 * 1024)], 'large-file.pdf', { 
        type: 'application/pdf' 
      });
      
      const input = document.querySelector('[data-testid="file-input"]');
      const dt = new DataTransfer();
      dt.items.add(largeFile);
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    
    // Should indicate chunked upload
    await expect(page.locator('[data-testid="chunked-upload-indicator"]')).toBeVisible();
    
    await page.click('[data-testid="start-upload"]');
    
    // Should show chunk progress
    await expect(page.locator('[data-testid="chunk-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="chunks-completed"]')).toBeVisible();
  });

  test('should validate file content during upload', async ({ page }) => {
    // Upload file with corrupted content
    const corruptedFile = path.join(__dirname, '../fixtures/corrupted-file.pdf');
    
    await page.setInputFiles('[data-testid="file-input"]', corruptedFile);
    await page.click('[data-testid="start-upload"]');
    
    // Should detect content validation failure
    await expect(page.locator('[data-testid="content-validation-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="content-validation-error"]'))
      .toContainText('File content is invalid or corrupted');
  });

  test('should show upload history', async ({ page }) => {
    // Navigate to upload history
    await page.click('[data-testid="upload-history-tab"]');
    
    // Should show recent uploads
    await expect(page.locator('[data-testid="upload-history-list"]')).toBeVisible();
    
    // Should show upload details
    await expect(page.locator('[data-testid="upload-entry"]').first()).toBeVisible();
    await expect(page.locator('[data-testid="upload-timestamp"]').first()).toBeVisible();
    await expect(page.locator('[data-testid="upload-status"]').first()).toBeVisible();
    await expect(page.locator('[data-testid="upload-file-size"]').first()).toBeVisible();
  });
});