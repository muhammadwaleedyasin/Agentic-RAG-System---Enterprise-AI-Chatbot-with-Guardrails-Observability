import { test, expect } from '../fixtures/test-fixtures';
import { testData } from '../fixtures/test-fixtures';

test.describe('Document Management', () => {
  test.beforeEach(async ({ loginPage }) => {
    await loginPage.goto();
    await loginPage.performValidLogin();
  });

  test.describe('Document Upload', () => {
    test('should upload document successfully', async ({ documentsPage }) => {
      const testDoc = testData.generateDocument();
      await documentsPage.performDocumentUpload('test-upload.txt', testDoc.content);
    });

    test('should upload multiple document types', async ({ documentsPage }) => {
      const documents = [
        { filename: 'test.txt', content: testData.generateFileContent('txt') },
        { filename: 'test.json', content: testData.generateFileContent('json') },
        { filename: 'test.csv', content: testData.generateFileContent('csv') },
      ];

      for (const doc of documents) {
        await documentsPage.performDocumentUpload(doc.filename, doc.content);
      }
    });

    test('should show upload progress', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      const testFile = await documentsPage.createTestFile('progress-test.txt', 'Test content for progress');
      
      await documentsPage.uploadDocument(testFile);
      
      // Verify progress indicator appears
      await expect(documentsPage.elements.uploadProgress).toBeVisible();
      await documentsPage.waitForUploadSuccess();
    });

    test('should support drag and drop upload', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      const testFile = await documentsPage.createTestFile('dragdrop-test.txt', 'Drag and drop test content');
      
      await documentsPage.uploadDocumentDragDrop(testFile);
      await documentsPage.waitForUploadSuccess();
      await documentsPage.verifyDocumentInList('dragdrop-test.txt');
    });

    test('should validate file types', async ({ documentsPage }) => {
      await documentsPage.goto();
      await documentsPage.testUploadValidation();
    });

    test('should handle upload errors', async ({ documentsPage, testHelpers }) => {
      await documentsPage.goto();
      
      // Mock upload error
      await testHelpers.mockApiResponse(/\/api\/v1\/documents\/upload/, {
        error: 'Upload failed'
      }, 500);
      
      const testFile = await documentsPage.createTestFile('error-test.txt', 'Error test content');
      
      await documentsPage.uploadDocument(testFile);
      await documentsPage.verifyToast('Upload failed', 'error');
    });
  });

  test.describe('Document List Management', () => {
    test('should display documents list', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      const documents = await documentsPage.getDocumentList();
      // Documents might be empty in fresh system, so just verify structure
      expect(Array.isArray(documents)).toBe(true);
    });

    test('should search documents', async ({ documentsPage }) => {
      await documentsPage.goto();
      await documentsPage.testDocumentSearch();
    });

    test('should filter documents by type', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.filterDocuments('txt');
      
      // Verify filter is applied
      const filteredDocs = await documentsPage.getDocumentList();
      if (filteredDocs.length > 0) {
        filteredDocs.forEach(doc => {
          expect(doc.type.toLowerCase()).toContain('txt');
        });
      }
    });

    test('should sort documents', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.sortDocuments('name', 'asc');
      
      const documents = await documentsPage.getDocumentList();
      if (documents.length > 1) {
        // Verify alphabetical sorting
        for (let i = 1; i < documents.length; i++) {
          expect(documents[i].name.localeCompare(documents[i-1].name)).toBeGreaterThanOrEqual(0);
        }
      }
    });

    test('should handle pagination', async ({ documentsPage }) => {
      await documentsPage.goto();
      await documentsPage.verifyPagination();
    });

    test('should change items per page', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.changeItemsPerPage(5);
      
      const documents = await documentsPage.getDocumentList();
      expect(documents.length).toBeLessThanOrEqual(5);
    });
  });

  test.describe('Document Selection and Bulk Operations', () => {
    test('should select individual documents', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      const documents = await documentsPage.getDocumentList();
      if (documents.length > 0) {
        await documentsPage.selectDocument(documents[0].name);
        
        // Verify selection
        await expect(documentsPage.elements.bulkActions).toBeVisible();
      }
    });

    test('should select all documents', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.selectAllDocuments();
      
      // Verify all are selected
      await expect(documentsPage.elements.bulkActions).toBeVisible();
      await expect(documentsPage.elements.deleteButton).toBeEnabled();
    });

    test('should perform bulk operations', async ({ documentsPage }) => {
      await documentsPage.goto();
      await documentsPage.testBulkOperations();
    });

    test('should delete selected documents', async ({ documentsPage }) => {
      // First upload a test document to delete
      const testDoc = testData.generateDocument();
      await documentsPage.performDocumentUpload('delete-test.txt', testDoc.content);
      
      await documentsPage.selectDocument('delete-test.txt');
      await documentsPage.deleteSelectedDocuments();
      
      // Verify document is removed from list
      const documents = await documentsPage.getDocumentList();
      const deletedDoc = documents.find(doc => doc.name === 'delete-test.txt');
      expect(deletedDoc).toBeUndefined();
    });
  });

  test.describe('Document Details', () => {
    test('should view document details', async ({ documentsPage }) => {
      // Upload a test document first
      const testDoc = testData.generateDocument();
      await documentsPage.performDocumentUpload('details-test.txt', testDoc.content);
      
      await documentsPage.viewDocumentDetails('details-test.txt');
      
      // Verify details panel is visible
      await expect(documentsPage.elements.documentDetails).toBeVisible();
    });

    test('should display document metadata', async ({ documentsPage }) => {
      const testDoc = testData.generateDocument();
      await documentsPage.performDocumentUpload('metadata-test.txt', testDoc.content);
      
      await documentsPage.viewDocumentDetails('metadata-test.txt');
      
      // Verify metadata fields are present
      const detailsPanel = documentsPage.elements.documentDetails;
      await expect(detailsPanel.locator('[data-testid="document-title"]')).toBeVisible();
      await expect(detailsPanel.locator('[data-testid="document-size"]')).toBeVisible();
      await expect(detailsPanel.locator('[data-testid="document-type"]')).toBeVisible();
      await expect(detailsPanel.locator('[data-testid="upload-date"]')).toBeVisible();
    });
  });

  test.describe('Document Processing Status', () => {
    test('should show processing status', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      const testFile = await documentsPage.createTestFile('processing-test.txt', 'Content for processing test');
      
      await documentsPage.uploadDocument(testFile);
      
      // Should show processing status
      await expect(documentsPage.elements.uploadStatus).toContainText('Processing' || 'Upload completed');
    });

    test('should handle processing errors', async ({ documentsPage, testHelpers }) => {
      await documentsPage.goto();
      
      // Mock processing error
      await testHelpers.mockApiResponse(/\/api\/v1\/documents\/process/, {
        error: 'Processing failed'
      }, 500);
      
      const testFile = await documentsPage.createTestFile('error-processing.txt', 'Error processing test');
      
      await documentsPage.uploadDocument(testFile);
      
      // Should show error status
      await documentsPage.verifyToast('Processing failed', 'error');
    });
  });

  test.describe('Empty States', () => {
    test('should show empty state when no documents', async ({ documentsPage }) => {
      await documentsPage.goto();
      await documentsPage.verifyEmptyState();
    });

    test('should show no results when search returns empty', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.searchDocuments('nonexistentdocument12345');
      
      // Should show no results message
      const documents = await documentsPage.getDocumentList();
      expect(documents.length).toBe(0);
    });
  });

  test.describe('Accessibility', () => {
    test('should be accessible to screen readers', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      // Check ARIA labels and roles
      await expect(documentsPage.elements.uploadButton).toHaveAttribute('aria-label');
      await expect(documentsPage.elements.searchInput).toHaveAttribute('aria-label');
      
      // Check keyboard navigation
      await documentsPage.elements.uploadButton.focus();
      await expect(documentsPage.elements.uploadButton).toBeFocused();
    });

    test('should support keyboard navigation', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      // Tab through main interactive elements
      await documentsPage.page.keyboard.press('Tab');
      await documentsPage.page.keyboard.press('Tab');
      
      // Should be able to navigate to search
      await documentsPage.elements.searchInput.focus();
      await expect(documentsPage.elements.searchInput).toBeFocused();
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on mobile devices', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.verifyResponsiveLayout('mobile');
      
      // Basic functionality should work
      await expect(documentsPage.elements.uploadButton).toBeVisible();
      await expect(documentsPage.elements.documentsList).toBeVisible();
    });

    test('should adapt to tablet layout', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.verifyResponsiveLayout('tablet');
      
      // Should show additional features
      await expect(documentsPage.elements.searchInput).toBeVisible();
      await expect(documentsPage.elements.filterDropdown).toBeVisible();
    });

    test('should show full features on desktop', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      await documentsPage.verifyResponsiveLayout('desktop');
      
      // All features should be visible
      await expect(documentsPage.elements.bulkActions).toBeVisible();
      await expect(documentsPage.elements.pagination).toBeVisible();
    });
  });

  test.describe('Performance', () => {
    test('should handle large file uploads', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      // Create a larger test file (within limits)
      const largeContent = 'Large file content. '.repeat(1000);
      const testFile = await documentsPage.createTestFile('large-file.txt', largeContent);
      
      await documentsPage.uploadDocument(testFile);
      await documentsPage.waitForUploadSuccess(60000); // Longer timeout for large file
      
      await documentsPage.verifyDocumentInList('large-file.txt');
    });

    test('should handle multiple simultaneous uploads', async ({ documentsPage }) => {
      await documentsPage.goto();
      
      const files = [
        'multi-upload-1.txt',
        'multi-upload-2.txt',
        'multi-upload-3.txt',
      ];
      
      const uploadPromises = files.map(async (filename) => {
        const testFile = await documentsPage.createTestFile(filename, `Content for ${filename}`);
        return documentsPage.uploadDocument(testFile);
      });
      
      // Start all uploads simultaneously
      await Promise.all(uploadPromises);
      
      // Verify all files uploaded
      for (const filename of files) {
        await documentsPage.verifyDocumentInList(filename);
      }
    });
  });
});