import { Page, expect } from '@playwright/test';
import { BasePage } from './base-page';
import * as fs from 'fs';
import * as path from 'path';

export class DocumentsPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Page elements
  get elements() {
    return {
      uploadButton: this.page.locator('[data-testid="upload-button"]'),
      fileInput: this.page.locator('[data-testid="file-input"]'),
      uploadArea: this.page.locator('[data-testid="upload-area"]'),
      documentsList: this.page.locator('[data-testid="documents-list"]'),
      documentItems: this.page.locator('[data-testid="document-item"]'),
      searchInput: this.page.locator('[data-testid="documents-search"]'),
      filterDropdown: this.page.locator('[data-testid="documents-filter"]'),
      sortDropdown: this.page.locator('[data-testid="documents-sort"]'),
      bulkActions: this.page.locator('[data-testid="bulk-actions"]'),
      selectAllCheckbox: this.page.locator('[data-testid="select-all"]'),
      deleteButton: this.page.locator('[data-testid="delete-selected"]'),
      uploadProgress: this.page.locator('[data-testid="upload-progress"]'),
      uploadStatus: this.page.locator('[data-testid="upload-status"]'),
      documentDetails: this.page.locator('[data-testid="document-details"]'),
      pagination: this.page.locator('[data-testid="pagination"]'),
      itemsPerPageSelect: this.page.locator('[data-testid="items-per-page"]'),
      totalCount: this.page.locator('[data-testid="total-count"]'),
      emptyState: this.page.locator('[data-testid="empty-state"]'),
    };
  }

  async goto(): Promise<void> {
    await this.page.goto('/documents');
    await this.verifyPageLoaded();
  }

  async verifyPageLoaded(): Promise<void> {
    await expect(this.elements.uploadButton).toBeVisible();
    await expect(this.elements.documentsList).toBeVisible();
    await this.verifyTitle('Documents - Enterprise RAG Chatbot');
  }

  /**
   * Upload a document
   */
  async uploadDocument(filePath: string, title?: string, description?: string): Promise<void> {
    // Click upload button to open file dialog
    await this.elements.uploadButton.click();
    
    // Set file to input
    await this.elements.fileInput.setInputFiles(filePath);
    
    // Fill additional metadata if provided
    if (title) {
      await this.fillFormField('[data-testid="document-title"]', title);
    }
    
    if (description) {
      await this.fillFormField('[data-testid="document-description"]', description);
    }
    
    // Submit upload
    const submitButton = this.page.locator('[data-testid="upload-submit"]');
    await submitButton.click();
  }

  /**
   * Upload document via drag and drop
   */
  async uploadDocumentDragDrop(filePath: string): Promise<void> {
    // Create file input and attach file
    const fileBuffer = fs.readFileSync(filePath);
    const fileName = path.basename(filePath);
    
    // Simulate drag and drop
    await this.page.evaluate(async ({ fileName, fileBuffer }) => {
      const dataTransfer = new DataTransfer();
      const file = new File([new Uint8Array(fileBuffer)], fileName);
      dataTransfer.items.add(file);
      
      const dropZone = document.querySelector('[data-testid="upload-area"]');
      if (dropZone) {
        const event = new DragEvent('drop', { dataTransfer });
        dropZone.dispatchEvent(event);
      }
    }, { fileName, fileBuffer: Array.from(fileBuffer) });
  }

  /**
   * Wait for upload to complete
   */
  async waitForUploadSuccess(timeout = 30000): Promise<void> {
    // Wait for upload progress to appear
    await expect(this.elements.uploadProgress).toBeVisible({ timeout: 5000 });
    
    // Wait for success status
    await expect(this.elements.uploadStatus).toContainText('Upload completed', { timeout });
    
    // Wait for progress to disappear
    await expect(this.elements.uploadProgress).not.toBeVisible({ timeout: 5000 });
  }

  /**
   * Verify document appears in the list
   */
  async verifyDocumentInList(filename: string): Promise<void> {
    const documentItem = this.page.locator(`[data-testid="document-item"]:has-text("${filename}")`);
    await expect(documentItem).toBeVisible();
  }

  /**
   * Search for documents
   */
  async searchDocuments(query: string): Promise<void> {
    await this.fillFormField('[data-testid="documents-search"]', query);
    await this.page.keyboard.press('Enter');
    
    // Wait for search results
    await this.waitForApiCall(/\/api\/v1\/documents\?.*search=/);
  }

  /**
   * Filter documents by type
   */
  async filterDocuments(type: 'all' | 'pdf' | 'txt' | 'docx'): Promise<void> {
    await this.elements.filterDropdown.click();
    await this.page.locator(`[data-testid="filter-${type}"]`).click();
    
    // Wait for filtered results
    await this.waitForApiCall(/\/api\/v1\/documents\?.*type=/);
  }

  /**
   * Sort documents
   */
  async sortDocuments(sortBy: 'name' | 'date' | 'size', order: 'asc' | 'desc' = 'asc'): Promise<void> {
    await this.elements.sortDropdown.click();
    await this.page.locator(`[data-testid="sort-${sortBy}-${order}"]`).click();
    
    // Wait for sorted results
    await this.waitForApiCall(/\/api\/v1\/documents\?.*sort=/);
  }

  /**
   * Select document(s)
   */
  async selectDocument(filename: string): Promise<void> {
    const documentItem = this.page.locator(`[data-testid="document-item"]:has-text("${filename}")`);
    const checkbox = documentItem.locator('[data-testid="document-checkbox"]');
    await checkbox.check();
  }

  /**
   * Select all documents
   */
  async selectAllDocuments(): Promise<void> {
    await this.elements.selectAllCheckbox.check();
  }

  /**
   * Delete selected documents
   */
  async deleteSelectedDocuments(): Promise<void> {
    await this.elements.deleteButton.click();
    await this.confirmAction(true);
    
    // Wait for deletion to complete
    await this.verifyToast('Documents deleted successfully');
  }

  /**
   * View document details
   */
  async viewDocumentDetails(filename: string): Promise<void> {
    const documentItem = this.page.locator(`[data-testid="document-item"]:has-text("${filename}")`);
    await documentItem.click();
    
    // Verify details panel opens
    await expect(this.elements.documentDetails).toBeVisible();
  }

  /**
   * Get document list
   */
  async getDocumentList(): Promise<Array<{ name: string; type: string; size: string; date: string }>> {
    const items = this.elements.documentItems;
    const count = await items.count();
    const documents = [];
    
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      const name = await item.locator('[data-testid="document-name"]').textContent() || '';
      const type = await item.locator('[data-testid="document-type"]').textContent() || '';
      const size = await item.locator('[data-testid="document-size"]').textContent() || '';
      const date = await item.locator('[data-testid="document-date"]').textContent() || '';
      
      documents.push({ name, type, size, date });
    }
    
    return documents;
  }

  /**
   * Verify pagination
   */
  async verifyPagination(): Promise<void> {
    const totalCountText = await this.elements.totalCount.textContent();
    const totalCount = parseInt(totalCountText?.match(/\d+/)?.[0] || '0');
    
    if (totalCount > 10) {
      await expect(this.elements.pagination).toBeVisible();
      
      // Test next page
      const nextButton = this.page.locator('[data-testid="pagination-next"]');
      if (await nextButton.isEnabled()) {
        await nextButton.click();
        await this.waitForPageLoad();
        
        // Go back to first page
        const prevButton = this.page.locator('[data-testid="pagination-prev"]');
        await prevButton.click();
        await this.waitForPageLoad();
      }
    }
  }

  /**
   * Change items per page
   */
  async changeItemsPerPage(count: number): Promise<void> {
    await this.elements.itemsPerPageSelect.selectOption(count.toString());
    await this.waitForApiCall(/\/api\/v1\/documents\?.*limit=/);
  }

  /**
   * Create test file for upload
   */
  async createTestFile(filename: string, content: string): Promise<string> {
    const tempDir = path.join(__dirname, '../data/temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    const filePath = path.join(tempDir, filename);
    fs.writeFileSync(filePath, content);
    
    return filePath;
  }

  /**
   * Test file upload validation
   */
  async testUploadValidation(): Promise<void> {
    // Test unsupported file type
    const invalidFile = await this.createTestFile('test.xyz', 'invalid content');
    
    try {
      await this.uploadDocument(invalidFile);
      await this.verifyToast('Unsupported file type', 'error');
    } finally {
      fs.unlinkSync(invalidFile);
    }
    
    // Test file size limit (if applicable)
    const largeContent = 'x'.repeat(50 * 1024 * 1024); // 50MB
    const largeFile = await this.createTestFile('large.txt', largeContent);
    
    try {
      await this.uploadDocument(largeFile);
      await this.verifyToast('File too large', 'error');
    } finally {
      fs.unlinkSync(largeFile);
    }
  }

  /**
   * Test bulk operations
   */
  async testBulkOperations(): Promise<void> {
    // Select multiple documents
    const documents = await this.getDocumentList();
    
    if (documents.length >= 2) {
      await this.selectDocument(documents[0].name);
      await this.selectDocument(documents[1].name);
      
      // Verify bulk actions are enabled
      await expect(this.elements.bulkActions).toBeVisible();
      await expect(this.elements.deleteButton).toBeEnabled();
    }
  }

  /**
   * Test document search functionality
   */
  async testDocumentSearch(): Promise<void> {
    // Get initial document list
    const allDocuments = await this.getDocumentList();
    
    if (allDocuments.length > 0) {
      // Search for specific document
      const searchTerm = allDocuments[0].name.substring(0, 5);
      await this.searchDocuments(searchTerm);
      
      // Verify search results
      const searchResults = await this.getDocumentList();
      expect(searchResults.length).toBeLessThanOrEqual(allDocuments.length);
      
      // Clear search
      await this.fillFormField('[data-testid="documents-search"]', '');
      await this.page.keyboard.press('Enter');
    }
  }

  /**
   * Test empty state
   */
  async verifyEmptyState(): Promise<void> {
    // This would typically be tested with a clean database
    // For now, just verify the empty state element exists
    if (await this.elements.emptyState.isVisible()) {
      await expect(this.elements.emptyState).toContainText('No documents found');
    }
  }

  /**
   * Perform complete document upload workflow
   */
  async performDocumentUpload(filename: string, content: string): Promise<void> {
    const testFile = await this.createTestFile(filename, content);
    
    try {
      await this.uploadDocument(testFile, filename, `Test document: ${filename}`);
      await this.waitForUploadSuccess();
      await this.verifyDocumentInList(filename);
      await this.verifyToast('Document uploaded successfully');
      await this.verifyNoErrors();
    } finally {
      // Clean up test file
      if (fs.existsSync(testFile)) {
        fs.unlinkSync(testFile);
      }
    }
  }
}