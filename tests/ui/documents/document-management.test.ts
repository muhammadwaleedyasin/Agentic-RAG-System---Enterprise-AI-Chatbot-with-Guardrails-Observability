import { test, expect } from '@playwright/test';
import { test as authTest } from '../fixtures/auth-fixtures';

/**
 * Document Management Tests
 * Tests document listing, search, organization, and management features
 */

test.describe('Document Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login as user
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'user_test');
    await page.fill('[data-testid="password-input"]', 'UserTest123!');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
    await page.goto('/documents');
  });

  test('should display document library correctly', async ({ page }) => {
    // Verify main components
    await expect(page.locator('[data-testid="document-library"]')).toBeVisible();
    await expect(page.locator('[data-testid="document-list"]')).toBeVisible();
    await expect(page.locator('[data-testid="document-search"]')).toBeVisible();
    await expect(page.locator('[data-testid="filter-controls"]')).toBeVisible();
    
    // Verify view options
    await expect(page.locator('[data-testid="list-view"]')).toBeVisible();
    await expect(page.locator('[data-testid="grid-view"]')).toBeVisible();
    await expect(page.locator('[data-testid="table-view"]')).toBeVisible();
    
    // Verify sorting options
    await expect(page.locator('[data-testid="sort-dropdown"]')).toBeVisible();
  });

  test('should list documents with correct information', async ({ page }) => {
    // Should show document cards/items
    await expect(page.locator('[data-testid="document-item"]')).toHaveCount(expect.any(Number));
    
    // Each document should show required info
    const firstDoc = page.locator('[data-testid="document-item"]').first();
    await expect(firstDoc.locator('[data-testid="document-title"]')).toBeVisible();
    await expect(firstDoc.locator('[data-testid="document-type"]')).toBeVisible();
    await expect(firstDoc.locator('[data-testid="document-size"]')).toBeVisible();
    await expect(firstDoc.locator('[data-testid="document-date"]')).toBeVisible();
    await expect(firstDoc.locator('[data-testid="document-status"]')).toBeVisible();
  });

  test('should search documents by title', async ({ page }) => {
    const searchTerm = 'policy';
    
    // Search for documents
    await page.fill('[data-testid="document-search"]', searchTerm);
    await page.keyboard.press('Enter');
    
    // Should filter results
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
    
    // All visible documents should contain search term
    const documentTitles = page.locator('[data-testid="document-title"]');
    const count = await documentTitles.count();
    
    for (let i = 0; i < count; i++) {
      const title = await documentTitles.nth(i).textContent();
      expect(title.toLowerCase()).toContain(searchTerm.toLowerCase());
    }
    
    // Should show search summary
    await expect(page.locator('[data-testid="search-summary"]')).toBeVisible();
    await expect(page.locator('[data-testid="search-summary"]'))
      .toContainText(`Found ${count} documents`);
  });

  test('should search documents by content', async ({ page }) => {
    // Enable content search
    await page.check('[data-testid="search-content-toggle"]');
    
    const searchTerm = 'artificial intelligence';
    await page.fill('[data-testid="document-search"]', searchTerm);
    await page.keyboard.press('Enter');
    
    // Should show content matches
    await expect(page.locator('[data-testid="content-matches"]')).toBeVisible();
    
    // Should highlight search terms in content previews
    await expect(page.locator('[data-testid="highlighted-match"]')).toBeVisible();
  });

  test('should filter documents by type', async ({ page }) => {
    // Open filter dropdown
    await page.click('[data-testid="filter-type"]');
    
    // Select PDF filter
    await page.check('[data-testid="filter-pdf"]');
    await page.click('[data-testid="apply-filters"]');
    
    // Should show only PDF documents
    const documentTypes = page.locator('[data-testid="document-type"]');
    const count = await documentTypes.count();
    
    for (let i = 0; i < count; i++) {
      const type = await documentTypes.nth(i).textContent();
      expect(type.toLowerCase()).toContain('pdf');
    }
    
    // Should show filter indicator
    await expect(page.locator('[data-testid="active-filters"]')).toBeVisible();
    await expect(page.locator('[data-testid="filter-pdf-active"]')).toBeVisible();
  });

  test('should filter documents by date range', async ({ page }) => {
    // Open date filter
    await page.click('[data-testid="filter-date"]');
    
    // Set date range
    await page.fill('[data-testid="date-from"]', '2024-01-01');
    await page.fill('[data-testid="date-to"]', '2024-12-31');
    await page.click('[data-testid="apply-date-filter"]');
    
    // Should filter by date range
    await expect(page.locator('[data-testid="date-filter-active"]')).toBeVisible();
    
    // Verify dates are within range
    const documentDates = page.locator('[data-testid="document-date"]');
    const count = await documentDates.count();
    
    for (let i = 0; i < count; i++) {
      const dateText = await documentDates.nth(i).textContent();
      const docDate = new Date(dateText);
      expect(docDate.getFullYear()).toBe(2024);
    }
  });

  test('should sort documents by different criteria', async ({ page }) => {
    // Sort by name
    await page.selectOption('[data-testid="sort-dropdown"]', 'name');
    
    // Verify alphabetical sorting
    let titles = await page.locator('[data-testid="document-title"]').allTextContents();
    const sortedTitles = [...titles].sort();
    expect(titles).toEqual(sortedTitles);
    
    // Sort by date (newest first)
    await page.selectOption('[data-testid="sort-dropdown"]', 'date-desc');
    
    // Verify date sorting
    const dates = await page.locator('[data-testid="document-date"]').allTextContents();
    const parsedDates = dates.map(d => new Date(d));
    
    for (let i = 0; i < parsedDates.length - 1; i++) {
      expect(parsedDates[i].getTime()).toBeGreaterThanOrEqual(parsedDates[i + 1].getTime());
    }
  });

  test('should switch between view modes', async ({ page }) => {
    // Start in list view
    await expect(page.locator('[data-testid="document-list"]')).toHaveClass(/list-view/);
    
    // Switch to grid view
    await page.click('[data-testid="grid-view"]');
    await expect(page.locator('[data-testid="document-list"]')).toHaveClass(/grid-view/);
    
    // Switch to table view
    await page.click('[data-testid="table-view"]');
    await expect(page.locator('[data-testid="document-table"]')).toBeVisible();
    
    // Table should have headers
    await expect(page.locator('[data-testid="table-header-name"]')).toBeVisible();
    await expect(page.locator('[data-testid="table-header-type"]')).toBeVisible();
    await expect(page.locator('[data-testid="table-header-size"]')).toBeVisible();
    await expect(page.locator('[data-testid="table-header-date"]')).toBeVisible();
  });

  test('should select multiple documents', async ({ page }) => {
    // Select multiple documents
    await page.check('[data-testid="document-checkbox"]:nth-child(1)');
    await page.check('[data-testid="document-checkbox"]:nth-child(2)');
    await page.check('[data-testid="document-checkbox"]:nth-child(3)');
    
    // Should show selection count
    await expect(page.locator('[data-testid="selection-count"]')).toBeVisible();
    await expect(page.locator('[data-testid="selection-count"]'))
      .toContainText('3 documents selected');
    
    // Should show bulk actions
    await expect(page.locator('[data-testid="bulk-actions"]')).toBeVisible();
    await expect(page.locator('[data-testid="bulk-delete"]')).toBeVisible();
    await expect(page.locator('[data-testid="bulk-download"]')).toBeVisible();
    await expect(page.locator('[data-testid="bulk-move"]')).toBeVisible();
  });

  test('should perform bulk operations', async ({ page }) => {
    // Select documents
    await page.check('[data-testid="select-all"]');
    
    // Bulk download
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="bulk-download"]');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toContain('documents');
    expect(download.suggestedFilename()).toContain('.zip');
    
    // Bulk move to folder
    await page.click('[data-testid="bulk-move"]');
    await expect(page.locator('[data-testid="move-dialog"]')).toBeVisible();
    
    await page.selectOption('[data-testid="destination-folder"]', 'archived');
    await page.click('[data-testid="confirm-move"]');
    
    await expect(page.locator('[data-testid="move-success"]')).toBeVisible();
  });

  test('should view document details', async ({ page }) => {
    // Click on document to view details
    await page.click('[data-testid="document-item"]');
    
    // Should open document details panel
    await expect(page.locator('[data-testid="document-details"]')).toBeVisible();
    
    // Should show comprehensive information
    await expect(page.locator('[data-testid="detail-title"]')).toBeVisible();
    await expect(page.locator('[data-testid="detail-type"]')).toBeVisible();
    await expect(page.locator('[data-testid="detail-size"]')).toBeVisible();
    await expect(page.locator('[data-testid="detail-created"]')).toBeVisible();
    await expect(page.locator('[data-testid="detail-modified"]')).toBeVisible();
    await expect(page.locator('[data-testid="detail-author"]')).toBeVisible();
    
    // Should show tags and categories
    await expect(page.locator('[data-testid="document-tags"]')).toBeVisible();
    await expect(page.locator('[data-testid="document-category"]')).toBeVisible();
    
    // Should show processing status
    await expect(page.locator('[data-testid="processing-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="embedding-status"]')).toBeVisible();
  });

  test('should preview document content', async ({ page }) => {
    // Click on document
    await page.click('[data-testid="document-item"]');
    
    // Click preview tab
    await page.click('[data-testid="preview-tab"]');
    
    // Should show document preview
    await expect(page.locator('[data-testid="document-preview"]')).toBeVisible();
    
    // For text documents, should show content
    const docType = await page.locator('[data-testid="document-type"]').textContent();
    
    if (docType.includes('txt') || docType.includes('md')) {
      await expect(page.locator('[data-testid="text-content"]')).toBeVisible();
    } else if (docType.includes('pdf')) {
      await expect(page.locator('[data-testid="pdf-viewer"]')).toBeVisible();
    }
    
    // Should have navigation for multi-page documents
    if (docType.includes('pdf') || docType.includes('doc')) {
      await expect(page.locator('[data-testid="page-navigation"]')).toBeVisible();
    }
  });

  test('should edit document metadata', async ({ page }) => {
    // Open document details
    await page.click('[data-testid="document-item"]');
    
    // Click edit button
    await page.click('[data-testid="edit-metadata"]');
    
    // Should show edit form
    await expect(page.locator('[data-testid="metadata-form"]')).toBeVisible();
    
    // Edit fields
    await page.fill('[data-testid="edit-title"]', 'Updated Document Title');
    await page.fill('[data-testid="edit-description"]', 'Updated description');
    await page.selectOption('[data-testid="edit-category"]', 'technical');
    await page.fill('[data-testid="edit-tags"]', 'updated, test, document');
    
    // Save changes
    await page.click('[data-testid="save-metadata"]');
    
    // Should show success message
    await expect(page.locator('[data-testid="metadata-updated"]')).toBeVisible();
    
    // Should reflect changes
    await expect(page.locator('[data-testid="detail-title"]'))
      .toContainText('Updated Document Title');
  });

  test('should delete document', async ({ page }) => {
    // Open document menu
    await page.click('[data-testid="document-menu"]');
    
    // Click delete
    await page.click('[data-testid="delete-document"]');
    
    // Should show confirmation dialog
    await expect(page.locator('[data-testid="delete-confirmation"]')).toBeVisible();
    await expect(page.locator('[data-testid="delete-confirmation"]'))
      .toContainText('Are you sure you want to delete this document?');
    
    // Confirm deletion
    await page.click('[data-testid="confirm-delete"]');
    
    // Should show success message
    await expect(page.locator('[data-testid="delete-success"]')).toBeVisible();
    
    // Document should be removed from list
    await expect(page.locator('[data-testid="document-item"]')).toHaveCount(expect.any(Number));
  });

  test('should create and manage folders', async ({ page }) => {
    // Create new folder
    await page.click('[data-testid="create-folder"]');
    
    // Should show create folder dialog
    await expect(page.locator('[data-testid="create-folder-dialog"]')).toBeVisible();
    
    await page.fill('[data-testid="folder-name"]', 'Test Folder');
    await page.fill('[data-testid="folder-description"]', 'Test folder description');
    await page.click('[data-testid="create-folder-button"]');
    
    // Should create folder
    await expect(page.locator('[data-testid="folder-created"]')).toBeVisible();
    
    // Should appear in folder list
    await expect(page.locator('[data-testid="folder-item"]'))
      .toContainText('Test Folder');
    
    // Navigate into folder
    await page.click('[data-testid="folder-item"]');
    
    // Should show empty folder
    await expect(page.locator('[data-testid="empty-folder"]')).toBeVisible();
    await expect(page.locator('[data-testid="breadcrumb"]'))
      .toContainText('Test Folder');
  });

  test('should move documents to folders', async ({ page }) => {
    // Select document
    await page.check('[data-testid="document-checkbox"]:nth-child(1)');
    
    // Move to folder
    await page.click('[data-testid="move-to-folder"]');
    
    // Should show folder selection
    await expect(page.locator('[data-testid="folder-selection"]')).toBeVisible();
    
    await page.click('[data-testid="select-folder"]');
    await page.click('[data-testid="confirm-move"]');
    
    // Should show success
    await expect(page.locator('[data-testid="move-success"]')).toBeVisible();
    
    // Document should be moved
    await expect(page.locator('[data-testid="document-item"]')).toHaveCount(expect.any(Number));
  });

  test('should handle pagination', async ({ page }) => {
    // Should show pagination if many documents
    const documentCount = await page.locator('[data-testid="document-item"]').count();
    
    if (documentCount >= 20) { // Assuming 20 items per page
      await expect(page.locator('[data-testid="pagination"]')).toBeVisible();
      
      // Navigate to next page
      await page.click('[data-testid="next-page"]');
      
      // Should show different documents
      await expect(page.locator('[data-testid="page-indicator"]'))
        .toContainText('Page 2');
      
      // Navigate back
      await page.click('[data-testid="prev-page"]');
      
      await expect(page.locator('[data-testid="page-indicator"]'))
        .toContainText('Page 1');
    }
  });

  test('should export document list', async ({ page }) => {
    // Export documents
    await page.click('[data-testid="export-list"]');
    
    // Should show export options
    await expect(page.locator('[data-testid="export-options"]')).toBeVisible();
    
    // Select CSV export
    await page.click('[data-testid="export-csv"]');
    
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="confirm-export"]');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toContain('documents');
    expect(download.suggestedFilename()).toContain('.csv');
  });

  test('should show document analytics', async ({ page }) => {
    // Navigate to analytics tab
    await page.click('[data-testid="analytics-tab"]');
    
    // Should show document statistics
    await expect(page.locator('[data-testid="document-stats"]')).toBeVisible();
    await expect(page.locator('[data-testid="total-documents"]')).toBeVisible();
    await expect(page.locator('[data-testid="total-size"]')).toBeVisible();
    await expect(page.locator('[data-testid="document-types-chart"]')).toBeVisible();
    
    // Should show recent activity
    await expect(page.locator('[data-testid="recent-activity"]')).toBeVisible();
    
    // Should show popular documents
    await expect(page.locator('[data-testid="popular-documents"]')).toBeVisible();
  });
});