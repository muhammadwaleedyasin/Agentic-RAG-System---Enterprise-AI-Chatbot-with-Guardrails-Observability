import { test, expect } from '@playwright/test';

/**
 * Document Processing Tests
 * Tests document processing pipeline, chunking, embedding, and indexing
 */

test.describe('Document Processing', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin to see processing details
    await page.goto('/login');
    await page.fill('[data-testid="username-input"]', 'admin_test');
    await page.fill('[data-testid="password-input"]', 'AdminTest123!');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
    await page.goto('/admin/processing');
  });

  test('should display processing queue', async ({ page }) => {
    // Verify processing dashboard
    await expect(page.locator('[data-testid="processing-dashboard"]')).toBeVisible();
    await expect(page.locator('[data-testid="processing-queue"]')).toBeVisible();
    await expect(page.locator('[data-testid="processing-stats"]')).toBeVisible();
    
    // Verify queue statistics
    await expect(page.locator('[data-testid="queue-pending"]')).toBeVisible();
    await expect(page.locator('[data-testid="queue-processing"]')).toBeVisible();
    await expect(page.locator('[data-testid="queue-completed"]')).toBeVisible();
    await expect(page.locator('[data-testid="queue-failed"]')).toBeVisible();
  });

  test('should track document processing stages', async ({ page }) => {
    // Upload a document to track processing
    await page.goto('/documents');
    const filePath = path.join(__dirname, '../fixtures/test-document.pdf');
    await page.setInputFiles('[data-testid="file-input"]', filePath);
    await page.click('[data-testid="start-upload"]');
    
    // Go to processing dashboard
    await page.goto('/admin/processing');
    
    // Should show document in queue
    await expect(page.locator('[data-testid="processing-item"]').first()).toBeVisible();
    
    // Check processing stages
    const processingItem = page.locator('[data-testid="processing-item"]').first();
    
    // Stage 1: Document parsing
    await expect(processingItem.locator('[data-testid="stage-parsing"]')).toBeVisible();
    await expect(processingItem.locator('[data-testid="stage-parsing"]'))
      .toHaveClass(/in-progress|completed/);
    
    // Stage 2: Text extraction
    await expect(processingItem.locator('[data-testid="stage-extraction"]')).toBeVisible();
    
    // Stage 3: Chunking
    await expect(processingItem.locator('[data-testid="stage-chunking"]')).toBeVisible();
    
    // Stage 4: Embedding generation
    await expect(processingItem.locator('[data-testid="stage-embedding"]')).toBeVisible();
    
    // Stage 5: Vector indexing
    await expect(processingItem.locator('[data-testid="stage-indexing"]')).toBeVisible();
  });

  test('should show processing progress', async ({ page }) => {
    // Find a document in processing
    const processingItem = page.locator('[data-testid="processing-item"]').first();
    
    if (await processingItem.isVisible()) {
      // Should show overall progress
      await expect(processingItem.locator('[data-testid="overall-progress"]')).toBeVisible();
      
      // Should show stage-specific progress
      await expect(processingItem.locator('[data-testid="stage-progress"]')).toBeVisible();
      
      // Should show estimated time remaining
      await expect(processingItem.locator('[data-testid="eta"]')).toBeVisible();
      
      // Should show processing speed
      await expect(processingItem.locator('[data-testid="processing-speed"]')).toBeVisible();
    }
  });

  test('should handle processing errors', async ({ page }) => {
    // Look for failed processing items
    const failedItems = page.locator('[data-testid="processing-failed"]');
    
    if (await failedItems.count() > 0) {
      const failedItem = failedItems.first();
      
      // Should show error status
      await expect(failedItem.locator('[data-testid="error-status"]')).toBeVisible();
      
      // Should show error details
      await failedItem.click();
      await expect(page.locator('[data-testid="error-details"]')).toBeVisible();
      
      // Should show retry option
      await expect(page.locator('[data-testid="retry-processing"]')).toBeVisible();
      
      // Should show error log
      await expect(page.locator('[data-testid="error-log"]')).toBeVisible();
    }
  });

  test('should allow manual processing retry', async ({ page }) => {
    // Find failed processing item
    const failedItem = page.locator('[data-testid="processing-failed"]').first();
    
    if (await failedItem.isVisible()) {
      // Retry processing
      await failedItem.click();
      await page.click('[data-testid="retry-processing"]');
      
      // Should confirm retry
      await expect(page.locator('[data-testid="retry-confirmation"]')).toBeVisible();
      await page.click('[data-testid="confirm-retry"]');
      
      // Should move back to processing queue
      await expect(page.locator('[data-testid="retry-success"]')).toBeVisible();
    }
  });

  test('should configure processing settings', async ({ page }) => {
    // Navigate to processing settings
    await page.click('[data-testid="processing-settings"]');
    
    // Should show configuration options
    await expect(page.locator('[data-testid="chunking-settings"]')).toBeVisible();
    await expect(page.locator('[data-testid="embedding-settings"]')).toBeVisible();
    await expect(page.locator('[data-testid="processing-limits"]')).toBeVisible();
    
    // Configure chunk size
    await page.fill('[data-testid="chunk-size"]', '1000');
    await page.fill('[data-testid="chunk-overlap"]', '200');
    
    // Configure concurrent processing
    await page.fill('[data-testid="max-concurrent"]', '3');
    
    // Configure timeout settings
    await page.fill('[data-testid="processing-timeout"]', '300');
    
    // Save settings
    await page.click('[data-testid="save-settings"]');
    
    // Should show success
    await expect(page.locator('[data-testid="settings-saved"]')).toBeVisible();
  });

  test('should show chunking details', async ({ page }) => {
    // Find a processed document
    const processedItem = page.locator('[data-testid="processing-completed"]').first();
    
    if (await processedItem.isVisible()) {
      // View chunking details
      await processedItem.click();
      await page.click('[data-testid="view-chunks"]');
      
      // Should show chunk information
      await expect(page.locator('[data-testid="chunk-list"]')).toBeVisible();
      await expect(page.locator('[data-testid="chunk-count"]')).toBeVisible();
      
      // Each chunk should show details
      const firstChunk = page.locator('[data-testid="chunk-item"]').first();
      await expect(firstChunk.locator('[data-testid="chunk-text"]')).toBeVisible();
      await expect(firstChunk.locator('[data-testid="chunk-size"]')).toBeVisible();
      await expect(firstChunk.locator('[data-testid="chunk-tokens"]')).toBeVisible();
      
      // Should be able to view full chunk
      await firstChunk.click();
      await expect(page.locator('[data-testid="chunk-details"]')).toBeVisible();
    }
  });

  test('should show embedding status', async ({ page }) => {
    // View embedding dashboard
    await page.click('[data-testid="embedding-tab"]');
    
    // Should show embedding statistics
    await expect(page.locator('[data-testid="embedding-stats"]')).toBeVisible();
    await expect(page.locator('[data-testid="total-embeddings"]')).toBeVisible();
    await expect(page.locator('[data-testid="embedding-model"]')).toBeVisible();
    await expect(page.locator('[data-testid="embedding-dimensions"]')).toBeVisible();
    
    // Should show recent embedding activity
    await expect(page.locator('[data-testid="embedding-activity"]')).toBeVisible();
  });

  test('should handle batch processing', async ({ page }) => {
    // Navigate to batch processing
    await page.click('[data-testid="batch-processing"]');
    
    // Should show batch options
    await expect(page.locator('[data-testid="batch-options"]')).toBeVisible();
    
    // Select documents for batch processing
    await page.check('[data-testid="select-pending"]');
    
    // Start batch processing
    await page.click('[data-testid="start-batch"]');
    
    // Should show batch progress
    await expect(page.locator('[data-testid="batch-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="batch-eta"]')).toBeVisible();
    
    // Should be able to pause/resume
    await expect(page.locator('[data-testid="pause-batch"]')).toBeVisible();
  });

  test('should monitor processing performance', async ({ page }) => {
    // Navigate to performance monitoring
    await page.click('[data-testid="performance-tab"]');
    
    // Should show performance metrics
    await expect(page.locator('[data-testid="performance-metrics"]')).toBeVisible();
    await expect(page.locator('[data-testid="processing-throughput"]')).toBeVisible();
    await expect(page.locator('[data-testid="average-processing-time"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-rate"]')).toBeVisible();
    
    // Should show resource usage
    await expect(page.locator('[data-testid="cpu-usage"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-usage"]')).toBeVisible();
    
    // Should show processing trends
    await expect(page.locator('[data-testid="performance-chart"]')).toBeVisible();
  });

  test('should handle processing queue management', async ({ page }) => {
    // Should be able to pause processing
    await page.click('[data-testid="pause-processing"]');
    await expect(page.locator('[data-testid="processing-paused"]')).toBeVisible();
    
    // Should be able to resume processing
    await page.click('[data-testid="resume-processing"]');
    await expect(page.locator('[data-testid="processing-resumed"]')).toBeVisible();
    
    // Should be able to clear failed items
    await page.click('[data-testid="clear-failed"]');
    await expect(page.locator('[data-testid="clear-failed-confirmation"]')).toBeVisible();
    await page.click('[data-testid="confirm-clear-failed"]');
    
    // Should be able to prioritize items
    const queueItem = page.locator('[data-testid="processing-item"]').first();
    await queueItem.click();
    await page.click('[data-testid="prioritize-item"]');
    await expect(page.locator('[data-testid="item-prioritized"]')).toBeVisible();
  });

  test('should validate processed documents', async ({ page }) => {
    // Find completed processing item
    const completedItem = page.locator('[data-testid="processing-completed"]').first();
    
    if (await completedItem.isVisible()) {
      // Run validation
      await completedItem.click();
      await page.click('[data-testid="validate-processing"]');
      
      // Should show validation results
      await expect(page.locator('[data-testid="validation-results"]')).toBeVisible();
      
      // Should check various aspects
      await expect(page.locator('[data-testid="text-extraction-quality"]')).toBeVisible();
      await expect(page.locator('[data-testid="chunk-quality"]')).toBeVisible();
      await expect(page.locator('[data-testid="embedding-quality"]')).toBeVisible();
      await expect(page.locator('[data-testid="indexing-status"]')).toBeVisible();
      
      // Should show overall quality score
      await expect(page.locator('[data-testid="quality-score"]')).toBeVisible();
    }
  });

  test('should export processing reports', async ({ page }) => {
    // Navigate to reports
    await page.click('[data-testid="processing-reports"]');
    
    // Generate processing report
    await page.selectOption('[data-testid="report-type"]', 'processing-summary');
    await page.fill('[data-testid="report-date-from"]', '2024-01-01');
    await page.fill('[data-testid="report-date-to"]', '2024-12-31');
    
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="generate-report"]');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toContain('processing-report');
    expect(download.suggestedFilename()).toMatch(/\.(pdf|csv|xlsx)$/);
  });

  test('should handle processing webhooks', async ({ page }) => {
    // Navigate to webhook settings
    await page.click('[data-testid="webhook-settings"]');
    
    // Should show webhook configuration
    await expect(page.locator('[data-testid="webhook-config"]')).toBeVisible();
    
    // Add webhook URL
    await page.fill('[data-testid="webhook-url"]', 'https://example.com/webhook');
    
    // Select events
    await page.check('[data-testid="webhook-processing-complete"]');
    await page.check('[data-testid="webhook-processing-failed"]');
    
    // Save webhook
    await page.click('[data-testid="save-webhook"]');
    
    // Should show success
    await expect(page.locator('[data-testid="webhook-saved"]')).toBeVisible();
    
    // Test webhook
    await page.click('[data-testid="test-webhook"]');
    await expect(page.locator('[data-testid="webhook-test-result"]')).toBeVisible();
  });
});