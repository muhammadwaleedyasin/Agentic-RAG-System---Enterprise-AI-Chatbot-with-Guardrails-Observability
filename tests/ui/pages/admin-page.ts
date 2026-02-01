import { Page, expect } from '@playwright/test';
import { BasePage } from './base-page';

export class AdminPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Page elements
  get elements() {
    return {
      adminNavigation: this.page.locator('[data-testid="admin-navigation"]'),
      userManagement: this.page.locator('[data-testid="user-management"]'),
      systemSettings: this.page.locator('[data-testid="system-settings"]'),
      documentManagement: this.page.locator('[data-testid="document-management"]'),
      analyticsPanel: this.page.locator('[data-testid="analytics-panel"]'),
      auditLogs: this.page.locator('[data-testid="audit-logs"]'),
      
      // User Management
      usersTable: this.page.locator('[data-testid="users-table"]'),
      userRows: this.page.locator('[data-testid="user-row"]'),
      createUserButton: this.page.locator('[data-testid="create-user"]'),
      editUserButton: this.page.locator('[data-testid="edit-user"]'),
      deleteUserButton: this.page.locator('[data-testid="delete-user"]'),
      userForm: this.page.locator('[data-testid="user-form"]'),
      
      // System Settings
      llmProviderSettings: this.page.locator('[data-testid="llm-provider-settings"]'),
      vectorDbSettings: this.page.locator('[data-testid="vector-db-settings"]'),
      securitySettings: this.page.locator('[data-testid="security-settings"]'),
      cacheSettings: this.page.locator('[data-testid="cache-settings"]'),
      saveSettingsButton: this.page.locator('[data-testid="save-settings"]'),
      
      // Document Management
      documentsTable: this.page.locator('[data-testid="admin-documents-table"]'),
      documentRows: this.page.locator('[data-testid="admin-document-row"]'),
      bulkDeleteButton: this.page.locator('[data-testid="bulk-delete-documents"]'),
      reindexButton: this.page.locator('[data-testid="reindex-documents"]'),
      
      // Analytics
      analyticsCharts: this.page.locator('[data-testid="analytics-chart"]'),
      usageStats: this.page.locator('[data-testid="usage-stats"]'),
      performanceMetrics: this.page.locator('[data-testid="performance-metrics"]'),
      
      // Audit Logs
      logsTable: this.page.locator('[data-testid="audit-logs-table"]'),
      logRows: this.page.locator('[data-testid="log-row"]'),
      logFilters: this.page.locator('[data-testid="log-filters"]'),
      exportLogsButton: this.page.locator('[data-testid="export-logs"]'),
    };
  }

  async goto(): Promise<void> {
    await this.page.goto('/admin');
    await this.verifyPageLoaded();
  }

  async verifyPageLoaded(): Promise<void> {
    await expect(this.elements.adminNavigation).toBeVisible();
    await this.verifyTitle('Admin Panel - Enterprise RAG Chatbot');
  }

  /**
   * Navigate to admin section
   */
  async navigateToSection(section: 'users' | 'settings' | 'documents' | 'analytics' | 'logs'): Promise<void> {
    const navItem = this.page.locator(`[data-testid="admin-nav-${section}"]`);
    await navItem.click();
    await this.waitForPageLoad();
  }

  // User Management Methods
  /**
   * Create a new user
   */
  async createUser(userData: {
    username: string;
    email: string;
    password: string;
    role: 'admin' | 'user' | 'viewer';
    firstName?: string;
    lastName?: string;
  }): Promise<void> {
    await this.navigateToSection('users');
    await this.elements.createUserButton.click();
    
    // Fill user form
    await this.fillFormField('[data-testid="username-input"]', userData.username);
    await this.fillFormField('[data-testid="email-input"]', userData.email);
    await this.fillFormField('[data-testid="password-input"]', userData.password);
    
    if (userData.firstName) {
      await this.fillFormField('[data-testid="first-name-input"]', userData.firstName);
    }
    
    if (userData.lastName) {
      await this.fillFormField('[data-testid="last-name-input"]', userData.lastName);
    }
    
    // Select role
    const roleSelect = this.page.locator('[data-testid="role-select"]');
    await roleSelect.selectOption(userData.role);
    
    // Submit form
    const submitButton = this.page.locator('[data-testid="submit-user"]');
    await submitButton.click();
    
    // Wait for success
    await this.verifyToast('User created successfully');
  }

  /**
   * Get list of users
   */
  async getUserList(): Promise<Array<{ username: string; email: string; role: string; status: string }>> {
    await this.navigateToSection('users');
    
    const userRows = this.elements.userRows;
    const count = await userRows.count();
    const users = [];
    
    for (let i = 0; i < count; i++) {
      const row = userRows.nth(i);
      const username = await row.locator('[data-testid="user-username"]').textContent() || '';
      const email = await row.locator('[data-testid="user-email"]').textContent() || '';
      const role = await row.locator('[data-testid="user-role"]').textContent() || '';
      const status = await row.locator('[data-testid="user-status"]').textContent() || '';
      
      users.push({ username, email, role, status });
    }
    
    return users;
  }

  /**
   * Edit user
   */
  async editUser(username: string, updates: Partial<{
    email: string;
    role: string;
    status: string;
  }>): Promise<void> {
    await this.navigateToSection('users');
    
    const userRow = this.page.locator(`[data-testid="user-row"]:has-text("${username}")`);
    const editButton = userRow.locator('[data-testid="edit-user"]');
    await editButton.click();
    
    // Apply updates
    if (updates.email) {
      await this.fillFormField('[data-testid="email-input"]', updates.email);
    }
    
    if (updates.role) {
      const roleSelect = this.page.locator('[data-testid="role-select"]');
      await roleSelect.selectOption(updates.role);
    }
    
    if (updates.status) {
      const statusSelect = this.page.locator('[data-testid="status-select"]');
      await statusSelect.selectOption(updates.status);
    }
    
    // Submit changes
    const saveButton = this.page.locator('[data-testid="save-user"]');
    await saveButton.click();
    
    await this.verifyToast('User updated successfully');
  }

  /**
   * Delete user
   */
  async deleteUser(username: string): Promise<void> {
    await this.navigateToSection('users');
    
    const userRow = this.page.locator(`[data-testid="user-row"]:has-text("${username}")`);
    const deleteButton = userRow.locator('[data-testid="delete-user"]');
    await deleteButton.click();
    
    // Confirm deletion
    await this.confirmAction(true);
    
    await this.verifyToast('User deleted successfully');
  }

  // System Settings Methods
  /**
   * Update LLM provider settings
   */
  async updateLLMSettings(settings: {
    provider: string;
    apiKey?: string;
    baseUrl?: string;
    model?: string;
  }): Promise<void> {
    await this.navigateToSection('settings');
    
    const llmSection = this.elements.llmProviderSettings;
    
    // Provider selection
    const providerSelect = llmSection.locator('[data-testid="llm-provider-select"]');
    await providerSelect.selectOption(settings.provider);
    
    // API Key
    if (settings.apiKey) {
      await this.fillFormField('[data-testid="llm-api-key"]', settings.apiKey);
    }
    
    // Base URL
    if (settings.baseUrl) {
      await this.fillFormField('[data-testid="llm-base-url"]', settings.baseUrl);
    }
    
    // Model
    if (settings.model) {
      await this.fillFormField('[data-testid="llm-model"]', settings.model);
    }
    
    // Save settings
    await this.elements.saveSettingsButton.click();
    await this.verifyToast('LLM settings updated successfully');
  }

  /**
   * Update security settings
   */
  async updateSecuritySettings(settings: {
    enableAuth?: boolean;
    sessionTimeout?: number;
    maxLoginAttempts?: number;
    enableAuditLogging?: boolean;
  }): Promise<void> {
    await this.navigateToSection('settings');
    
    const securitySection = this.elements.securitySettings;
    
    if (settings.enableAuth !== undefined) {
      const authToggle = securitySection.locator('[data-testid="enable-auth-toggle"]');
      if (settings.enableAuth) {
        await authToggle.check();
      } else {
        await authToggle.uncheck();
      }
    }
    
    if (settings.sessionTimeout) {
      await this.fillFormField('[data-testid="session-timeout"]', settings.sessionTimeout.toString());
    }
    
    if (settings.maxLoginAttempts) {
      await this.fillFormField('[data-testid="max-login-attempts"]', settings.maxLoginAttempts.toString());
    }
    
    if (settings.enableAuditLogging !== undefined) {
      const auditToggle = securitySection.locator('[data-testid="enable-audit-toggle"]');
      if (settings.enableAuditLogging) {
        await auditToggle.check();
      } else {
        await auditToggle.uncheck();
      }
    }
    
    await this.elements.saveSettingsButton.click();
    await this.verifyToast('Security settings updated successfully');
  }

  // Document Management Methods
  /**
   * Get admin documents list
   */
  async getAdminDocumentsList(): Promise<Array<{
    name: string;
    owner: string;
    size: string;
    status: string;
    uploadDate: string;
  }>> {
    await this.navigateToSection('documents');
    
    const documentRows = this.elements.documentRows;
    const count = await documentRows.count();
    const documents = [];
    
    for (let i = 0; i < count; i++) {
      const row = documentRows.nth(i);
      const name = await row.locator('[data-testid="doc-name"]').textContent() || '';
      const owner = await row.locator('[data-testid="doc-owner"]').textContent() || '';
      const size = await row.locator('[data-testid="doc-size"]').textContent() || '';
      const status = await row.locator('[data-testid="doc-status"]').textContent() || '';
      const uploadDate = await row.locator('[data-testid="doc-upload-date"]').textContent() || '';
      
      documents.push({ name, owner, size, status, uploadDate });
    }
    
    return documents;
  }

  /**
   * Reindex documents
   */
  async reindexDocuments(): Promise<void> {
    await this.navigateToSection('documents');
    await this.elements.reindexButton.click();
    
    // Confirm reindexing
    await this.confirmAction(true);
    
    await this.verifyToast('Document reindexing started');
  }

  // Analytics Methods
  /**
   * Get usage statistics
   */
  async getUsageStatistics(): Promise<Record<string, string>> {
    await this.navigateToSection('analytics');
    
    const statsContainer = this.elements.usageStats;
    const statItems = statsContainer.locator('[data-testid="usage-stat"]');
    const count = await statItems.count();
    const stats: Record<string, string> = {};
    
    for (let i = 0; i < count; i++) {
      const item = statItems.nth(i);
      const label = await item.locator('[data-testid="stat-label"]').textContent() || '';
      const value = await item.locator('[data-testid="stat-value"]').textContent() || '';
      
      if (label && value) {
        stats[label] = value;
      }
    }
    
    return stats;
  }

  /**
   * Verify analytics charts are loaded
   */
  async verifyAnalyticsCharts(): Promise<void> {
    await this.navigateToSection('analytics');
    
    const charts = this.elements.analyticsCharts;
    const count = await charts.count();
    
    expect(count).toBeGreaterThan(0);
    
    for (let i = 0; i < count; i++) {
      const chart = charts.nth(i);
      await expect(chart).toBeVisible();
      
      // Check if chart content is loaded
      const chartContent = chart.locator('svg, canvas, .chart-content');
      await expect(chartContent).toBeVisible();
    }
  }

  // Audit Logs Methods
  /**
   * Get audit logs
   */
  async getAuditLogs(limit = 10): Promise<Array<{
    timestamp: string;
    user: string;
    action: string;
    resource: string;
    details: string;
  }>> {
    await this.navigateToSection('logs');
    
    const logRows = this.elements.logRows;
    const count = Math.min(await logRows.count(), limit);
    const logs = [];
    
    for (let i = 0; i < count; i++) {
      const row = logRows.nth(i);
      const timestamp = await row.locator('[data-testid="log-timestamp"]').textContent() || '';
      const user = await row.locator('[data-testid="log-user"]').textContent() || '';
      const action = await row.locator('[data-testid="log-action"]').textContent() || '';
      const resource = await row.locator('[data-testid="log-resource"]').textContent() || '';
      const details = await row.locator('[data-testid="log-details"]').textContent() || '';
      
      logs.push({ timestamp, user, action, resource, details });
    }
    
    return logs;
  }

  /**
   * Filter audit logs
   */
  async filterAuditLogs(filters: {
    user?: string;
    action?: string;
    dateFrom?: string;
    dateTo?: string;
  }): Promise<void> {
    await this.navigateToSection('logs');
    
    const filtersSection = this.elements.logFilters;
    
    if (filters.user) {
      await this.fillFormField('[data-testid="filter-user"]', filters.user);
    }
    
    if (filters.action) {
      const actionSelect = filtersSection.locator('[data-testid="filter-action"]');
      await actionSelect.selectOption(filters.action);
    }
    
    if (filters.dateFrom) {
      await this.fillFormField('[data-testid="filter-date-from"]', filters.dateFrom);
    }
    
    if (filters.dateTo) {
      await this.fillFormField('[data-testid="filter-date-to"]', filters.dateTo);
    }
    
    // Apply filters
    const applyButton = filtersSection.locator('[data-testid="apply-filters"]');
    await applyButton.click();
    
    await this.waitForPageLoad();
  }

  /**
   * Export audit logs
   */
  async exportAuditLogs(format: 'csv' | 'json' = 'csv'): Promise<void> {
    await this.navigateToSection('logs');
    
    // Set export format
    const formatSelect = this.page.locator('[data-testid="export-format"]');
    await formatSelect.selectOption(format);
    
    // Start download
    const downloadPromise = this.page.waitForEvent('download');
    await this.elements.exportLogsButton.click();
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toContain(`audit-logs.${format}`);
  }

  /**
   * Perform comprehensive admin verification
   */
  async performAdminVerification(): Promise<void> {
    // Verify admin access
    await this.verifyPageLoaded();
    
    // Test user management
    const users = await this.getUserList();
    expect(users.length).toBeGreaterThan(0);
    
    // Test analytics
    await this.verifyAnalyticsCharts();
    const stats = await this.getUsageStatistics();
    expect(Object.keys(stats).length).toBeGreaterThan(0);
    
    // Test audit logs
    const logs = await this.getAuditLogs(5);
    // Logs might be empty in a fresh system
    
    // Verify no errors
    await this.verifyNoErrors();
  }
}