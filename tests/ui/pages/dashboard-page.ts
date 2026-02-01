import { Page, expect } from '@playwright/test';
import { BasePage } from './base-page';

export class DashboardPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  // Page elements
  get elements() {
    return {
      welcomeMessage: this.page.locator('[data-testid="welcome-message"]'),
      statsCards: this.page.locator('[data-testid="stats-card"]'),
      recentDocuments: this.page.locator('[data-testid="recent-documents"]'),
      recentConversations: this.page.locator('[data-testid="recent-conversations"]'),
      quickActions: this.page.locator('[data-testid="quick-actions"]'),
      systemStatus: this.page.locator('[data-testid="system-status"]'),
      userProfile: this.page.locator('[data-testid="user-profile"]'),
      navigationMenu: this.page.locator('[data-testid="navigation-menu"]'),
      searchBar: this.page.locator('[data-testid="global-search"]'),
      notificationCenter: this.page.locator('[data-testid="notifications"]'),
      chatButton: this.page.locator('[data-testid="quick-chat"]'),
      uploadButton: this.page.locator('[data-testid="quick-upload"]'),
      settingsButton: this.page.locator('[data-testid="settings"]'),
      logoutButton: this.page.locator('[data-testid="logout"]'),
      activityFeed: this.page.locator('[data-testid="activity-feed"]'),
      performanceChart: this.page.locator('[data-testid="performance-chart"]'),
    };
  }

  async goto(): Promise<void> {
    await this.page.goto('/dashboard');
    await this.verifyPageLoaded();
  }

  async verifyPageLoaded(): Promise<void> {
    await expect(this.elements.welcomeMessage).toBeVisible();
    await expect(this.elements.statsCards.first()).toBeVisible();
    await expect(this.elements.navigationMenu).toBeVisible();
    await this.verifyTitle('Dashboard - Enterprise RAG Chatbot');
  }

  /**
   * Get dashboard statistics
   */
  async getDashboardStats(): Promise<Record<string, string>> {
    const statsCards = this.elements.statsCards;
    const count = await statsCards.count();
    const stats: Record<string, string> = {};
    
    for (let i = 0; i < count; i++) {
      const card = statsCards.nth(i);
      const label = await card.locator('[data-testid="stat-label"]').textContent() || '';
      const value = await card.locator('[data-testid="stat-value"]').textContent() || '';
      
      if (label && value) {
        stats[label] = value;
      }
    }
    
    return stats;
  }

  /**
   * Verify system status indicators
   */
  async verifySystemStatus(): Promise<void> {
    await expect(this.elements.systemStatus).toBeVisible();
    
    const status = await this.elements.systemStatus.getAttribute('data-status');
    expect(['healthy', 'warning', 'error']).toContain(status);
    
    // Verify individual service statuses
    const services = ['API', 'Vector DB', 'LLM Provider', 'Memory Store'];
    
    for (const service of services) {
      const serviceStatus = this.page.locator(`[data-testid="service-${service.toLowerCase().replace(/\s/g, '-')}"]`);
      if (await serviceStatus.isVisible()) {
        const status = await serviceStatus.getAttribute('data-status');
        expect(['online', 'offline', 'degraded']).toContain(status);
      }
    }
  }

  /**
   * Navigate to chat from dashboard
   */
  async navigateToChat(): Promise<void> {
    await this.elements.chatButton.click();
    await this.page.waitForURL('/chat');
  }

  /**
   * Navigate to documents from dashboard
   */
  async navigateToDocuments(): Promise<void> {
    await this.elements.uploadButton.click();
    await this.page.waitForURL('/documents');
  }

  /**
   * Get recent documents list
   */
  async getRecentDocuments(): Promise<Array<{ name: string; date: string; status: string }>> {
    const container = this.elements.recentDocuments;
    const items = container.locator('[data-testid="recent-document-item"]');
    const count = await items.count();
    const documents = [];
    
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      const name = await item.locator('[data-testid="document-name"]').textContent() || '';
      const date = await item.locator('[data-testid="document-date"]').textContent() || '';
      const status = await item.locator('[data-testid="document-status"]').textContent() || '';
      
      documents.push({ name, date, status });
    }
    
    return documents;
  }

  /**
   * Get recent conversations list
   */
  async getRecentConversations(): Promise<Array<{ title: string; date: string; messageCount: string }>> {
    const container = this.elements.recentConversations;
    const items = container.locator('[data-testid="recent-conversation-item"]');
    const count = await items.count();
    const conversations = [];
    
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      const title = await item.locator('[data-testid="conversation-title"]').textContent() || '';
      const date = await item.locator('[data-testid="conversation-date"]').textContent() || '';
      const messageCount = await item.locator('[data-testid="message-count"]').textContent() || '';
      
      conversations.push({ title, date, messageCount });
    }
    
    return conversations;
  }

  /**
   * Perform global search
   */
  async performGlobalSearch(query: string): Promise<void> {
    await this.fillFormField('[data-testid="global-search"]', query);
    await this.page.keyboard.press('Enter');
    
    // Wait for search results page or modal
    await this.page.waitForSelector('[data-testid="search-results"]', { timeout: 10000 });
  }

  /**
   * Check notifications
   */
  async checkNotifications(): Promise<Array<{ title: string; message: string; type: string; time: string }>> {
    await this.elements.notificationCenter.click();
    
    // Wait for notifications panel to open
    await this.page.waitForSelector('[data-testid="notification-panel"]');
    
    const items = this.page.locator('[data-testid="notification-item"]');
    const count = await items.count();
    const notifications = [];
    
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      const title = await item.locator('[data-testid="notification-title"]').textContent() || '';
      const message = await item.locator('[data-testid="notification-message"]').textContent() || '';
      const type = await item.getAttribute('data-type') || '';
      const time = await item.locator('[data-testid="notification-time"]').textContent() || '';
      
      notifications.push({ title, message, type, time });
    }
    
    return notifications;
  }

  /**
   * Access user profile
   */
  async accessUserProfile(): Promise<void> {
    await this.elements.userProfile.click();
    
    // Wait for profile modal or page
    await this.page.waitForSelector('[data-testid="user-profile-modal"]');
  }

  /**
   * Logout from dashboard
   */
  async logout(): Promise<void> {
    await this.elements.logoutButton.click();
    
    // Wait for logout confirmation or redirect
    await this.page.waitForURL('/login');
  }

  /**
   * Get activity feed items
   */
  async getActivityFeed(): Promise<Array<{ action: string; user: string; time: string; details: string }>> {
    const container = this.elements.activityFeed;
    const items = container.locator('[data-testid="activity-item"]');
    const count = await items.count();
    const activities = [];
    
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      const action = await item.locator('[data-testid="activity-action"]').textContent() || '';
      const user = await item.locator('[data-testid="activity-user"]').textContent() || '';
      const time = await item.locator('[data-testid="activity-time"]').textContent() || '';
      const details = await item.locator('[data-testid="activity-details"]').textContent() || '';
      
      activities.push({ action, user, time, details });
    }
    
    return activities;
  }

  /**
   * Verify performance chart is loaded
   */
  async verifyPerformanceChart(): Promise<void> {
    await expect(this.elements.performanceChart).toBeVisible();
    
    // Check if chart has loaded (look for SVG or canvas elements)
    const chartContent = this.elements.performanceChart.locator('svg, canvas, .chart-content');
    await expect(chartContent).toBeVisible();
  }

  /**
   * Test responsive dashboard layout
   */
  async testResponsiveDashboard(): Promise<void> {
    // Test mobile layout
    await this.verifyResponsiveLayout('mobile');
    await expect(this.elements.navigationMenu).toBeVisible();
    
    // Test tablet layout
    await this.verifyResponsiveLayout('tablet');
    await expect(this.elements.statsCards.first()).toBeVisible();
    
    // Test desktop layout
    await this.verifyResponsiveLayout('desktop');
    await expect(this.elements.recentDocuments).toBeVisible();
    await expect(this.elements.recentConversations).toBeVisible();
  }

  /**
   * Verify quick actions functionality
   */
  async verifyQuickActions(): Promise<void> {
    const quickActions = this.elements.quickActions.locator('[data-testid="quick-action"]');
    const count = await quickActions.count();
    
    expect(count).toBeGreaterThan(0);
    
    // Test each quick action
    for (let i = 0; i < count; i++) {
      const action = quickActions.nth(i);
      await expect(action).toBeVisible();
      await expect(action).toBeEnabled();
    }
  }

  /**
   * Test dashboard data refresh
   */
  async testDataRefresh(): Promise<void> {
    // Get initial stats
    const initialStats = await this.getDashboardStats();
    
    // Trigger refresh (if refresh button exists)
    const refreshButton = this.page.locator('[data-testid="refresh-dashboard"]');
    
    if (await refreshButton.isVisible()) {
      await refreshButton.click();
      await this.waitForPageLoad();
      
      // Verify stats updated (or at least page refreshed)
      const updatedStats = await this.getDashboardStats();
      
      // Stats structure should remain the same
      expect(Object.keys(updatedStats)).toEqual(Object.keys(initialStats));
    }
  }

  /**
   * Verify user permissions on dashboard
   */
  async verifyUserPermissions(expectedRole: 'admin' | 'user' | 'viewer'): Promise<void> {
    const userInfo = await this.elements.userProfile.textContent();
    expect(userInfo).toContain(expectedRole);
    
    // Verify role-specific elements
    switch (expectedRole) {
      case 'admin':
        await expect(this.page.locator('[data-testid="admin-panel-link"]')).toBeVisible();
        break;
      case 'user':
        await expect(this.elements.uploadButton).toBeVisible();
        break;
      case 'viewer':
        await expect(this.elements.uploadButton).not.toBeVisible();
        break;
    }
  }

  /**
   * Complete dashboard verification
   */
  async performDashboardVerification(): Promise<void> {
    // Verify page loads correctly
    await this.verifyPageLoaded();
    
    // Verify main dashboard components
    await this.verifySystemStatus();
    await this.verifyPerformanceChart();
    await this.verifyQuickActions();
    
    // Verify data is present
    const stats = await this.getDashboardStats();
    expect(Object.keys(stats).length).toBeGreaterThan(0);
    
    const recentDocs = await this.getRecentDocuments();
    // Recent docs might be empty in a fresh system
    
    // Verify no errors
    await this.verifyNoErrors();
    
    // Test navigation
    await this.navigateToChat();
    await this.page.goBack();
    await this.verifyPageLoaded();
  }
}