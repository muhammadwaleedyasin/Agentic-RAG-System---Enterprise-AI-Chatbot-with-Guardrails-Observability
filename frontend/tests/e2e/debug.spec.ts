import { test, expect } from '@playwright/test';

test.describe('Debug Tests', () => {
  test('should load homepage and check basic elements', async ({ page }) => {
    await page.goto('/');
    
    // Take a screenshot to see what's actually loaded
    await page.screenshot({ path: 'debug-homepage.png', fullPage: true });
    
    // Check if page loads at all
    await expect(page).toHaveURL(/localhost:3000/);
    
    // Log page content for debugging
    const title = await page.title();
    console.log('Page title:', title);
    
    const bodyText = await page.locator('body').textContent();
    console.log('Body text (first 200 chars):', bodyText?.substring(0, 200));
    
    // Check for any error messages
    const errorElements = await page.locator('[role="alert"]').count();
    console.log('Error elements found:', errorElements);
    
    // Check for loading indicators
    const loadingElements = await page.locator('[aria-label*="loading"], [aria-label*="Loading"]').count();
    console.log('Loading elements found:', loadingElements);
  });

  test('should load login page directly', async ({ page }) => {
    await page.goto('/login');
    
    // Take a screenshot
    await page.screenshot({ path: 'debug-login.png', fullPage: true });
    
    // Check basic login page elements
    await expect(page).toHaveURL(/.*login/);
    
    // Log what we can see
    const title = await page.title();
    console.log('Login page title:', title);
    
    const bodyText = await page.locator('body').textContent();
    console.log('Login body text (first 200 chars):', bodyText?.substring(0, 200));
    
    // Check for form elements
    const formCount = await page.locator('form').count();
    console.log('Forms found:', formCount);
    
    const inputCount = await page.locator('input').count();
    console.log('Inputs found:', inputCount);
    
    const buttonCount = await page.locator('button').count();
    console.log('Buttons found:', buttonCount);
  });
});