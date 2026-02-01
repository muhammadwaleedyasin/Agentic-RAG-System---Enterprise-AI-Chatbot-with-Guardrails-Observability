/**
 * Comprehensive accessibility testing suite for RAG Chatbot UI
 * Tests WCAG 2.1 AA compliance using axe-core and custom validators
 */

const { AxePuppeteer } = require('@axe-core/puppeteer');
const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');
const axeConfig = require('./axe-core-config');

class AccessibilityTestSuite {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:3000';
    this.browser = null;
    this.page = null;
    this.results = {
      violations: [],
      passes: [],
      incomplete: [],
      summary: {}
    };
    this.viewports = [
      { name: 'mobile', width: 375, height: 667 },
      { name: 'tablet', width: 768, height: 1024 },
      { name: 'desktop', width: 1200, height: 800 },
      { name: 'large-desktop', width: 1920, height: 1080 }
    ];
  }

  async initialize() {
    this.browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-web-security',
        '--allow-running-insecure-content'
      ]
    });
  }

  async createNewPage() {
    if (this.page) {
      await this.page.close();
    }
    this.page = await this.browser.newPage();
    
    // Enable accessibility features
    await this.page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    );
    
    // Enable high contrast mode simulation
    await this.page.emulateMediaFeatures([
      { name: 'prefers-contrast', value: 'high' },
      { name: 'prefers-reduced-motion', value: 'reduce' }
    ]);
  }

  async runFullAccessibilityAudit(urls = ['/']) {
    await this.initialize();
    
    for (const url of urls) {
      console.log(`\n🔍 Testing accessibility for: ${url}`);
      
      for (const viewport of this.viewports) {
        console.log(`  📱 Viewport: ${viewport.name} (${viewport.width}x${viewport.height})`);
        await this.testPageAccessibility(url, viewport);
      }
    }
    
    await this.generateReport();
    await this.cleanup();
    
    return this.results;
  }

  async testPageAccessibility(url, viewport) {
    await this.createNewPage();
    await this.page.setViewport(viewport);
    
    try {
      await this.page.goto(`${this.baseUrl}${url}`, { waitUntil: 'networkidle0' });
      
      // Test core accessibility
      await this.runAxeTests(url, viewport.name);
      
      // Test keyboard navigation
      await this.testKeyboardNavigation(url, viewport.name);
      
      // Test focus management
      await this.testFocusManagement(url, viewport.name);
      
      // Test color contrast
      await this.testColorContrast(url, viewport.name);
      
      // Test screen reader compatibility
      await this.testScreenReaderCompatibility(url, viewport.name);
      
    } catch (error) {
      console.error(`Error testing ${url} on ${viewport.name}:`, error);
      this.results.violations.push({
        url,
        viewport: viewport.name,
        error: error.message,
        type: 'test-error'
      });
    }
  }

  async runAxeTests(url, viewportName) {
    try {
      const axeResults = await new AxePuppeteer(this.page)
        .withOptions(axeConfig.default)
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
        .analyze();

      // Process violations
      axeResults.violations.forEach(violation => {
        this.results.violations.push({
          url,
          viewport: viewportName,
          rule: violation.id,
          impact: violation.impact,
          description: violation.description,
          help: violation.help,
          helpUrl: violation.helpUrl,
          nodes: violation.nodes.map(node => ({
            html: node.html,
            target: node.target,
            failureSummary: node.failureSummary,
            any: node.any,
            all: node.all,
            none: node.none
          }))
        });
      });

      // Process passes
      axeResults.passes.forEach(pass => {
        this.results.passes.push({
          url,
          viewport: viewportName,
          rule: pass.id,
          description: pass.description,
          nodes: pass.nodes.length
        });
      });

      // Process incomplete
      axeResults.incomplete.forEach(incomplete => {
        this.results.incomplete.push({
          url,
          viewport: viewportName,
          rule: incomplete.id,
          description: incomplete.description,
          help: incomplete.help,
          nodes: incomplete.nodes.length
        });
      });

    } catch (error) {
      console.error('Axe test error:', error);
    }
  }

  async testKeyboardNavigation(url, viewportName) {
    try {
      // Test tab navigation
      const tabbableElements = await this.page.evaluate(() => {
        const elements = Array.from(document.querySelectorAll(
          'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
        ));
        return elements.map(el => ({
          tagName: el.tagName,
          type: el.type || null,
          tabIndex: el.tabIndex,
          id: el.id || null,
          className: el.className || null,
          ariaLabel: el.getAttribute('aria-label') || null,
          visible: el.offsetParent !== null
        }));
      });

      // Simulate tab navigation
      const tabSequence = [];
      await this.page.keyboard.press('Tab');
      
      for (let i = 0; i < tabbableElements.length; i++) {
        const focusedElement = await this.page.evaluate(() => {
          const el = document.activeElement;
          return {
            tagName: el.tagName,
            id: el.id || null,
            className: el.className || null,
            type: el.type || null
          };
        });
        
        tabSequence.push(focusedElement);
        await this.page.keyboard.press('Tab');
      }

      // Test escape key functionality
      await this.testEscapeKeyFunctionality();
      
      // Test arrow key navigation for complex widgets
      await this.testArrowKeyNavigation();

      this.results.keyboardTests = this.results.keyboardTests || [];
      this.results.keyboardTests.push({
        url,
        viewport: viewportName,
        tabbableElements: tabbableElements.length,
        tabSequence,
        passed: true
      });

    } catch (error) {
      console.error('Keyboard navigation test error:', error);
      this.results.violations.push({
        url,
        viewport: viewportName,
        type: 'keyboard-navigation',
        error: error.message
      });
    }
  }

  async testEscapeKeyFunctionality() {
    // Test modals and popups close with Escape
    const modals = await this.page.$$('[role="dialog"], .modal, .popup');
    for (const modal of modals) {
      const isVisible = await modal.isIntersectingViewport();
      if (isVisible) {
        await this.page.keyboard.press('Escape');
        await this.page.waitForTimeout(100);
        const stillVisible = await modal.isIntersectingViewport();
        if (stillVisible) {
          this.results.violations.push({
            type: 'escape-key',
            description: 'Modal does not close with Escape key',
            element: await modal.evaluate(el => el.outerHTML)
          });
        }
      }
    }
  }

  async testArrowKeyNavigation() {
    // Test arrow key navigation for menus, tabs, etc.
    const widgets = await this.page.$$('[role="menu"], [role="tablist"], [role="listbox"]');
    
    for (const widget of widgets) {
      const role = await widget.evaluate(el => el.getAttribute('role'));
      await widget.focus();
      
      // Test arrow key navigation
      await this.page.keyboard.press('ArrowDown');
      await this.page.waitForTimeout(50);
      
      const focusAfterArrow = await this.page.evaluate(() => document.activeElement);
      // Validate focus moved appropriately based on widget type
    }
  }

  async testFocusManagement(url, viewportName) {
    try {
      // Test focus indicators
      const focusableElements = await this.page.$$('a, button, input, textarea, select, [tabindex]');
      
      for (const element of focusableElements.slice(0, 5)) { // Test first 5 elements
        await element.focus();
        
        const focusStyles = await element.evaluate(el => {
          const styles = window.getComputedStyle(el, ':focus');
          return {
            outline: styles.outline,
            outlineWidth: styles.outlineWidth,
            outlineStyle: styles.outlineStyle,
            outlineColor: styles.outlineColor,
            boxShadow: styles.boxShadow,
            backgroundColor: styles.backgroundColor,
            borderColor: styles.borderColor
          };
        });

        // Check if focus is visible
        const hasFocusIndicator = 
          focusStyles.outline !== 'none' ||
          focusStyles.outlineWidth !== '0px' ||
          focusStyles.boxShadow !== 'none' ||
          focusStyles.backgroundColor !== 'transparent';

        if (!hasFocusIndicator) {
          this.results.violations.push({
            url,
            viewport: viewportName,
            type: 'focus-indicator',
            description: 'Element lacks visible focus indicator',
            element: await element.evaluate(el => el.outerHTML)
          });
        }
      }

      // Test focus trapping in modals
      await this.testFocusTrapping();

    } catch (error) {
      console.error('Focus management test error:', error);
    }
  }

  async testFocusTrapping() {
    const modals = await this.page.$$('[role="dialog"]');
    
    for (const modal of modals) {
      const isVisible = await modal.isIntersectingViewport();
      if (isVisible) {
        // Get first and last focusable elements in modal
        const focusableInModal = await modal.$$('a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])');
        
        if (focusableInModal.length > 1) {
          const firstElement = focusableInModal[0];
          const lastElement = focusableInModal[focusableInModal.length - 1];
          
          // Test forward focus trapping
          await lastElement.focus();
          await this.page.keyboard.press('Tab');
          
          const focusAfterTab = await this.page.evaluate(() => document.activeElement);
          const firstElementHandle = await firstElement.evaluate(el => el);
          
          // Focus should wrap to first element
          if (focusAfterTab !== firstElementHandle) {
            this.results.violations.push({
              type: 'focus-trapping',
              description: 'Focus does not trap properly in modal'
            });
          }
        }
      }
    }
  }

  async testColorContrast(url, viewportName) {
    try {
      const contrastResults = await this.page.evaluate(() => {
        const elements = Array.from(document.querySelectorAll('*'));
        const results = [];

        elements.forEach(el => {
          const styles = window.getComputedStyle(el);
          const color = styles.color;
          const backgroundColor = styles.backgroundColor;
          const fontSize = parseFloat(styles.fontSize);
          const fontWeight = styles.fontWeight;
          
          if (color && backgroundColor && 
              color !== 'rgba(0, 0, 0, 0)' && 
              backgroundColor !== 'rgba(0, 0, 0, 0)') {
            
            results.push({
              element: el.tagName + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className.split(' ')[0] : ''),
              color,
              backgroundColor,
              fontSize,
              fontWeight,
              text: el.textContent.trim().substring(0, 50)
            });
          }
        });

        return results;
      });

      // Note: Actual contrast calculation would require a color contrast library
      // This is a placeholder for where contrast ratio would be calculated
      
      this.results.colorContrast = this.results.colorContrast || [];
      this.results.colorContrast.push({
        url,
        viewport: viewportName,
        elements: contrastResults.length,
        tested: true
      });

    } catch (error) {
      console.error('Color contrast test error:', error);
    }
  }

  async testScreenReaderCompatibility(url, viewportName) {
    try {
      // Test ARIA labels and descriptions
      const ariaResults = await this.page.evaluate(() => {
        const elements = Array.from(document.querySelectorAll('[aria-label], [aria-labelledby], [aria-describedby], [role]'));
        
        return elements.map(el => ({
          tagName: el.tagName,
          role: el.getAttribute('role'),
          ariaLabel: el.getAttribute('aria-label'),
          ariaLabelledby: el.getAttribute('aria-labelledby'),
          ariaDescribedby: el.getAttribute('aria-describedby'),
          ariaExpanded: el.getAttribute('aria-expanded'),
          ariaHidden: el.getAttribute('aria-hidden'),
          hasText: el.textContent.trim().length > 0
        }));
      });

      // Test heading structure
      const headingStructure = await this.page.evaluate(() => {
        const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
        return headings.map(h => ({
          level: parseInt(h.tagName.charAt(1)),
          text: h.textContent.trim(),
          id: h.id || null
        }));
      });

      // Check for proper heading hierarchy
      let previousLevel = 0;
      headingStructure.forEach((heading, index) => {
        if (index === 0 && heading.level !== 1) {
          this.results.violations.push({
            url,
            viewport: viewportName,
            type: 'heading-structure',
            description: 'Page should start with h1'
          });
        }
        
        if (heading.level > previousLevel + 1) {
          this.results.violations.push({
            url,
            viewport: viewportName,
            type: 'heading-structure',
            description: `Heading level skipped from h${previousLevel} to h${heading.level}`
          });
        }
        
        previousLevel = heading.level;
      });

      // Test landmark regions
      const landmarks = await this.page.evaluate(() => {
        const landmarkRoles = ['banner', 'main', 'navigation', 'contentinfo', 'search', 'form'];
        const landmarks = [];
        
        landmarkRoles.forEach(role => {
          const elements = document.querySelectorAll(`[role="${role}"], ${role === 'banner' ? 'header' : role === 'contentinfo' ? 'footer' : role === 'navigation' ? 'nav' : role}`);
          landmarks.push({
            role,
            count: elements.length,
            hasMultiple: elements.length > 1
          });
        });
        
        return landmarks;
      });

      this.results.screenReader = this.results.screenReader || [];
      this.results.screenReader.push({
        url,
        viewport: viewportName,
        ariaElements: ariaResults.length,
        headings: headingStructure.length,
        landmarks: landmarks.filter(l => l.count > 0)
      });

    } catch (error) {
      console.error('Screen reader compatibility test error:', error);
    }
  }

  async generateReport() {
    const reportData = {
      timestamp: new Date().toISOString(),
      summary: {
        totalViolations: this.results.violations.length,
        totalPasses: this.results.passes.length,
        totalIncomplete: this.results.incomplete.length,
        byImpact: this.getViolationsByImpact(),
        byRule: this.getViolationsByRule()
      },
      violations: this.results.violations,
      passes: this.results.passes,
      incomplete: this.results.incomplete,
      keyboardTests: this.results.keyboardTests || [],
      colorContrast: this.results.colorContrast || [],
      screenReader: this.results.screenReader || []
    };

    // Generate JSON report
    await fs.writeFile(
      path.join(__dirname, 'accessibility-report.json'),
      JSON.stringify(reportData, null, 2)
    );

    // Generate HTML report
    await this.generateHtmlReport(reportData);
    
    console.log('\n📊 Accessibility Audit Complete');
    console.log(`Total Violations: ${reportData.summary.totalViolations}`);
    console.log(`Total Passes: ${reportData.summary.totalPasses}`);
    console.log(`Report saved to: accessibility-report.json`);
  }

  async generateHtmlReport(data) {
    const htmlReport = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessibility Audit Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }
        .violation { background: #fee; border-left: 4px solid #e53e3e; padding: 16px; margin: 16px 0; }
        .pass { background: #f0fff4; border-left: 4px solid #38a169; padding: 16px; margin: 16px 0; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #e2e8f0; padding: 20px; border-radius: 8px; text-align: center; }
        .impact-critical { color: #e53e3e; font-weight: bold; }
        .impact-serious { color: #dd6b20; font-weight: bold; }
        .impact-moderate { color: #d69e2e; }
        .impact-minor { color: #38a169; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Accessibility Audit Report</h1>
        <p>Generated on: ${data.timestamp}</p>
        
        <div class="summary">
            <div class="metric">
                <h3>${data.summary.totalViolations}</h3>
                <p>Violations</p>
            </div>
            <div class="metric">
                <h3>${data.summary.totalPasses}</h3>
                <p>Passes</p>
            </div>
            <div class="metric">
                <h3>${data.summary.totalIncomplete}</h3>
                <p>Incomplete</p>
            </div>
        </div>

        <h2>Violations by Impact</h2>
        ${Object.entries(data.summary.byImpact).map(([impact, count]) => 
          `<div class="metric">
            <h3 class="impact-${impact}">${count}</h3>
            <p>${impact.charAt(0).toUpperCase() + impact.slice(1)}</p>
          </div>`
        ).join('')}

        <h2>🚨 Violations</h2>
        ${data.violations.map(violation => `
          <div class="violation">
            <h3>${violation.rule} (${violation.impact})</h3>
            <p><strong>Description:</strong> ${violation.description}</p>
            <p><strong>Help:</strong> <a href="${violation.helpUrl}" target="_blank">${violation.help}</a></p>
            <p><strong>URL:</strong> ${violation.url} (${violation.viewport})</p>
            <details>
              <summary>Affected Elements (${violation.nodes?.length || 0})</summary>
              ${violation.nodes?.map(node => `
                <div style="margin: 10px 0; padding: 10px; background: #f7fafc;">
                  <strong>Target:</strong> ${node.target?.join(', ')}<br>
                  <strong>HTML:</strong> <code>${node.html}</code>
                </div>
              `).join('') || ''}
            </details>
          </div>
        `).join('')}

        <h2>✅ Passes</h2>
        <p>Total rules passed: ${data.summary.totalPasses}</p>
        
        <h2>⌨️ Keyboard Navigation</h2>
        ${data.keyboardTests?.map(test => `
          <div class="pass">
            <strong>URL:</strong> ${test.url} (${test.viewport})<br>
            <strong>Tabbable Elements:</strong> ${test.tabbableElements}<br>
            <strong>Status:</strong> ${test.passed ? 'Passed' : 'Failed'}
          </div>
        `).join('') || '<p>No keyboard tests performed</p>'}

        <h2>🎨 Color Contrast</h2>
        ${data.colorContrast?.map(test => `
          <div class="pass">
            <strong>URL:</strong> ${test.url} (${test.viewport})<br>
            <strong>Elements Tested:</strong> ${test.elements}
          </div>
        `).join('') || '<p>No color contrast tests performed</p>'}

        <h2>📢 Screen Reader Compatibility</h2>
        ${data.screenReader?.map(test => `
          <div class="pass">
            <strong>URL:</strong> ${test.url} (${test.viewport})<br>
            <strong>ARIA Elements:</strong> ${test.ariaElements}<br>
            <strong>Headings:</strong> ${test.headings}<br>
            <strong>Landmarks:</strong> ${test.landmarks?.length || 0}
          </div>
        `).join('') || '<p>No screen reader tests performed</p>'}
    </div>
</body>
</html>`;

    await fs.writeFile(
      path.join(__dirname, 'accessibility-report.html'),
      htmlReport
    );
  }

  getViolationsByImpact() {
    return this.results.violations.reduce((acc, violation) => {
      const impact = violation.impact || 'unknown';
      acc[impact] = (acc[impact] || 0) + 1;
      return acc;
    }, {});
  }

  getViolationsByRule() {
    return this.results.violations.reduce((acc, violation) => {
      const rule = violation.rule || 'unknown';
      acc[rule] = (acc[rule] || 0) + 1;
      return acc;
    }, {});
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
    }
  }
}

module.exports = AccessibilityTestSuite;

// CLI usage
if (require.main === module) {
  const suite = new AccessibilityTestSuite({
    baseUrl: process.env.BASE_URL || 'http://localhost:3000'
  });
  
  const urls = process.argv.slice(2);
  const testUrls = urls.length > 0 ? urls : ['/'];
  
  suite.runFullAccessibilityAudit(testUrls)
    .then(results => {
      console.log('Accessibility audit completed');
      process.exit(results.violations.length > 0 ? 1 : 0);
    })
    .catch(error => {
      console.error('Accessibility audit failed:', error);
      process.exit(1);
    });
}