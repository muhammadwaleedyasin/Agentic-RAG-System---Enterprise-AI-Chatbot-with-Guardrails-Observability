/**
 * Performance Testing for Accessibility Features
 * Tests the performance impact of accessibility features and assistive technologies
 */

const puppeteer = require('puppeteer');
const lighthouse = require('lighthouse');
const fs = require('fs').promises;
const path = require('path');

class PerformanceAccessibilityTest {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:3000';
    this.browser = null;
    this.results = {
      lighthouseScores: [],
      performanceMetrics: [],
      accessibilityPerformance: [],
      comparisons: []
    };
  }

  async initialize() {
    this.browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--remote-debugging-port=9222'
      ]
    });
  }

  async runPerformanceAccessibilityTest() {
    await this.initialize();
    
    console.log('⚡ Running performance accessibility tests...');
    
    // Test baseline performance
    await this.testBaselinePerformance();
    
    // Test with screen reader simulation
    await this.testScreenReaderPerformance();
    
    // Test with high contrast mode
    await this.testHighContrastPerformance();
    
    // Test with reduced motion
    await this.testReducedMotionPerformance();
    
    // Test keyboard navigation performance
    await this.testKeyboardNavigationPerformance();
    
    // Run Lighthouse accessibility audit
    await this.runLighthouseAccessibilityAudit();
    
    await this.generatePerformanceReport();
    await this.cleanup();
    
    return this.results;
  }

  async testBaselinePerformance() {
    console.log('  📊 Testing baseline performance...');
    
    const page = await this.browser.newPage();
    
    // Enable performance monitoring
    await page.coverage.startJSCoverage();
    await page.coverage.startCSSCoverage();
    
    const startTime = Date.now();
    
    await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
    
    const loadTime = Date.now() - startTime;
    
    // Get performance metrics
    const metrics = await page.metrics();
    const performanceEntries = await page.evaluate(() => {
      return JSON.parse(JSON.stringify(performance.getEntriesByType('navigation')));
    });
    
    // Get coverage data
    const jsCoverage = await page.coverage.stopJSCoverage();
    const cssCoverage = await page.coverage.stopCSSCoverage();
    
    const totalJSBytes = jsCoverage.reduce((total, entry) => total + entry.text.length, 0);
    const usedJSBytes = jsCoverage.reduce((total, entry) => total + (entry.ranges || []).reduce((sum, range) => sum + (range.end - range.start), 0), 0);
    
    const totalCSSBytes = cssCoverage.reduce((total, entry) => total + entry.text.length, 0);
    const usedCSSBytes = cssCoverage.reduce((total, entry) => total + (entry.ranges || []).reduce((sum, range) => sum + (range.end - range.start), 0), 0);

    this.results.performanceMetrics.push({
      test: 'baseline',
      loadTime,
      metrics: {
        ...metrics,
        jsHeapUsedSize: metrics.JSHeapUsedSize,
        jsHeapTotalSize: metrics.JSHeapTotalSize,
        timestamp: metrics.Timestamp
      },
      navigation: performanceEntries[0] || {},
      coverage: {
        js: {
          total: totalJSBytes,
          used: usedJSBytes,
          percentage: totalJSBytes > 0 ? (usedJSBytes / totalJSBytes) * 100 : 0
        },
        css: {
          total: totalCSSBytes,
          used: usedCSSBytes,
          percentage: totalCSSBytes > 0 ? (usedCSSBytes / totalCSSBytes) * 100 : 0
        }
      }
    });
    
    await page.close();
  }

  async testScreenReaderPerformance() {
    console.log('  🔊 Testing screen reader performance impact...');
    
    const page = await this.browser.newPage();
    
    // Simulate screen reader behavior
    await page.evaluateOnNewDocument(() => {
      // Override focus to simulate screen reader navigation
      const originalFocus = HTMLElement.prototype.focus;
      HTMLElement.prototype.focus = function() {
        originalFocus.call(this);
        // Simulate screen reader announcing element
        this.setAttribute('aria-live', 'polite');
        setTimeout(() => this.removeAttribute('aria-live'), 100);
      };
      
      // Simulate ARIA live region announcements
      window.announceToScreenReader = function(message) {
        const announcer = document.createElement('div');
        announcer.setAttribute('aria-live', 'assertive');
        announcer.setAttribute('aria-atomic', 'true');
        announcer.style.position = 'absolute';
        announcer.style.left = '-10000px';
        announcer.textContent = message;
        document.body.appendChild(announcer);
        setTimeout(() => document.body.removeChild(announcer), 1000);
      };
    });

    const startTime = Date.now();
    await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
    const loadTime = Date.now() - startTime;

    // Simulate screen reader navigation through page
    const navigationStartTime = Date.now();
    
    // Get all focusable elements
    const focusableElements = await page.$$('a, button, input, textarea, select, [tabindex]');
    
    // Simulate screen reader moving through elements
    for (let i = 0; i < Math.min(focusableElements.length, 10); i++) {
      await focusableElements[i].focus();
      await page.waitForTimeout(50); // Simulate screen reader processing time
      
      // Simulate reading element content
      await page.evaluate(() => {
        const focused = document.activeElement;
        if (focused) {
          const text = focused.textContent || focused.ariaLabel || focused.title || '';
          window.announceToScreenReader(text);
        }
      });
    }
    
    const navigationTime = Date.now() - navigationStartTime;
    const metrics = await page.metrics();
    
    this.results.accessibilityPerformance.push({
      test: 'screen_reader_simulation',
      loadTime,
      navigationTime,
      focusableElements: focusableElements.length,
      memoryUsage: metrics.JSHeapUsedSize,
      performanceImpact: {
        memoryIncrease: 0, // Will be calculated in comparison
        timeIncrease: 0    // Will be calculated in comparison
      }
    });
    
    await page.close();
  }

  async testHighContrastPerformance() {
    console.log('  🎨 Testing high contrast mode performance...');
    
    const page = await this.browser.newPage();
    
    // Enable high contrast mode
    await page.emulateMediaFeatures([
      { name: 'prefers-contrast', value: 'high' },
      { name: 'prefers-color-scheme', value: 'dark' }
    ]);
    
    // Inject high contrast CSS
    await page.addStyleTag({
      content: `
        * {
          filter: contrast(150%) !important;
        }
        body {
          background: black !important;
          color: white !important;
        }
        a {
          color: yellow !important;
        }
        button {
          background: white !important;
          color: black !important;
          border: 2px solid white !important;
        }
      `
    });

    const startTime = Date.now();
    await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
    const loadTime = Date.now() - startTime;

    // Test style recalculation performance
    const styleRecalcStartTime = Date.now();
    
    await page.evaluate(() => {
      // Force style recalculation
      document.body.offsetHeight;
      
      // Simulate contrast adjustments
      const elements = document.querySelectorAll('*');
      elements.forEach(el => {
        el.style.filter = 'contrast(200%)';
        el.offsetHeight; // Force reflow
      });
    });
    
    const styleRecalcTime = Date.now() - styleRecalcStartTime;
    const metrics = await page.metrics();
    
    this.results.accessibilityPerformance.push({
      test: 'high_contrast_mode',
      loadTime,
      styleRecalcTime,
      memoryUsage: metrics.JSHeapUsedSize,
      layoutRecalculations: await page.evaluate(() => {
        return performance.getEntriesByType('measure').filter(entry => 
          entry.name.includes('style') || entry.name.includes('layout')
        ).length;
      })
    });
    
    await page.close();
  }

  async testReducedMotionPerformance() {
    console.log('  🎭 Testing reduced motion performance...');
    
    const page = await this.browser.newPage();
    
    // Enable reduced motion preference
    await page.emulateMediaFeatures([
      { name: 'prefers-reduced-motion', value: 'reduce' }
    ]);
    
    // Inject CSS to disable animations
    await page.addStyleTag({
      content: `
        *, *::before, *::after {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
          scroll-behavior: auto !important;
        }
      `
    });

    const startTime = Date.now();
    await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
    const loadTime = Date.now() - startTime;

    // Test interaction performance without animations
    const interactionStartTime = Date.now();
    
    const buttons = await page.$$('button, [role="button"]');
    for (const button of buttons.slice(0, 5)) {
      await button.click();
      await page.waitForTimeout(10); // Minimal wait since animations are disabled
    }
    
    const interactionTime = Date.now() - interactionStartTime;
    const metrics = await page.metrics();
    
    // Check for remaining animations
    const animationsCount = await page.evaluate(() => {
      return document.getAnimations().length;
    });
    
    this.results.accessibilityPerformance.push({
      test: 'reduced_motion',
      loadTime,
      interactionTime,
      memoryUsage: metrics.JSHeapUsedSize,
      remainingAnimations: animationsCount,
      performanceGain: 0 // Will be calculated in comparison
    });
    
    await page.close();
  }

  async testKeyboardNavigationPerformance() {
    console.log('  ⌨️ Testing keyboard navigation performance...');
    
    const page = await this.browser.newPage();
    await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });

    // Test tab navigation performance
    const tabStartTime = Date.now();
    
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press('Tab');
      await page.waitForTimeout(10);
      
      // Check focus ring rendering
      await page.evaluate(() => {
        const focused = document.activeElement;
        if (focused) {
          // Force focus ring rendering
          focused.style.outline = '2px solid blue';
          focused.offsetHeight; // Force reflow
        }
      });
    }
    
    const tabTime = Date.now() - tabStartTime;
    
    // Test arrow key navigation performance
    const arrowStartTime = Date.now();
    
    const widgets = await page.$$('[role="menu"], [role="tablist"], [role="listbox"]');
    for (const widget of widgets) {
      await widget.focus();
      for (let i = 0; i < 5; i++) {
        await page.keyboard.press('ArrowDown');
        await page.waitForTimeout(10);
      }
    }
    
    const arrowTime = Date.now() - arrowStartTime;
    const metrics = await page.metrics();
    
    // Test focus management efficiency
    const focusableElements = await page.evaluate(() => {
      const elements = document.querySelectorAll(
        'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
      );
      return elements.length;
    });
    
    this.results.accessibilityPerformance.push({
      test: 'keyboard_navigation',
      tabNavigationTime: tabTime,
      arrowNavigationTime: arrowTime,
      focusableElements,
      averageTimePerElement: focusableElements > 0 ? tabTime / Math.min(20, focusableElements) : 0,
      memoryUsage: metrics.JSHeapUsedSize
    });
    
    await page.close();
  }

  async runLighthouseAccessibilityAudit() {
    console.log('  🏠 Running Lighthouse accessibility audit...');
    
    try {
      const config = {
        extends: 'lighthouse:default',
        settings: {
          onlyCategories: ['accessibility', 'performance'],
          output: 'json'
        }
      };
      
      const runnerResult = await lighthouse(this.baseUrl, {
        port: 9222,
        output: 'json',
        logLevel: 'info'
      }, config);
      
      if (runnerResult && runnerResult.lhr) {
        const { lhr } = runnerResult;
        
        this.results.lighthouseScores.push({
          accessibility: {
            score: lhr.categories.accessibility.score * 100,
            audits: Object.keys(lhr.audits).filter(key => 
              lhr.audits[key].scoreDisplayMode !== 'notApplicable' &&
              key.includes('accessibility') || key.includes('color-contrast') || 
              key.includes('focus') || key.includes('aria')
            ).map(key => ({
              id: key,
              title: lhr.audits[key].title,
              score: lhr.audits[key].score,
              displayValue: lhr.audits[key].displayValue,
              description: lhr.audits[key].description
            }))
          },
          performance: {
            score: lhr.categories.performance.score * 100,
            metrics: {
              firstContentfulPaint: lhr.audits['first-contentful-paint'].numericValue,
              largestContentfulPaint: lhr.audits['largest-contentful-paint'].numericValue,
              speedIndex: lhr.audits['speed-index'].numericValue,
              totalBlockingTime: lhr.audits['total-blocking-time'].numericValue,
              cumulativeLayoutShift: lhr.audits['cumulative-layout-shift'].numericValue
            }
          }
        });
      }
    } catch (error) {
      console.error('Lighthouse audit failed:', error);
      this.results.lighthouseScores.push({
        error: error.message,
        accessibility: { score: 0, audits: [] },
        performance: { score: 0, metrics: {} }
      });
    }
  }

  async generatePerformanceReport() {
    // Calculate performance comparisons
    this.calculatePerformanceComparisons();
    
    const reportData = {
      timestamp: new Date().toISOString(),
      summary: {
        baselinePerformance: this.getBaselineMetrics(),
        accessibilityImpact: this.calculateAccessibilityImpact(),
        lighthouseScore: this.getLighthouseScore(),
        recommendations: this.generatePerformanceRecommendations()
      },
      performanceMetrics: this.results.performanceMetrics,
      accessibilityPerformance: this.results.accessibilityPerformance,
      lighthouseScores: this.results.lighthouseScores,
      comparisons: this.results.comparisons
    };

    // Save JSON report
    await fs.writeFile(
      path.join(__dirname, 'performance-accessibility-report.json'),
      JSON.stringify(reportData, null, 2)
    );

    // Generate HTML report
    await this.generatePerformanceHtmlReport(reportData);
    
    console.log('\n📊 Performance Accessibility Testing Complete');
    console.log(`Lighthouse Accessibility Score: ${reportData.summary.lighthouseScore}/100`);
    console.log(`Performance Impact: ${reportData.summary.accessibilityImpact}%`);
    console.log(`Report saved to: performance-accessibility-report.json`);
  }

  calculatePerformanceComparisons() {
    const baseline = this.results.performanceMetrics.find(m => m.test === 'baseline');
    if (!baseline) return;
    
    this.results.accessibilityPerformance.forEach(test => {
      const comparison = {
        test: test.test,
        loadTimeComparison: {
          baseline: baseline.loadTime,
          test: test.loadTime,
          difference: test.loadTime - baseline.loadTime,
          percentageIncrease: ((test.loadTime - baseline.loadTime) / baseline.loadTime) * 100
        },
        memoryComparison: {
          baseline: baseline.metrics.jsHeapUsedSize,
          test: test.memoryUsage,
          difference: test.memoryUsage - baseline.metrics.jsHeapUsedSize,
          percentageIncrease: ((test.memoryUsage - baseline.metrics.jsHeapUsedSize) / baseline.metrics.jsHeapUsedSize) * 100
        }
      };
      
      this.results.comparisons.push(comparison);
    });
  }

  getBaselineMetrics() {
    const baseline = this.results.performanceMetrics.find(m => m.test === 'baseline');
    return baseline ? {
      loadTime: baseline.loadTime,
      memoryUsage: baseline.metrics.jsHeapUsedSize,
      jsCodeCoverage: baseline.coverage.js.percentage,
      cssCodeCoverage: baseline.coverage.css.percentage
    } : null;
  }

  calculateAccessibilityImpact() {
    if (this.results.comparisons.length === 0) return 0;
    
    const avgLoadTimeIncrease = this.results.comparisons.reduce((sum, comp) => 
      sum + comp.loadTimeComparison.percentageIncrease, 0) / this.results.comparisons.length;
    
    return Math.round(avgLoadTimeIncrease * 100) / 100;
  }

  getLighthouseScore() {
    if (this.results.lighthouseScores.length === 0) return 0;
    return this.results.lighthouseScores[0].accessibility.score;
  }

  generatePerformanceRecommendations() {
    const recommendations = [];
    
    // Check performance impact
    const avgImpact = this.calculateAccessibilityImpact();
    if (avgImpact > 10) {
      recommendations.push({
        priority: 'high',
        category: 'performance_optimization',
        description: `Accessibility features are causing ${avgImpact}% performance impact`,
        suggestions: [
          'Optimize CSS for high contrast mode',
          'Use efficient ARIA live region updates',
          'Implement lazy loading for accessibility features',
          'Consider server-side accessibility optimizations'
        ]
      });
    }
    
    // Check Lighthouse score
    const lighthouseScore = this.getLighthouseScore();
    if (lighthouseScore < 90) {
      recommendations.push({
        priority: 'medium',
        category: 'accessibility_compliance',
        description: `Lighthouse accessibility score is ${lighthouseScore}/100`,
        suggestions: [
          'Fix color contrast issues',
          'Add missing ARIA labels',
          'Improve keyboard navigation',
          'Ensure proper heading hierarchy'
        ]
      });
    }
    
    // Check code coverage
    const baseline = this.getBaselineMetrics();
    if (baseline && baseline.jsCodeCoverage < 60) {
      recommendations.push({
        priority: 'medium',
        category: 'code_optimization',
        description: `JavaScript code coverage is only ${baseline.jsCodeCoverage}%`,
        suggestions: [
          'Remove unused JavaScript code',
          'Implement code splitting',
          'Load accessibility features on demand',
          'Optimize bundle size'
        ]
      });
    }
    
    return recommendations;
  }

  async generatePerformanceHtmlReport(data) {
    const htmlReport = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Accessibility Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }
        .metric { background: #e2e8f0; padding: 20px; border-radius: 8px; text-align: center; margin: 10px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .test-result { background: #f0fff4; border-left: 4px solid #38a169; padding: 16px; margin: 16px 0; }
        .warning { background: #fffbf0; border-left: 4px solid #d69e2e; padding: 16px; margin: 16px 0; }
        .comparison { background: #f7fafc; border: 1px solid #e2e8f0; padding: 16px; margin: 16px 0; border-radius: 8px; }
        .score-excellent { color: #38a169; font-weight: bold; }
        .score-good { color: #3182ce; font-weight: bold; }
        .score-fair { color: #d69e2e; font-weight: bold; }
        .score-poor { color: #e53e3e; font-weight: bold; }
        .chart { width: 100%; height: 300px; background: #f7fafc; border: 1px solid #e2e8f0; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #f7fafc; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Performance Accessibility Test Report</h1>
        <p>Generated on: ${data.timestamp}</p>
        
        <div class="summary">
            <div class="metric">
                <h3 class="${this.getScoreClass(data.summary.lighthouseScore)}">${data.summary.lighthouseScore}/100</h3>
                <p>Lighthouse A11y Score</p>
            </div>
            <div class="metric">
                <h3>${data.summary.accessibilityImpact}%</h3>
                <p>Performance Impact</p>
            </div>
            <div class="metric">
                <h3>${data.summary.baselinePerformance?.loadTime || 'N/A'}ms</h3>
                <p>Baseline Load Time</p>
            </div>
            <div class="metric">
                <h3>${Math.round((data.summary.baselinePerformance?.jsCodeCoverage || 0))}%</h3>
                <p>JS Code Coverage</p>
            </div>
        </div>

        <h2>📊 Performance Comparisons</h2>
        ${data.comparisons.map(comp => `
          <div class="comparison">
            <h3>${comp.test.replace(/_/g, ' ').toUpperCase()}</h3>
            <table>
              <tr>
                <th>Metric</th>
                <th>Baseline</th>
                <th>With A11y</th>
                <th>Difference</th>
                <th>% Change</th>
              </tr>
              <tr>
                <td>Load Time</td>
                <td>${comp.loadTimeComparison.baseline}ms</td>
                <td>${comp.loadTimeComparison.test}ms</td>
                <td>${comp.loadTimeComparison.difference}ms</td>
                <td class="${comp.loadTimeComparison.percentageIncrease > 10 ? 'score-poor' : comp.loadTimeComparison.percentageIncrease > 5 ? 'score-fair' : 'score-good'}">
                  ${comp.loadTimeComparison.percentageIncrease.toFixed(1)}%
                </td>
              </tr>
              <tr>
                <td>Memory Usage</td>
                <td>${Math.round(comp.memoryComparison.baseline / 1024 / 1024)}MB</td>
                <td>${Math.round(comp.memoryComparison.test / 1024 / 1024)}MB</td>
                <td>${Math.round(comp.memoryComparison.difference / 1024 / 1024)}MB</td>
                <td class="${comp.memoryComparison.percentageIncrease > 15 ? 'score-poor' : comp.memoryComparison.percentageIncrease > 8 ? 'score-fair' : 'score-good'}">
                  ${comp.memoryComparison.percentageIncrease.toFixed(1)}%
                </td>
              </tr>
            </table>
          </div>
        `).join('')}

        <h2>🧪 Accessibility Performance Tests</h2>
        ${data.accessibilityPerformance.map(test => `
          <div class="test-result">
            <h3>${test.test.replace(/_/g, ' ').toUpperCase()}</h3>
            <p><strong>Load Time:</strong> ${test.loadTime}ms</p>
            <p><strong>Memory Usage:</strong> ${Math.round(test.memoryUsage / 1024 / 1024)}MB</p>
            ${test.navigationTime ? `<p><strong>Navigation Time:</strong> ${test.navigationTime}ms</p>` : ''}
            ${test.focusableElements ? `<p><strong>Focusable Elements:</strong> ${test.focusableElements}</p>` : ''}
            ${test.remainingAnimations !== undefined ? `<p><strong>Remaining Animations:</strong> ${test.remainingAnimations}</p>` : ''}
          </div>
        `).join('')}

        ${data.lighthouseScores.length > 0 && !data.lighthouseScores[0].error ? `
        <h2>🏠 Lighthouse Accessibility Audit</h2>
        <div class="test-result">
          <h3>Overall Accessibility Score: <span class="${this.getScoreClass(data.lighthouseScores[0].accessibility.score)}">${data.lighthouseScores[0].accessibility.score}/100</span></h3>
          <h4>Failed Audits:</h4>
          <ul>
            ${data.lighthouseScores[0].accessibility.audits.filter(audit => audit.score !== 1).map(audit => `
              <li><strong>${audit.title}:</strong> ${audit.description}</li>
            `).join('')}
          </ul>
        </div>
        ` : ''}

        <h2>💡 Performance Recommendations</h2>
        ${data.summary.recommendations.map(rec => `
          <div class="warning">
            <h3>${rec.category.replace(/_/g, ' ').toUpperCase()} (${rec.priority.toUpperCase()} Priority)</h3>
            <p>${rec.description}</p>
            <h4>Suggestions:</h4>
            <ul>
              ${rec.suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
            </ul>
          </div>
        `).join('')}

        <h2>📈 Performance Metrics Details</h2>
        ${data.performanceMetrics.map(metric => `
          <div class="test-result">
            <h3>Baseline Performance</h3>
            <table>
              <tr><td>Load Time</td><td>${metric.loadTime}ms</td></tr>
              <tr><td>JS Heap Used</td><td>${Math.round(metric.metrics.jsHeapUsedSize / 1024 / 1024)}MB</td></tr>
              <tr><td>JS Code Coverage</td><td>${metric.coverage.js.percentage.toFixed(1)}%</td></tr>
              <tr><td>CSS Code Coverage</td><td>${metric.coverage.css.percentage.toFixed(1)}%</td></tr>
            </table>
          </div>
        `).join('')}
    </div>
</body>
</html>`;

    await fs.writeFile(
      path.join(__dirname, 'performance-accessibility-report.html'),
      htmlReport
    );
  }

  getScoreClass(score) {
    if (score >= 90) return 'score-excellent';
    if (score >= 75) return 'score-good';
    if (score >= 60) return 'score-fair';
    return 'score-poor';
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
    }
  }
}

module.exports = PerformanceAccessibilityTest;

// CLI usage
if (require.main === module) {
  const test = new PerformanceAccessibilityTest({
    baseUrl: process.env.BASE_URL || 'http://localhost:3000'
  });
  
  test.runPerformanceAccessibilityTest()
    .then(results => {
      console.log('Performance accessibility testing completed');
      const impact = test.calculateAccessibilityImpact();
      process.exit(impact > 25 ? 1 : 0); // Fail if impact > 25%
    })
    .catch(error => {
      console.error('Performance accessibility testing failed:', error);
      process.exit(1);
    });
}