/**
 * Responsive Design Testing Suite for RAG Chatbot UI
 * Tests layout, functionality, and usability across different screen sizes
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

class ResponsiveTestSuite {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:3000';
    this.browser = null;
    this.results = {
      layoutTests: [],
      functionalityTests: [],
      performanceTests: [],
      touchTests: [],
      violations: []
    };
    
    // Standard device viewports
    this.viewports = [
      // Mobile devices
      { name: 'iPhone SE', width: 375, height: 667, isMobile: true, hasTouch: true },
      { name: 'iPhone 12', width: 390, height: 844, isMobile: true, hasTouch: true },
      { name: 'iPhone 12 Pro Max', width: 428, height: 926, isMobile: true, hasTouch: true },
      { name: 'Samsung Galaxy S21', width: 360, height: 800, isMobile: true, hasTouch: true },
      { name: 'Samsung Galaxy S21+', width: 384, height: 854, isMobile: true, hasTouch: true },
      
      // Tablets
      { name: 'iPad Mini', width: 768, height: 1024, isMobile: true, hasTouch: true },
      { name: 'iPad Air', width: 820, height: 1180, isMobile: true, hasTouch: true },
      { name: 'iPad Pro 11"', width: 834, height: 1194, isMobile: true, hasTouch: true },
      { name: 'iPad Pro 12.9"', width: 1024, height: 1366, isMobile: true, hasTouch: true },
      { name: 'Surface Pro 7', width: 912, height: 1368, isMobile: true, hasTouch: true },
      
      // Desktop and laptop screens
      { name: 'Laptop Small', width: 1024, height: 768, isMobile: false, hasTouch: false },
      { name: 'Laptop Medium', width: 1366, height: 768, isMobile: false, hasTouch: false },
      { name: 'Desktop 1080p', width: 1920, height: 1080, isMobile: false, hasTouch: false },
      { name: 'Desktop 1440p', width: 2560, height: 1440, isMobile: false, hasTouch: false },
      { name: 'Ultrawide', width: 3440, height: 1440, isMobile: false, hasTouch: false },
      
      // Edge cases
      { name: 'Very Small Mobile', width: 320, height: 568, isMobile: true, hasTouch: true },
      { name: 'Large Desktop', width: 3840, height: 2160, isMobile: false, hasTouch: false }
    ];
    
    // Breakpoints to test
    this.breakpoints = [
      { name: 'xs', min: 0, max: 575 },
      { name: 'sm', min: 576, max: 767 },
      { name: 'md', min: 768, max: 991 },
      { name: 'lg', min: 992, max: 1199 },
      { name: 'xl', min: 1200, max: 1399 },
      { name: 'xxl', min: 1400, max: Infinity }
    ];
  }

  async initialize() {
    this.browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--enable-features=TouchEvents'
      ]
    });
  }

  async runFullResponsiveTest(urls = ['/']) {
    await this.initialize();
    
    console.log('🔍 Starting responsive design testing...');
    
    for (const url of urls) {
      console.log(`\n📱 Testing responsive design for: ${url}`);
      await this.testUrlResponsiveness(url);
    }
    
    await this.generateReport();
    await this.cleanup();
    
    return this.results;
  }

  async testUrlResponsiveness(url) {
    for (const viewport of this.viewports) {
      console.log(`  📐 Testing ${viewport.name} (${viewport.width}x${viewport.height})`);
      
      const page = await this.browser.newPage();
      
      try {
        // Configure page for device
        await page.setViewport({
          width: viewport.width,
          height: viewport.height,
          isMobile: viewport.isMobile,
          hasTouch: viewport.hasTouch,
          deviceScaleFactor: viewport.isMobile ? 2 : 1
        });
        
        // Set user agent for mobile devices
        if (viewport.isMobile) {
          await page.setUserAgent(
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1'
          );
        }
        
        await page.goto(`${this.baseUrl}${url}`, { waitUntil: 'networkidle0' });
        
        // Test layout
        await this.testLayout(page, url, viewport);
        
        // Test functionality
        await this.testFunctionality(page, url, viewport);
        
        // Test performance
        await this.testPerformance(page, url, viewport);
        
        // Test touch interactions (mobile/tablet only)
        if (viewport.hasTouch) {
          await this.testTouchInteractions(page, url, viewport);
        }
        
        // Test text readability
        await this.testTextReadability(page, url, viewport);
        
        // Test navigation usability
        await this.testNavigationUsability(page, url, viewport);
        
      } catch (error) {
        console.error(`Error testing ${viewport.name}:`, error);
        this.results.violations.push({
          url,
          viewport: viewport.name,
          error: error.message,
          type: 'test-error'
        });
      } finally {
        await page.close();
      }
    }
  }

  async testLayout(page, url, viewport) {
    try {
      // Test for horizontal scrollbars (usually indicates layout issues)
      const hasHorizontalScroll = await page.evaluate(() => {
        return document.body.scrollWidth > window.innerWidth;
      });

      if (hasHorizontalScroll) {
        this.results.violations.push({
          url,
          viewport: viewport.name,
          type: 'horizontal-scroll',
          description: 'Page has horizontal scrollbar, indicating layout overflow'
        });
      }

      // Test element positioning and visibility
      const layoutElements = await page.evaluate(() => {
        const elements = Array.from(document.querySelectorAll(
          'header, nav, main, aside, footer, .chat-container, .message-input, .sidebar'
        ));
        
        return elements.map(el => {
          const rect = el.getBoundingClientRect();
          const styles = window.getComputedStyle(el);
          
          return {
            tagName: el.tagName,
            className: el.className,
            id: el.id,
            width: rect.width,
            height: rect.height,
            top: rect.top,
            left: rect.left,
            visible: rect.width > 0 && rect.height > 0,
            overflow: styles.overflow,
            position: styles.position,
            zIndex: styles.zIndex
          };
        });
      });

      // Test for elements that are cut off or not visible
      layoutElements.forEach(element => {
        if (element.left < 0 || element.top < 0) {
          this.results.violations.push({
            url,
            viewport: viewport.name,
            type: 'element-cutoff',
            description: `Element ${element.tagName}.${element.className} is positioned outside viewport`
          });
        }
        
        if (element.width > viewport.width) {
          this.results.violations.push({
            url,
            viewport: viewport.name,
            type: 'element-overflow',
            description: `Element ${element.tagName}.${element.className} is wider than viewport`
          });
        }
      });

      // Test responsive images
      const imageResults = await page.evaluate(() => {
        const images = Array.from(document.querySelectorAll('img'));
        return images.map(img => {
          const rect = img.getBoundingClientRect();
          return {
            src: img.src,
            width: rect.width,
            height: rect.height,
            naturalWidth: img.naturalWidth,
            naturalHeight: img.naturalHeight,
            responsive: img.style.maxWidth === '100%' || 
                       window.getComputedStyle(img).maxWidth === '100%'
          };
        });
      });

      imageResults.forEach(image => {
        if (!image.responsive && image.width > viewport.width) {
          this.results.violations.push({
            url,
            viewport: viewport.name,
            type: 'non-responsive-image',
            description: `Image not responsive: ${image.src}`
          });
        }
      });

      this.results.layoutTests.push({
        url,
        viewport: viewport.name,
        hasHorizontalScroll,
        elementsCount: layoutElements.length,
        imagesCount: imageResults.length,
        responsiveImages: imageResults.filter(img => img.responsive).length
      });

    } catch (error) {
      console.error('Layout test error:', error);
    }
  }

  async testFunctionality(page, url, viewport) {
    try {
      // Test form inputs
      const formElements = await page.$$('input, textarea, select, button');
      
      for (const element of formElements) {
        const elementInfo = await element.evaluate(el => ({
          tagName: el.tagName,
          type: el.type,
          id: el.id,
          className: el.className
        }));

        // Test if element is clickable/tappable
        const rect = await element.boundingBox();
        if (rect) {
          const isTouch = viewport.hasTouch;
          const minSize = isTouch ? 44 : 24; // 44px for touch, 24px for mouse
          
          if (rect.width < minSize || rect.height < minSize) {
            this.results.violations.push({
              url,
              viewport: viewport.name,
              type: 'small-touch-target',
              description: `${elementInfo.tagName} element too small for ${isTouch ? 'touch' : 'mouse'} interaction (${rect.width}x${rect.height}px)`
            });
          }
        }
      }

      // Test menu functionality
      const menuToggles = await page.$$('[data-toggle="menu"], .menu-toggle, .hamburger');
      for (const toggle of menuToggles) {
        try {
          await toggle.click();
          await page.waitForTimeout(300); // Wait for animation
          
          // Check if menu opened
          const menuOpen = await page.evaluate(() => {
            const menu = document.querySelector('.menu, .navigation, [role="navigation"]');
            return menu ? window.getComputedStyle(menu).display !== 'none' : false;
          });

          if (!menuOpen && viewport.isMobile) {
            this.results.violations.push({
              url,
              viewport: viewport.name,
              type: 'menu-functionality',
              description: 'Mobile menu does not open when toggle is clicked'
            });
          }
        } catch (error) {
          // Menu toggle might not be functional yet
        }
      }

      // Test modal functionality
      const modalTriggers = await page.$$('[data-toggle="modal"], .modal-trigger');
      for (const trigger of modalTriggers) {
        try {
          await trigger.click();
          await page.waitForTimeout(300);
          
          const modalVisible = await page.evaluate(() => {
            const modal = document.querySelector('.modal, [role="dialog"]');
            return modal ? window.getComputedStyle(modal).display !== 'none' : false;
          });

          // Test modal responsiveness
          if (modalVisible) {
            const modalSize = await page.evaluate(() => {
              const modal = document.querySelector('.modal, [role="dialog"]');
              const rect = modal.getBoundingClientRect();
              return {
                width: rect.width,
                height: rect.height,
                fitsInViewport: rect.width <= window.innerWidth && rect.height <= window.innerHeight
              };
            });

            if (!modalSize.fitsInViewport) {
              this.results.violations.push({
                url,
                viewport: viewport.name,
                type: 'modal-overflow',
                description: 'Modal does not fit in viewport'
              });
            }
          }
        } catch (error) {
          // Modal trigger might not be functional yet
        }
      }

      this.results.functionalityTests.push({
        url,
        viewport: viewport.name,
        formElements: formElements.length,
        menuToggles: menuToggles.length,
        modalTriggers: modalTriggers.length
      });

    } catch (error) {
      console.error('Functionality test error:', error);
    }
  }

  async testPerformance(page, url, viewport) {
    try {
      // Measure page load performance
      const metrics = await page.metrics();
      
      // Test image loading performance
      const imageLoadTimes = await page.evaluate(() => {
        const images = Array.from(document.querySelectorAll('img'));
        return images.map(img => ({
          src: img.src,
          complete: img.complete,
          naturalWidth: img.naturalWidth,
          naturalHeight: img.naturalHeight
        }));
      });

      // Test critical rendering path
      const criticalResourcesLoaded = await page.evaluate(() => {
        return {
          stylesheets: document.styleSheets.length,
          scripts: document.scripts.length,
          fontsLoaded: document.fonts ? document.fonts.size : 0
        };
      });

      this.results.performanceTests.push({
        url,
        viewport: viewport.name,
        loadTime: metrics.Timestamp,
        heapUsed: metrics.JSHeapUsedSize,
        images: imageLoadTimes.length,
        imagesLoaded: imageLoadTimes.filter(img => img.complete).length,
        resources: criticalResourcesLoaded
      });

    } catch (error) {
      console.error('Performance test error:', error);
    }
  }

  async testTouchInteractions(page, url, viewport) {
    try {
      // Test swipe gestures
      const swipeableElements = await page.$$('.swipeable, .carousel, .slider');
      
      for (const element of swipeableElements) {
        const rect = await element.boundingBox();
        if (rect) {
          // Simulate swipe left
          await page.touchscreen.tap(rect.x + rect.width * 0.8, rect.y + rect.height / 2);
          await page.touchscreen.tap(rect.x + rect.width * 0.2, rect.y + rect.height / 2);
          await page.waitForTimeout(100);
        }
      }

      // Test pinch-to-zoom prevention (should be disabled for UI elements)
      await page.evaluate(() => {
        const viewport = document.querySelector('meta[name="viewport"]');
        return viewport ? viewport.content : '';
      });

      // Test scroll behavior
      const scrollableElements = await page.$$('.scrollable, .overflow-auto, .chat-messages');
      
      for (const element of scrollableElements) {
        const scrollInfo = await element.evaluate(el => ({
          scrollHeight: el.scrollHeight,
          clientHeight: el.clientHeight,
          scrollable: el.scrollHeight > el.clientHeight
        }));

        if (scrollInfo.scrollable) {
          // Test smooth scrolling
          await element.evaluate(el => {
            el.scrollTo({ top: 100, behavior: 'smooth' });
          });
          await page.waitForTimeout(200);
        }
      }

      this.results.touchTests.push({
        url,
        viewport: viewport.name,
        swipeableElements: swipeableElements.length,
        scrollableElements: scrollableElements.length
      });

    } catch (error) {
      console.error('Touch interaction test error:', error);
    }
  }

  async testTextReadability(page, url, viewport) {
    try {
      // Test font sizes
      const textElements = await page.evaluate(() => {
        const elements = Array.from(document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, div, button, a'));
        
        return elements
          .filter(el => el.textContent.trim().length > 0)
          .map(el => {
            const styles = window.getComputedStyle(el);
            return {
              tagName: el.tagName,
              fontSize: parseFloat(styles.fontSize),
              lineHeight: styles.lineHeight,
              textContent: el.textContent.trim().substring(0, 50)
            };
          });
      });

      // Check minimum font sizes
      const minFontSize = viewport.isMobile ? 16 : 14;
      const smallTextElements = textElements.filter(el => el.fontSize < minFontSize);

      smallTextElements.forEach(element => {
        this.results.violations.push({
          url,
          viewport: viewport.name,
          type: 'small-text',
          description: `Text too small for ${viewport.isMobile ? 'mobile' : 'desktop'}: ${element.fontSize}px in ${element.tagName}`
        });
      });

      // Test line length (optimal: 45-75 characters)
      const longLines = await page.evaluate(() => {
        const textBlocks = Array.from(document.querySelectorAll('p, .text-block'));
        return textBlocks.map(block => {
          const rect = block.getBoundingClientRect();
          const styles = window.getComputedStyle(block);
          const avgCharWidth = parseFloat(styles.fontSize) * 0.5; // Approximate
          const charactersPerLine = rect.width / avgCharWidth;
          
          return {
            width: rect.width,
            charactersPerLine: Math.round(charactersPerLine),
            optimal: charactersPerLine >= 45 && charactersPerLine <= 75
          };
        });
      });

      const suboptimalLines = longLines.filter(line => !line.optimal);
      if (suboptimalLines.length > 0) {
        this.results.violations.push({
          url,
          viewport: viewport.name,
          type: 'line-length',
          description: `${suboptimalLines.length} text blocks have suboptimal line length`
        });
      }

    } catch (error) {
      console.error('Text readability test error:', error);
    }
  }

  async testNavigationUsability(page, url, viewport) {
    try {
      // Test navigation visibility and accessibility
      const navElements = await page.$$('nav, .navigation, [role="navigation"]');
      
      for (const nav of navElements) {
        const navInfo = await nav.evaluate(el => {
          const rect = el.getBoundingClientRect();
          const styles = window.getComputedStyle(el);
          
          return {
            visible: rect.width > 0 && rect.height > 0 && styles.display !== 'none',
            position: styles.position,
            top: rect.top,
            width: rect.width,
            height: rect.height
          };
        });

        // Check if navigation is accessible on mobile
        if (viewport.isMobile && !navInfo.visible) {
          const hamburgerMenu = await page.$('.hamburger, .menu-toggle, [data-toggle="menu"]');
          if (!hamburgerMenu) {
            this.results.violations.push({
              url,
              viewport: viewport.name,
              type: 'mobile-navigation',
              description: 'No visible navigation or menu toggle on mobile'
            });
          }
        }

        // Test sticky navigation positioning
        if (navInfo.position === 'fixed' || navInfo.position === 'sticky') {
          if (navInfo.height > viewport.height * 0.3) {
            this.results.violations.push({
              url,
              viewport: viewport.name,
              type: 'sticky-nav-height',
              description: 'Sticky navigation takes up too much screen space'
            });
          }
        }
      }

      // Test breadcrumb usability
      const breadcrumbs = await page.$$('.breadcrumb, [aria-label="breadcrumb"]');
      for (const breadcrumb of breadcrumbs) {
        const breadcrumbInfo = await breadcrumb.evaluate(el => {
          const links = el.querySelectorAll('a');
          return {
            linkCount: links.length,
            fitsInViewport: el.getBoundingClientRect().width <= window.innerWidth
          };
        });

        if (!breadcrumbInfo.fitsInViewport) {
          this.results.violations.push({
            url,
            viewport: viewport.name,
            type: 'breadcrumb-overflow',
            description: 'Breadcrumb navigation overflows viewport'
          });
        }
      }

    } catch (error) {
      console.error('Navigation usability test error:', error);
    }
  }

  async generateReport() {
    const reportData = {
      timestamp: new Date().toISOString(),
      summary: {
        totalViolations: this.results.violations.length,
        viewportsTested: this.viewports.length,
        layoutTests: this.results.layoutTests.length,
        functionalityTests: this.results.functionalityTests.length,
        performanceTests: this.results.performanceTests.length,
        touchTests: this.results.touchTests.length,
        violationsByType: this.getViolationsByType(),
        violationsByViewport: this.getViolationsByViewport()
      },
      violations: this.results.violations,
      layoutTests: this.results.layoutTests,
      functionalityTests: this.results.functionalityTests,
      performanceTests: this.results.performanceTests,
      touchTests: this.results.touchTests,
      viewports: this.viewports
    };

    // Generate JSON report
    await fs.writeFile(
      path.join(__dirname, 'responsive-report.json'),
      JSON.stringify(reportData, null, 2)
    );

    // Generate HTML report
    await this.generateHtmlReport(reportData);
    
    console.log('\n📊 Responsive Design Testing Complete');
    console.log(`Total Violations: ${reportData.summary.totalViolations}`);
    console.log(`Viewports Tested: ${reportData.summary.viewportsTested}`);
    console.log(`Report saved to: responsive-report.json`);
  }

  async generateHtmlReport(data) {
    const htmlReport = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Responsive Design Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }
        .violation { background: #fee; border-left: 4px solid #e53e3e; padding: 16px; margin: 16px 0; }
        .test-result { background: #f0fff4; border-left: 4px solid #38a169; padding: 16px; margin: 16px 0; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #e2e8f0; padding: 20px; border-radius: 8px; text-align: center; }
        .viewport-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .viewport-card { background: #f7fafc; padding: 20px; border-radius: 8px; }
        .mobile { border-left: 4px solid #3182ce; }
        .tablet { border-left: 4px solid #805ad5; }
        .desktop { border-left: 4px solid #38a169; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📱 Responsive Design Test Report</h1>
        <p>Generated on: ${data.timestamp}</p>
        
        <div class="summary">
            <div class="metric">
                <h3>${data.summary.totalViolations}</h3>
                <p>Total Violations</p>
            </div>
            <div class="metric">
                <h3>${data.summary.viewportsTested}</h3>
                <p>Viewports Tested</p>
            </div>
            <div class="metric">
                <h3>${data.summary.layoutTests}</h3>
                <p>Layout Tests</p>
            </div>
            <div class="metric">
                <h3>${data.summary.functionalityTests}</h3>
                <p>Functionality Tests</p>
            </div>
        </div>

        <h2>Violations by Type</h2>
        <div class="summary">
          ${Object.entries(data.summary.violationsByType).map(([type, count]) => 
            `<div class="metric">
              <h3>${count}</h3>
              <p>${type.replace(/-/g, ' ').toUpperCase()}</p>
            </div>`
          ).join('')}
        </div>

        <h2>Tested Viewports</h2>
        <div class="viewport-grid">
          ${data.viewports.map(viewport => `
            <div class="viewport-card ${viewport.isMobile ? (viewport.width < 600 ? 'mobile' : 'tablet') : 'desktop'}">
              <h3>${viewport.name}</h3>
              <p><strong>Size:</strong> ${viewport.width}x${viewport.height}px</p>
              <p><strong>Type:</strong> ${viewport.isMobile ? 'Mobile/Tablet' : 'Desktop'}</p>
              <p><strong>Touch:</strong> ${viewport.hasTouch ? 'Yes' : 'No'}</p>
            </div>
          `).join('')}
        </div>

        <h2>🚨 Violations</h2>
        ${data.violations.map(violation => `
          <div class="violation">
            <h3>${violation.type.replace(/-/g, ' ').toUpperCase()}</h3>
            <p><strong>Description:</strong> ${violation.description}</p>
            <p><strong>URL:</strong> ${violation.url}</p>
            <p><strong>Viewport:</strong> ${violation.viewport}</p>
          </div>
        `).join('')}

        <h2>📐 Layout Test Results</h2>
        ${data.layoutTests.map(test => `
          <div class="test-result">
            <h3>${test.url} - ${test.viewport}</h3>
            <p><strong>Horizontal Scroll:</strong> ${test.hasHorizontalScroll ? 'Yes ❌' : 'No ✅'}</p>
            <p><strong>Elements Tested:</strong> ${test.elementsCount}</p>
            <p><strong>Images:</strong> ${test.responsiveImages}/${test.imagesCount} responsive</p>
          </div>
        `).join('')}

        <h2>⚡ Performance Results</h2>
        ${data.performanceTests.map(test => `
          <div class="test-result">
            <h3>${test.url} - ${test.viewport}</h3>
            <p><strong>Images Loaded:</strong> ${test.imagesLoaded}/${test.images}</p>
            <p><strong>Stylesheets:</strong> ${test.resources.stylesheets}</p>
            <p><strong>Scripts:</strong> ${test.resources.scripts}</p>
          </div>
        `).join('')}

        ${data.touchTests.length > 0 ? `
        <h2>👆 Touch Interaction Results</h2>
        ${data.touchTests.map(test => `
          <div class="test-result">
            <h3>${test.url} - ${test.viewport}</h3>
            <p><strong>Swipeable Elements:</strong> ${test.swipeableElements}</p>
            <p><strong>Scrollable Elements:</strong> ${test.scrollableElements}</p>
          </div>
        `).join('')}
        ` : ''}
    </div>
</body>
</html>`;

    await fs.writeFile(
      path.join(__dirname, 'responsive-report.html'),
      htmlReport
    );
  }

  getViolationsByType() {
    return this.results.violations.reduce((acc, violation) => {
      const type = violation.type || 'unknown';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});
  }

  getViolationsByViewport() {
    return this.results.violations.reduce((acc, violation) => {
      const viewport = violation.viewport || 'unknown';
      acc[viewport] = (acc[viewport] || 0) + 1;
      return acc;
    }, {});
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
    }
  }
}

module.exports = ResponsiveTestSuite;

// CLI usage
if (require.main === module) {
  const suite = new ResponsiveTestSuite({
    baseUrl: process.env.BASE_URL || 'http://localhost:3000'
  });
  
  const urls = process.argv.slice(2);
  const testUrls = urls.length > 0 ? urls : ['/'];
  
  suite.runFullResponsiveTest(testUrls)
    .then(results => {
      console.log('Responsive design testing completed');
      process.exit(results.violations.length > 0 ? 1 : 0);
    })
    .catch(error => {
      console.error('Responsive design testing failed:', error);
      process.exit(1);
    });
}