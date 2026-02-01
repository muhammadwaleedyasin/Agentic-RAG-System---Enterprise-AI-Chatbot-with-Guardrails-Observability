/**
 * WCAG 2.1 AA Compliance Validator
 * Comprehensive testing for WCAG 2.1 AA compliance including manual and automated checks
 */

const puppeteer = require('puppeteer');
const { AxePuppeteer } = require('@axe-core/puppeteer');
const fs = require('fs').promises;
const path = require('path');

class WCAGComplianceValidator {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:3000';
    this.browser = null;
    this.results = {
      complianceScore: 0,
      violations: [],
      passes: [],
      guidelines: {
        perceivable: { score: 0, violations: [], passes: [] },
        operable: { score: 0, violations: [], passes: [] },
        understandable: { score: 0, violations: [], passes: [] },
        robust: { score: 0, violations: [], passes: [] }
      },
      criteriaResults: {},
      manualChecks: []
    };
    
    // WCAG 2.1 AA Success Criteria mapping
    this.wcagCriteria = {
      // Perceivable
      '1.1.1': { level: 'A', name: 'Non-text Content', guideline: 'perceivable' },
      '1.2.1': { level: 'A', name: 'Audio-only and Video-only (Prerecorded)', guideline: 'perceivable' },
      '1.2.2': { level: 'A', name: 'Captions (Prerecorded)', guideline: 'perceivable' },
      '1.2.3': { level: 'A', name: 'Audio Description or Media Alternative (Prerecorded)', guideline: 'perceivable' },
      '1.2.4': { level: 'AA', name: 'Captions (Live)', guideline: 'perceivable' },
      '1.2.5': { level: 'AA', name: 'Audio Description (Prerecorded)', guideline: 'perceivable' },
      '1.3.1': { level: 'A', name: 'Info and Relationships', guideline: 'perceivable' },
      '1.3.2': { level: 'A', name: 'Meaningful Sequence', guideline: 'perceivable' },
      '1.3.3': { level: 'A', name: 'Sensory Characteristics', guideline: 'perceivable' },
      '1.3.4': { level: 'AA', name: 'Orientation', guideline: 'perceivable' },
      '1.3.5': { level: 'AA', name: 'Identify Input Purpose', guideline: 'perceivable' },
      '1.4.1': { level: 'A', name: 'Use of Color', guideline: 'perceivable' },
      '1.4.2': { level: 'A', name: 'Audio Control', guideline: 'perceivable' },
      '1.4.3': { level: 'AA', name: 'Contrast (Minimum)', guideline: 'perceivable' },
      '1.4.4': { level: 'AA', name: 'Resize text', guideline: 'perceivable' },
      '1.4.5': { level: 'AA', name: 'Images of Text', guideline: 'perceivable' },
      '1.4.10': { level: 'AA', name: 'Reflow', guideline: 'perceivable' },
      '1.4.11': { level: 'AA', name: 'Non-text Contrast', guideline: 'perceivable' },
      '1.4.12': { level: 'AA', name: 'Text Spacing', guideline: 'perceivable' },
      '1.4.13': { level: 'AA', name: 'Content on Hover or Focus', guideline: 'perceivable' },
      
      // Operable
      '2.1.1': { level: 'A', name: 'Keyboard', guideline: 'operable' },
      '2.1.2': { level: 'A', name: 'No Keyboard Trap', guideline: 'operable' },
      '2.1.4': { level: 'AA', name: 'Character Key Shortcuts', guideline: 'operable' },
      '2.2.1': { level: 'A', name: 'Timing Adjustable', guideline: 'operable' },
      '2.2.2': { level: 'A', name: 'Pause, Stop, Hide', guideline: 'operable' },
      '2.3.1': { level: 'A', name: 'Three Flashes or Below Threshold', guideline: 'operable' },
      '2.4.1': { level: 'A', name: 'Bypass Blocks', guideline: 'operable' },
      '2.4.2': { level: 'A', name: 'Page Titled', guideline: 'operable' },
      '2.4.3': { level: 'A', name: 'Focus Order', guideline: 'operable' },
      '2.4.4': { level: 'A', name: 'Link Purpose (In Context)', guideline: 'operable' },
      '2.4.5': { level: 'AA', name: 'Multiple Ways', guideline: 'operable' },
      '2.4.6': { level: 'AA', name: 'Headings and Labels', guideline: 'operable' },
      '2.4.7': { level: 'AA', name: 'Focus Visible', guideline: 'operable' },
      '2.5.1': { level: 'A', name: 'Pointer Gestures', guideline: 'operable' },
      '2.5.2': { level: 'A', name: 'Pointer Cancellation', guideline: 'operable' },
      '2.5.3': { level: 'A', name: 'Label in Name', guideline: 'operable' },
      '2.5.4': { level: 'A', name: 'Motion Actuation', guideline: 'operable' },
      
      // Understandable
      '3.1.1': { level: 'A', name: 'Language of Page', guideline: 'understandable' },
      '3.1.2': { level: 'AA', name: 'Language of Parts', guideline: 'understandable' },
      '3.2.1': { level: 'A', name: 'On Focus', guideline: 'understandable' },
      '3.2.2': { level: 'A', name: 'On Input', guideline: 'understandable' },
      '3.2.3': { level: 'AA', name: 'Consistent Navigation', guideline: 'understandable' },
      '3.2.4': { level: 'AA', name: 'Consistent Identification', guideline: 'understandable' },
      '3.3.1': { level: 'A', name: 'Error Identification', guideline: 'understandable' },
      '3.3.2': { level: 'A', name: 'Labels or Instructions', guideline: 'understandable' },
      '3.3.3': { level: 'AA', name: 'Error Suggestion', guideline: 'understandable' },
      '3.3.4': { level: 'AA', name: 'Error Prevention (Legal, Financial, Data)', guideline: 'understandable' },
      
      // Robust
      '4.1.1': { level: 'A', name: 'Parsing', guideline: 'robust' },
      '4.1.2': { level: 'A', name: 'Name, Role, Value', guideline: 'robust' },
      '4.1.3': { level: 'AA', name: 'Status Messages', guideline: 'robust' }
    };
  }

  async initialize() {
    this.browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage'
      ]
    });
  }

  async runWCAGComplianceTest(urls = ['/']) {
    await this.initialize();
    
    console.log('📋 Running WCAG 2.1 AA Compliance Test...');
    
    for (const url of urls) {
      console.log(`\n🔍 Testing WCAG compliance for: ${url}`);
      await this.testUrlCompliance(url);
    }
    
    await this.calculateComplianceScores();
    await this.generateComplianceReport();
    await this.cleanup();
    
    return this.results;
  }

  async testUrlCompliance(url) {
    const page = await this.browser.newPage();
    
    try {
      await page.goto(`${this.baseUrl}${url}`, { waitUntil: 'networkidle0' });
      
      // Run axe-core automated tests
      await this.runAxeTests(page, url);
      
      // Test specific WCAG criteria
      await this.testPerceivableCriteria(page, url);
      await this.testOperableCriteria(page, url);
      await this.testUnderstandableCriteria(page, url);
      await this.testRobustCriteria(page, url);
      
      // Run manual checks that require heuristics
      await this.runManualChecks(page, url);
      
    } catch (error) {
      console.error(`Error testing ${url}:`, error);
    } finally {
      await page.close();
    }
  }

  async runAxeTests(page, url) {
    try {
      const axeResults = await new AxePuppeteer(page)
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .analyze();

      // Process and categorize violations by WCAG guideline
      axeResults.violations.forEach(violation => {
        const guideline = this.categorizeViolationByGuideline(violation.id);
        
        const violationData = {
          url,
          rule: violation.id,
          impact: violation.impact,
          description: violation.description,
          help: violation.help,
          helpUrl: violation.helpUrl,
          guideline,
          wcagCriteria: this.mapRuleToWCAGCriteria(violation.id),
          nodes: violation.nodes.map(node => ({
            html: node.html,
            target: node.target,
            failureSummary: node.failureSummary
          }))
        };
        
        this.results.violations.push(violationData);
        this.results.guidelines[guideline].violations.push(violationData);
      });

      // Process passes
      axeResults.passes.forEach(pass => {
        const guideline = this.categorizeViolationByGuideline(pass.id);
        
        const passData = {
          url,
          rule: pass.id,
          description: pass.description,
          guideline,
          wcagCriteria: this.mapRuleToWCAGCriteria(pass.id),
          nodes: pass.nodes.length
        };
        
        this.results.passes.push(passData);
        this.results.guidelines[guideline].passes.push(passData);
      });

    } catch (error) {
      console.error('Axe test error:', error);
    }
  }

  async testPerceivableCriteria(page, url) {
    // 1.1.1 - Non-text Content
    await this.test_1_1_1_NonTextContent(page, url);
    
    // 1.3.1 - Info and Relationships
    await this.test_1_3_1_InfoAndRelationships(page, url);
    
    // 1.3.2 - Meaningful Sequence
    await this.test_1_3_2_MeaningfulSequence(page, url);
    
    // 1.4.1 - Use of Color
    await this.test_1_4_1_UseOfColor(page, url);
    
    // 1.4.3 - Contrast (Minimum)
    await this.test_1_4_3_Contrast(page, url);
    
    // 1.4.4 - Resize text
    await this.test_1_4_4_ResizeText(page, url);
    
    // 1.4.10 - Reflow
    await this.test_1_4_10_Reflow(page, url);
    
    // 1.4.11 - Non-text Contrast
    await this.test_1_4_11_NonTextContrast(page, url);
    
    // 1.4.12 - Text Spacing
    await this.test_1_4_12_TextSpacing(page, url);
  }

  async test_1_1_1_NonTextContent(page, url) {
    const images = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('img')).map(img => ({
        src: img.src,
        alt: img.alt,
        hasAlt: img.hasAttribute('alt'),
        isDecorative: img.alt === '',
        ariaHidden: img.getAttribute('aria-hidden') === 'true',
        role: img.getAttribute('role'),
        visible: img.offsetParent !== null
      }));
    });

    images.forEach(img => {
      if (img.visible && !img.hasAlt && !img.ariaHidden) {
        this.addCriteriaViolation('1.1.1', url, {
          description: 'Image missing alt attribute',
          element: `img[src="${img.src}"]`
        });
      } else if (img.hasAlt) {
        this.addCriteriaPass('1.1.1', url);
      }
    });
  }

  async test_1_3_1_InfoAndRelationships(page, url) {
    // Test form labels
    const formIssues = await page.evaluate(() => {
      const inputs = Array.from(document.querySelectorAll('input, textarea, select'));
      const issues = [];
      
      inputs.forEach(input => {
        const hasLabel = input.labels && input.labels.length > 0;
        const hasAriaLabel = input.getAttribute('aria-label');
        const hasAriaLabelledby = input.getAttribute('aria-labelledby');
        
        if (!hasLabel && !hasAriaLabel && !hasAriaLabelledby) {
          issues.push({
            element: input.tagName + (input.id ? '#' + input.id : ''),
            issue: 'No accessible label'
          });
        }
      });
      
      return issues;
    });

    formIssues.forEach(issue => {
      this.addCriteriaViolation('1.3.1', url, issue);
    });

    if (formIssues.length === 0) {
      this.addCriteriaPass('1.3.1', url);
    }
  }

  async test_1_3_2_MeaningfulSequence(page, url) {
    const readingOrder = await page.evaluate(() => {
      const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
      const issues = [];
      
      let previousLevel = 0;
      headings.forEach(heading => {
        const level = parseInt(heading.tagName.charAt(1));
        
        if (level > previousLevel + 1) {
          issues.push({
            element: heading.tagName,
            text: heading.textContent.trim().substring(0, 50),
            issue: `Heading level skipped from h${previousLevel} to h${level}`
          });
        }
        
        previousLevel = level;
      });
      
      return issues;
    });

    readingOrder.forEach(issue => {
      this.addCriteriaViolation('1.3.2', url, issue);
    });

    if (readingOrder.length === 0) {
      this.addCriteriaPass('1.3.2', url);
    }
  }

  async test_1_4_1_UseOfColor(page, url) {
    // This is a complex test that would require analyzing color usage
    // For now, we'll mark it as a manual check
    this.addManualCheck('1.4.1', url, {
      description: 'Verify that color is not the only means of conveying information',
      instructions: 'Check if information conveyed by color is also available through other means (text, patterns, etc.)'
    });
  }

  async test_1_4_3_Contrast(page, url) {
    // Color contrast testing would require sophisticated color analysis
    // Axe-core handles this, but we can add additional manual checks
    this.addManualCheck('1.4.3', url, {
      description: 'Verify color contrast ratios meet WCAG AA standards',
      instructions: 'Check that text has at least 4.5:1 contrast ratio (3:1 for large text)'
    });
  }

  async test_1_4_4_ResizeText(page, url) {
    // Test if text can be resized up to 200% without loss of functionality
    const originalSize = await page.evaluate(() => {
      const body = document.body;
      return window.getComputedStyle(body).fontSize;
    });

    // Simulate 200% zoom
    await page.evaluate(() => {
      document.body.style.fontSize = '200%';
    });

    await page.waitForTimeout(500);

    const hasHorizontalScroll = await page.evaluate(() => {
      return document.body.scrollWidth > window.innerWidth;
    });

    const hasOverflowingElements = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      return Array.from(elements).some(el => {
        const rect = el.getBoundingClientRect();
        return rect.right > window.innerWidth;
      });
    });

    // Reset zoom
    await page.evaluate(() => {
      document.body.style.fontSize = '';
    });

    if (hasHorizontalScroll || hasOverflowingElements) {
      this.addCriteriaViolation('1.4.4', url, {
        description: 'Content overflows or requires horizontal scrolling at 200% zoom'
      });
    } else {
      this.addCriteriaPass('1.4.4', url);
    }
  }

  async test_1_4_10_Reflow(page, url) {
    // Test content reflow at 320px width (mobile viewport)
    await page.setViewport({ width: 320, height: 568 });
    await page.waitForTimeout(500);

    const hasHorizontalScroll = await page.evaluate(() => {
      return document.body.scrollWidth > window.innerWidth;
    });

    if (hasHorizontalScroll) {
      this.addCriteriaViolation('1.4.10', url, {
        description: 'Content requires horizontal scrolling at 320px width'
      });
    } else {
      this.addCriteriaPass('1.4.10', url);
    }

    // Reset viewport
    await page.setViewport({ width: 1200, height: 800 });
  }

  async test_1_4_11_NonTextContrast(page, url) {
    this.addManualCheck('1.4.11', url, {
      description: 'Verify non-text elements have sufficient contrast',
      instructions: 'Check that UI components and graphics have at least 3:1 contrast ratio'
    });
  }

  async test_1_4_12_TextSpacing(page, url) {
    // Test text spacing adjustments
    await page.addStyleTag({
      content: `
        * {
          line-height: 1.5 !important;
          letter-spacing: 0.12em !important;
          word-spacing: 0.16em !important;
        }
        p {
          margin-bottom: 2em !important;
        }
      `
    });

    await page.waitForTimeout(500);

    const hasOverflow = await page.evaluate(() => {
      const elements = document.querySelectorAll('p, div, span');
      return Array.from(elements).some(el => {
        const rect = el.getBoundingClientRect();
        return rect.right > window.innerWidth || el.scrollWidth > el.clientWidth;
      });
    });

    if (hasOverflow) {
      this.addCriteriaViolation('1.4.12', url, {
        description: 'Content overflows when text spacing is adjusted'
      });
    } else {
      this.addCriteriaPass('1.4.12', url);
    }
  }

  async testOperableCriteria(page, url) {
    // 2.1.1 - Keyboard
    await this.test_2_1_1_Keyboard(page, url);
    
    // 2.1.2 - No Keyboard Trap
    await this.test_2_1_2_NoKeyboardTrap(page, url);
    
    // 2.4.1 - Bypass Blocks
    await this.test_2_4_1_BypassBlocks(page, url);
    
    // 2.4.2 - Page Titled
    await this.test_2_4_2_PageTitled(page, url);
    
    // 2.4.3 - Focus Order
    await this.test_2_4_3_FocusOrder(page, url);
    
    // 2.4.7 - Focus Visible
    await this.test_2_4_7_FocusVisible(page, url);
  }

  async test_2_1_1_Keyboard(page, url) {
    const keyboardAccessible = await page.evaluate(() => {
      const interactiveElements = Array.from(document.querySelectorAll(
        'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
      ));
      
      return interactiveElements.map(el => ({
        tagName: el.tagName,
        tabIndex: el.tabIndex,
        disabled: el.disabled,
        id: el.id || null,
        accessible: el.tabIndex >= 0 || ['A', 'BUTTON', 'INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)
      }));
    });

    const inaccessibleElements = keyboardAccessible.filter(el => !el.accessible);
    
    inaccessibleElements.forEach(el => {
      this.addCriteriaViolation('2.1.1', url, {
        description: `${el.tagName} element not keyboard accessible`,
        element: el.id ? `${el.tagName}#${el.id}` : el.tagName
      });
    });

    if (inaccessibleElements.length === 0) {
      this.addCriteriaPass('2.1.1', url);
    }
  }

  async test_2_1_2_NoKeyboardTrap(page, url) {
    // Test for keyboard traps by tabbing through elements
    let trapDetected = false;
    const maxTabs = 50;
    let previousFocus = null;
    let sameElementCount = 0;

    for (let i = 0; i < maxTabs; i++) {
      await page.keyboard.press('Tab');
      await page.waitForTimeout(50);
      
      const currentFocus = await page.evaluate(() => {
        const el = document.activeElement;
        return el ? {
          tagName: el.tagName,
          id: el.id,
          className: el.className
        } : null;
      });

      if (previousFocus && 
          currentFocus &&
          JSON.stringify(currentFocus) === JSON.stringify(previousFocus)) {
        sameElementCount++;
        if (sameElementCount > 3) {
          trapDetected = true;
          break;
        }
      } else {
        sameElementCount = 0;
      }

      previousFocus = currentFocus;
    }

    if (trapDetected) {
      this.addCriteriaViolation('2.1.2', url, {
        description: 'Keyboard trap detected - focus cannot move away from element'
      });
    } else {
      this.addCriteriaPass('2.1.2', url);
    }
  }

  async test_2_4_1_BypassBlocks(page, url) {
    const skipLinks = await page.evaluate(() => {
      const links = Array.from(document.querySelectorAll('a[href^="#"]'));
      return links.filter(link => {
        const text = link.textContent.toLowerCase();
        return text.includes('skip') || text.includes('jump') || text.includes('main');
      }).map(link => ({
        text: link.textContent.trim(),
        href: link.href,
        visible: link.offsetParent !== null
      }));
    });

    if (skipLinks.length === 0) {
      this.addCriteriaViolation('2.4.1', url, {
        description: 'No skip links found to bypass repetitive content'
      });
    } else {
      this.addCriteriaPass('2.4.1', url);
    }
  }

  async test_2_4_2_PageTitled(page, url) {
    const title = await page.title();
    
    if (!title || title.trim().length === 0) {
      this.addCriteriaViolation('2.4.2', url, {
        description: 'Page has no title or empty title'
      });
    } else if (title.length < 3) {
      this.addCriteriaViolation('2.4.2', url, {
        description: 'Page title is too short to be descriptive'
      });
    } else {
      this.addCriteriaPass('2.4.2', url);
    }
  }

  async test_2_4_3_FocusOrder(page, url) {
    // Test logical focus order
    const focusOrder = [];
    const focusableElements = await page.$$('a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])');
    
    for (const element of focusableElements.slice(0, 10)) {
      await element.focus();
      const position = await element.evaluate(el => {
        const rect = el.getBoundingClientRect();
        return { top: rect.top, left: rect.left };
      });
      focusOrder.push(position);
    }
    
    // Check if focus order follows visual order (top to bottom, left to right)
    let logicalOrder = true;
    for (let i = 1; i < focusOrder.length; i++) {
      const prev = focusOrder[i - 1];
      const curr = focusOrder[i];
      
      if (curr.top < prev.top - 10 || (Math.abs(curr.top - prev.top) < 10 && curr.left < prev.left - 10)) {
        logicalOrder = false;
        break;
      }
    }
    
    if (!logicalOrder) {
      this.addCriteriaViolation('2.4.3', url, {
        description: 'Focus order does not follow logical visual order'
      });
    } else {
      this.addCriteriaPass('2.4.3', url);
    }
  }

  async test_2_4_7_FocusVisible(page, url) {
    const focusableElements = await page.$$('a, button, input, textarea, select');
    let visibleFocusCount = 0;
    
    for (const element of focusableElements.slice(0, 5)) {
      await element.focus();
      
      const hasVisibleFocus = await element.evaluate(el => {
        const styles = window.getComputedStyle(el, ':focus');
        return styles.outline !== 'none' || 
               styles.outlineWidth !== '0px' ||
               styles.boxShadow !== 'none';
      });
      
      if (hasVisibleFocus) {
        visibleFocusCount++;
      }
    }
    
    if (visibleFocusCount === 0) {
      this.addCriteriaViolation('2.4.7', url, {
        description: 'No visible focus indicators found on interactive elements'
      });
    } else {
      this.addCriteriaPass('2.4.7', url);
    }
  }

  async testUnderstandableCriteria(page, url) {
    // 3.1.1 - Language of Page
    await this.test_3_1_1_LanguageOfPage(page, url);
    
    // 3.2.1 - On Focus
    await this.test_3_2_1_OnFocus(page, url);
    
    // 3.2.2 - On Input
    await this.test_3_2_2_OnInput(page, url);
    
    // 3.3.1 - Error Identification
    await this.test_3_3_1_ErrorIdentification(page, url);
    
    // 3.3.2 - Labels or Instructions
    await this.test_3_3_2_LabelsOrInstructions(page, url);
  }

  async test_3_1_1_LanguageOfPage(page, url) {
    const lang = await page.evaluate(() => {
      return document.documentElement.getAttribute('lang') || 
             document.documentElement.getAttribute('xml:lang');
    });
    
    if (!lang) {
      this.addCriteriaViolation('3.1.1', url, {
        description: 'Page language not specified in html element'
      });
    } else {
      this.addCriteriaPass('3.1.1', url);
    }
  }

  async test_3_2_1_OnFocus(page, url) {
    // Test that focusing elements doesn't cause unexpected context changes
    this.addManualCheck('3.2.1', url, {
      description: 'Verify that focusing elements does not cause unexpected context changes',
      instructions: 'Check that focusing on forms, links, or other controls does not automatically trigger navigation or form submission'
    });
  }

  async test_3_2_2_OnInput(page, url) {
    // Test that changing input values doesn't cause unexpected context changes
    this.addManualCheck('3.2.2', url, {
      description: 'Verify that changing input values does not cause unexpected context changes',
      instructions: 'Check that selecting options or entering data does not automatically trigger form submission or navigation'
    });
  }

  async test_3_3_1_ErrorIdentification(page, url) {
    // Look for forms and check if they have error handling
    const forms = await page.$$('form');
    
    if (forms.length > 0) {
      this.addManualCheck('3.3.1', url, {
        description: 'Verify that form errors are clearly identified',
        instructions: 'Test form validation and ensure errors are clearly described and associated with the relevant fields'
      });
    } else {
      this.addCriteriaPass('3.3.1', url);
    }
  }

  async test_3_3_2_LabelsOrInstructions(page, url) {
    const formElements = await page.evaluate(() => {
      const inputs = Array.from(document.querySelectorAll('input, textarea, select'));
      return inputs.map(input => ({
        type: input.type,
        hasLabel: input.labels && input.labels.length > 0,
        hasPlaceholder: input.placeholder && input.placeholder.length > 0,
        hasAriaLabel: input.getAttribute('aria-label'),
        required: input.required
      }));
    });
    
    const unlabeledRequired = formElements.filter(el => 
      el.required && !el.hasLabel && !el.hasAriaLabel
    );
    
    unlabeledRequired.forEach(el => {
      this.addCriteriaViolation('3.3.2', url, {
        description: `Required ${el.type} field lacks proper label or instructions`
      });
    });
    
    if (unlabeledRequired.length === 0) {
      this.addCriteriaPass('3.3.2', url);
    }
  }

  async testRobustCriteria(page, url) {
    // 4.1.1 - Parsing
    await this.test_4_1_1_Parsing(page, url);
    
    // 4.1.2 - Name, Role, Value
    await this.test_4_1_2_NameRoleValue(page, url);
    
    // 4.1.3 - Status Messages
    await this.test_4_1_3_StatusMessages(page, url);
  }

  async test_4_1_1_Parsing(page, url) {
    // HTML validation would require external service
    this.addManualCheck('4.1.1', url, {
      description: 'Verify HTML markup is valid',
      instructions: 'Validate HTML using W3C validator or similar tool'
    });
  }

  async test_4_1_2_NameRoleValue(page, url) {
    const customElements = await page.evaluate(() => {
      const elements = Array.from(document.querySelectorAll('[role], [aria-*]'));
      return elements.map(el => ({
        tagName: el.tagName,
        role: el.getAttribute('role'),
        ariaLabel: el.getAttribute('aria-label'),
        ariaLabelledby: el.getAttribute('aria-labelledby'),
        ariaDescribedby: el.getAttribute('aria-describedby'),
        hasName: el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') || el.textContent.trim()
      }));
    });
    
    const elementsWithoutName = customElements.filter(el => 
      el.role && !el.hasName && !['presentation', 'none'].includes(el.role)
    );
    
    elementsWithoutName.forEach(el => {
      this.addCriteriaViolation('4.1.2', url, {
        description: `Element with role="${el.role}" lacks accessible name`,
        element: `${el.tagName}[role="${el.role}"]`
      });
    });
    
    if (elementsWithoutName.length === 0) {
      this.addCriteriaPass('4.1.2', url);
    }
  }

  async test_4_1_3_StatusMessages(page, url) {
    const liveRegions = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('[aria-live], [role="status"], [role="alert"]')).length;
    });
    
    this.addManualCheck('4.1.3', url, {
      description: 'Verify status messages are programmatically announced',
      instructions: 'Check that important status changes (errors, success messages, loading states) are announced to screen readers',
      currentLiveRegions: liveRegions
    });
  }

  async runManualChecks(page, url) {
    // Additional manual checks that require human judgment
    
    // Check for auto-playing media
    const mediaElements = await page.$$('video, audio');
    if (mediaElements.length > 0) {
      this.addManualCheck('media_autoplay', url, {
        description: 'Check for auto-playing media content',
        instructions: 'Verify that audio/video content does not auto-play for more than 3 seconds'
      });
    }
    
    // Check for seizure-inducing content
    this.addManualCheck('seizures', url, {
      description: 'Check for content that may trigger seizures',
      instructions: 'Verify no content flashes more than 3 times per second'
    });
    
    // Check for consistent navigation
    this.addManualCheck('consistent_navigation', url, {
      description: 'Check navigation consistency across pages',
      instructions: 'Verify navigation elements appear in the same relative order across pages'
    });
  }

  addCriteriaViolation(criterion, url, details) {
    if (!this.results.criteriaResults[criterion]) {
      this.results.criteriaResults[criterion] = { passes: 0, violations: [] };
    }
    
    this.results.criteriaResults[criterion].violations.push({
      url,
      ...details
    });
  }

  addCriteriaPass(criterion, url) {
    if (!this.results.criteriaResults[criterion]) {
      this.results.criteriaResults[criterion] = { passes: 0, violations: [] };
    }
    
    this.results.criteriaResults[criterion].passes++;
  }

  addManualCheck(criterion, url, details) {
    this.results.manualChecks.push({
      criterion,
      url,
      ...details
    });
  }

  categorizeViolationByGuideline(ruleId) {
    // Map axe rules to WCAG guidelines
    const guidelineMap = {
      'color-contrast': 'perceivable',
      'image-alt': 'perceivable',
      'label': 'perceivable',
      'keyboard': 'operable',
      'focus-order': 'operable',
      'skip-link': 'operable',
      'page-has-heading-one': 'operable',
      'html-has-lang': 'understandable',
      'button-name': 'robust',
      'link-name': 'robust',
      'aria-valid-attr': 'robust'
    };
    
    for (const [key, guideline] of Object.entries(guidelineMap)) {
      if (ruleId.includes(key)) {
        return guideline;
      }
    }
    
    return 'robust'; // Default fallback
  }

  mapRuleToWCAGCriteria(ruleId) {
    // Map axe rules to specific WCAG criteria
    const criteriaMap = {
      'color-contrast': ['1.4.3'],
      'image-alt': ['1.1.1'],
      'label': ['1.3.1', '3.3.2'],
      'keyboard': ['2.1.1'],
      'focus-order': ['2.4.3'],
      'skip-link': ['2.4.1'],
      'html-has-lang': ['3.1.1'],
      'button-name': ['4.1.2'],
      'link-name': ['4.1.2']
    };
    
    for (const [key, criteria] of Object.entries(criteriaMap)) {
      if (ruleId.includes(key)) {
        return criteria;
      }
    }
    
    return [];
  }

  async calculateComplianceScores() {
    // Calculate scores for each guideline
    Object.keys(this.results.guidelines).forEach(guideline => {
      const data = this.results.guidelines[guideline];
      const totalChecks = data.violations.length + data.passes.length;
      
      if (totalChecks > 0) {
        data.score = Math.round((data.passes.length / totalChecks) * 100);
      } else {
        data.score = 100; // No tests means no violations
      }
    });
    
    // Calculate overall compliance score
    const guidelines = Object.values(this.results.guidelines);
    const avgScore = guidelines.reduce((sum, g) => sum + g.score, 0) / guidelines.length;
    this.results.complianceScore = Math.round(avgScore);
  }

  async generateComplianceReport() {
    const reportData = {
      timestamp: new Date().toISOString(),
      complianceScore: this.results.complianceScore,
      guidelines: this.results.guidelines,
      summary: {
        totalViolations: this.results.violations.length,
        totalPasses: this.results.passes.length,
        manualChecks: this.results.manualChecks.length,
        criteriaResults: Object.keys(this.results.criteriaResults).length,
        guidelineScores: Object.keys(this.results.guidelines).map(key => ({
          guideline: key,
          score: this.results.guidelines[key].score
        }))
      },
      violations: this.results.violations,
      passes: this.results.passes,
      criteriaResults: this.results.criteriaResults,
      manualChecks: this.results.manualChecks,
      recommendations: this.generateWCAGRecommendations()
    };

    // Save JSON report
    await fs.writeFile(
      path.join(__dirname, 'wcag-compliance-report.json'),
      JSON.stringify(reportData, null, 2)
    );

    // Generate HTML report
    await this.generateWCAGHtmlReport(reportData);
    
    console.log('\n📋 WCAG 2.1 AA Compliance Testing Complete');
    console.log(`Overall Compliance Score: ${reportData.complianceScore}/100`);
    console.log(`Violations: ${reportData.summary.totalViolations}`);
    console.log(`Manual Checks Required: ${reportData.summary.manualChecks}`);
    console.log(`Report saved to: wcag-compliance-report.json`);
  }

  generateWCAGRecommendations() {
    const recommendations = [];
    
    // Check overall compliance score
    if (this.results.complianceScore < 80) {
      recommendations.push({
        priority: 'high',
        category: 'overall_compliance',
        description: `WCAG 2.1 AA compliance score is ${this.results.complianceScore}/100`,
        actions: [
          'Address high-impact violations first',
          'Implement automated accessibility testing in CI/CD',
          'Provide accessibility training to development team',
          'Conduct regular accessibility audits'
        ]
      });
    }
    
    // Check individual guidelines
    Object.entries(this.results.guidelines).forEach(([guideline, data]) => {
      if (data.score < 70) {
        recommendations.push({
          priority: data.score < 50 ? 'high' : 'medium',
          category: `${guideline}_guideline`,
          description: `${guideline.charAt(0).toUpperCase() + guideline.slice(1)} guideline score is ${data.score}/100`,
          violations: data.violations.length,
          actions: this.getGuidelineSpecificActions(guideline)
        });
      }
    });
    
    // Check manual checks
    if (this.results.manualChecks.length > 5) {
      recommendations.push({
        priority: 'medium',
        category: 'manual_testing',
        description: `${this.results.manualChecks.length} manual checks required`,
        actions: [
          'Conduct manual accessibility testing',
          'Test with real assistive technologies',
          'Get feedback from users with disabilities',
          'Document manual testing procedures'
        ]
      });
    }
    
    return recommendations;
  }

  getGuidelineSpecificActions(guideline) {
    const actions = {
      perceivable: [
        'Add alt text to all images',
        'Ensure sufficient color contrast',
        'Provide text alternatives for non-text content',
        'Make content adaptable to different presentations'
      ],
      operable: [
        'Ensure all functionality is keyboard accessible',
        'Provide clear focus indicators',
        'Add skip links for navigation',
        'Use descriptive link text'
      ],
      understandable: [
        'Specify page language',
        'Provide clear form labels and instructions',
        'Make navigation consistent',
        'Provide clear error messages'
      ],
      robust: [
        'Use valid HTML markup',
        'Ensure compatibility with assistive technologies',
        'Provide proper ARIA labels and roles',
        'Implement status message announcements'
      ]
    };
    
    return actions[guideline] || [];
  }

  async generateWCAGHtmlReport(data) {
    const htmlReport = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WCAG 2.1 AA Compliance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }
        .score { font-size: 3em; font-weight: bold; text-align: center; margin: 20px 0; }
        .score-excellent { color: #38a169; }
        .score-good { color: #3182ce; }
        .score-fair { color: #d69e2e; }
        .score-poor { color: #e53e3e; }
        .guidelines { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .guideline-card { background: #f7fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #e2e8f0; }
        .violation { background: #fee; border-left: 4px solid #e53e3e; padding: 16px; margin: 16px 0; }
        .pass { background: #f0fff4; border-left: 4px solid #38a169; padding: 16px; margin: 16px 0; }
        .manual-check { background: #fffbf0; border-left: 4px solid #d69e2e; padding: 16px; margin: 16px 0; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #e2e8f0; padding: 20px; border-radius: 8px; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #f7fafc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📋 WCAG 2.1 AA Compliance Report</h1>
        <p>Generated on: ${data.timestamp}</p>
        
        <div class="score ${this.getScoreClass(data.complianceScore)}">
            ${data.complianceScore}/100
        </div>
        <p style="text-align: center; font-size: 1.2em;">Overall Compliance Score</p>
        
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
                <h3>${data.summary.manualChecks}</h3>
                <p>Manual Checks</p>
            </div>
            <div class="metric">
                <h3>${data.summary.criteriaResults}</h3>
                <p>Criteria Tested</p>
            </div>
        </div>

        <h2>📊 WCAG Guidelines Scores</h2>
        <div class="guidelines">
          ${Object.entries(data.guidelines).map(([guideline, guidelineData]) => `
            <div class="guideline-card">
              <h3>${guideline.charAt(0).toUpperCase() + guideline.slice(1)}</h3>
              <div class="score ${this.getScoreClass(guidelineData.score)}" style="font-size: 2em; margin: 10px 0;">
                ${guidelineData.score}/100
              </div>
              <p><strong>Violations:</strong> ${guidelineData.violations.length}</p>
              <p><strong>Passes:</strong> ${guidelineData.passes.length}</p>
            </div>
          `).join('')}
        </div>

        <h2>🚨 WCAG Violations</h2>
        ${data.violations.map(violation => `
          <div class="violation">
            <h3>${violation.rule} - ${violation.wcagCriteria?.join(', ') || 'N/A'}</h3>
            <p><strong>Impact:</strong> ${violation.impact}</p>
            <p><strong>Description:</strong> ${violation.description}</p>
            <p><strong>Help:</strong> <a href="${violation.helpUrl}" target="_blank">${violation.help}</a></p>
            <p><strong>Guideline:</strong> ${violation.guideline}</p>
            <details>
              <summary>Affected Elements (${violation.nodes?.length || 0})</summary>
              ${violation.nodes?.map(node => `
                <div style="margin: 10px 0; padding: 10px; background: #f7fafc;">
                  <strong>Target:</strong> ${node.target?.join(', ')}<br>
                  <strong>HTML:</strong> <code>${node.html}</code><br>
                  <strong>Issue:</strong> ${node.failureSummary}
                </div>
              `).join('') || ''}
            </details>
          </div>
        `).join('')}

        <h2>📝 Manual Checks Required</h2>
        ${data.manualChecks.map(check => `
          <div class="manual-check">
            <h3>${check.criterion} - ${check.url}</h3>
            <p><strong>Check:</strong> ${check.description}</p>
            <p><strong>Instructions:</strong> ${check.instructions}</p>
          </div>
        `).join('')}

        <h2>📈 Criteria Results Summary</h2>
        <table>
          <thead>
            <tr>
              <th>WCAG Criterion</th>
              <th>Level</th>
              <th>Name</th>
              <th>Passes</th>
              <th>Violations</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${Object.entries(data.criteriaResults).map(([criterion, result]) => {
              const wcagInfo = this.wcagCriteria[criterion];
              const status = result.violations.length === 0 ? 'Pass' : 'Fail';
              return `
                <tr>
                  <td>${criterion}</td>
                  <td>${wcagInfo?.level || 'N/A'}</td>
                  <td>${wcagInfo?.name || 'Unknown'}</td>
                  <td>${result.passes}</td>
                  <td>${result.violations.length}</td>
                  <td class="${status === 'Pass' ? 'score-good' : 'score-poor'}">${status}</td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>

        <h2>💡 Recommendations</h2>
        ${data.recommendations.map(rec => `
          <div class="violation">
            <h3>${rec.category.replace(/_/g, ' ').toUpperCase()} (${rec.priority.toUpperCase()} Priority)</h3>
            <p>${rec.description}</p>
            <h4>Recommended Actions:</h4>
            <ul>
              ${rec.actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
          </div>
        `).join('')}
    </div>
</body>
</html>`;

    await fs.writeFile(
      path.join(__dirname, 'wcag-compliance-report.html'),
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

module.exports = WCAGComplianceValidator;

// CLI usage
if (require.main === module) {
  const validator = new WCAGComplianceValidator({
    baseUrl: process.env.BASE_URL || 'http://localhost:3000'
  });
  
  const urls = process.argv.slice(2);
  const testUrls = urls.length > 0 ? urls : ['/'];
  
  validator.runWCAGComplianceTest(testUrls)
    .then(results => {
      console.log('WCAG compliance testing completed');
      process.exit(results.complianceScore < 70 ? 1 : 0);
    })
    .catch(error => {
      console.error('WCAG compliance testing failed:', error);
      process.exit(1);
    });
}