/**
 * Axe-core configuration for automated accessibility testing
 * WCAG 2.1 AA compliance testing configuration
 */

const axeConfig = {
  // Rules to run - all WCAG 2.1 AA rules
  rules: {
    // Color and contrast
    'color-contrast': { enabled: true },
    'color-contrast-enhanced': { enabled: true },
    
    // Focus and keyboard navigation
    'focus-order-semantics': { enabled: true },
    'focusable-content': { enabled: true },
    'keyboard': { enabled: true },
    'tabindex': { enabled: true },
    
    // Form accessibility
    'label': { enabled: true },
    'form-field-multiple-labels': { enabled: true },
    'fieldset': { enabled: true },
    'input-button-name': { enabled: true },
    'input-image-alt': { enabled: true },
    
    // Images and media
    'image-alt': { enabled: true },
    'object-alt': { enabled: true },
    'video-caption': { enabled: true },
    'audio-caption': { enabled: true },
    
    // Semantic structure
    'page-has-heading-one': { enabled: true },
    'heading-order': { enabled: true },
    'landmark-banner-is-top-level': { enabled: true },
    'landmark-contentinfo-is-top-level': { enabled: true },
    'landmark-main-is-top-level': { enabled: true },
    'landmark-no-duplicate-banner': { enabled: true },
    'landmark-no-duplicate-contentinfo': { enabled: true },
    'landmark-no-duplicate-main': { enabled: true },
    'landmark-one-main': { enabled: true },
    'landmark-unique': { enabled: true },
    'region': { enabled: true },
    
    // Links and buttons
    'button-name': { enabled: true },
    'link-name': { enabled: true },
    'link-in-text-block': { enabled: true },
    
    // Tables
    'table-duplicate-name': { enabled: true },
    'table-fake-caption': { enabled: true },
    'td-has-header': { enabled: true },
    'td-headers-attr': { enabled: true },
    'th-has-data-cells': { enabled: true },
    'scope-attr-valid': { enabled: true },
    
    // Language
    'html-has-lang': { enabled: true },
    'html-lang-valid': { enabled: true },
    'valid-lang': { enabled: true },
    
    // ARIA
    'aria-allowed-attr': { enabled: true },
    'aria-allowed-role': { enabled: true },
    'aria-dpub-role-fallback': { enabled: true },
    'aria-hidden-body': { enabled: true },
    'aria-hidden-focus': { enabled: true },
    'aria-input-field-name': { enabled: true },
    'aria-label': { enabled: true },
    'aria-labelledby': { enabled: true },
    'aria-required-attr': { enabled: true },
    'aria-required-children': { enabled: true },
    'aria-required-parent': { enabled: true },
    'aria-roledescription': { enabled: true },
    'aria-roles': { enabled: true },
    'aria-toggle-field-name': { enabled: true },
    'aria-valid-attr': { enabled: true },
    'aria-valid-attr-value': { enabled: true },
    
    // Document structure
    'document-title': { enabled: true },
    'html-xml-lang-mismatch': { enabled: true },
    'meta-refresh': { enabled: true },
    'meta-viewport': { enabled: true },
    'nested-interactive': { enabled: true },
    'no-autoplay-audio': { enabled: true },
    'role-img-alt': { enabled: true },
    'scrollable-region-focusable': { enabled: true },
    'server-side-image-map': { enabled: true },
    'skip-link': { enabled: true }
  },
  
  // Tags to include
  tags: [
    'wcag2a',
    'wcag2aa',
    'wcag21a',
    'wcag21aa',
    'best-practice'
  ],
  
  // Selectors to exclude from testing
  exclude: [
    '#skip-links', // Skip links are typically hidden
    '.sr-only' // Screen reader only content
  ],
  
  // Result types to report
  resultTypes: ['violations', 'incomplete', 'passes'],
  
  // Disable rules that might cause false positives
  disableOtherRules: false,
  
  // Timeout for running tests
  timeout: 30000,
  
  // Options for specific rules
  options: {
    'color-contrast': {
      // Ignore elements that are not visible
      ignoreUseOfColorAlone: false,
      ignoreLength: true
    },
    'image-alt': {
      // Check for meaningful alt text
      allowEmpty: false
    }
  }
};

// Environment-specific configurations
const environments = {
  development: {
    ...axeConfig,
    // More verbose reporting in development
    reporter: 'v2',
    verbose: true
  },
  
  production: {
    ...axeConfig,
    // Stricter rules in production
    rules: {
      ...axeConfig.rules,
      'color-contrast': { enabled: true, options: { noScroll: true } }
    }
  },
  
  ci: {
    ...axeConfig,
    // Fast execution for CI
    timeout: 10000,
    resultTypes: ['violations']
  }
};

module.exports = {
  default: axeConfig,
  environments
};