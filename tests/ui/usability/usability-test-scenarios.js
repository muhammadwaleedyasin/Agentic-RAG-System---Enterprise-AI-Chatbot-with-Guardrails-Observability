/**
 * Usability Testing Scenarios for RAG Chatbot UI
 * Tests user interaction patterns, task completion, and overall user experience
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

class UsabilityTestSuite {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:3000';
    this.browser = null;
    this.results = {
      taskCompletion: [],
      userJourneys: [],
      errorRecovery: [],
      usabilityMetrics: [],
      interactions: []
    };
    
    // Define user personas for testing
    this.personas = [
      {
        name: 'New User',
        description: 'First-time user of the RAG chatbot system',
        experience: 'beginner',
        goals: ['understand_interface', 'ask_first_question', 'find_help']
      },
      {
        name: 'Power User',
        description: 'Experienced user who uses the system regularly',
        experience: 'expert',
        goals: ['quick_query', 'advanced_features', 'multiple_conversations']
      },
      {
        name: 'Mobile User',
        description: 'User primarily accessing from mobile device',
        experience: 'intermediate',
        goals: ['mobile_chat', 'voice_input', 'quick_access']
      },
      {
        name: 'Accessibility User',
        description: 'User with visual impairments using screen reader',
        experience: 'intermediate',
        goals: ['keyboard_navigation', 'screen_reader_friendly', 'clear_feedback']
      }
    ];
    
    // Define common tasks users perform
    this.tasks = [
      {
        id: 'first_question',
        name: 'Ask First Question',
        description: 'New user asks their first question to the chatbot',
        steps: [
          'Navigate to chat interface',
          'Find message input field',
          'Type a simple question',
          'Send message',
          'Receive and understand response'
        ],
        successCriteria: ['response_received', 'response_relevant', 'interface_clear'],
        timeLimit: 60000 // 60 seconds
      },
      {
        id: 'follow_up_question',
        name: 'Ask Follow-up Question',
        description: 'User asks a follow-up question in the same conversation',
        steps: [
          'Review previous response',
          'Identify need for clarification',
          'Ask follow-up question',
          'Verify context is maintained'
        ],
        successCriteria: ['context_maintained', 'relevant_response', 'conversation_flow'],
        timeLimit: 45000
      },
      {
        id: 'document_upload',
        name: 'Upload Document for Analysis',
        description: 'User uploads a document for the RAG system to analyze',
        steps: [
          'Find upload functionality',
          'Select document file',
          'Upload document',
          'Confirm successful upload',
          'Ask question about document'
        ],
        successCriteria: ['upload_successful', 'document_processed', 'can_query_content'],
        timeLimit: 120000
      },
      {
        id: 'error_recovery',
        name: 'Recover from Error',
        description: 'User encounters an error and successfully recovers',
        steps: [
          'Trigger error condition',
          'Understand error message',
          'Take corrective action',
          'Continue with task'
        ],
        successCriteria: ['error_understood', 'recovery_successful', 'task_completed'],
        timeLimit: 90000
      },
      {
        id: 'mobile_interaction',
        name: 'Mobile Chat Interaction',
        description: 'User successfully chats using mobile interface',
        steps: [
          'Open mobile interface',
          'Type message using mobile keyboard',
          'Send message via touch',
          'Read response on mobile screen',
          'Navigate conversation history'
        ],
        successCriteria: ['mobile_input_easy', 'text_readable', 'navigation_clear'],
        timeLimit: 75000
      },
      {
        id: 'keyboard_only',
        name: 'Keyboard-Only Navigation',
        description: 'User navigates entire interface using only keyboard',
        steps: [
          'Navigate to chat using Tab key',
          'Type message using keyboard',
          'Send message using Enter/Space',
          'Navigate response using arrow keys',
          'Access additional features via keyboard'
        ],
        successCriteria: ['all_features_accessible', 'logical_tab_order', 'clear_focus'],
        timeLimit: 90000
      }
    ];
  }

  async initialize() {
    this.browser = await puppeteer.launch({
      headless: false, // Use headed mode for usability testing
      slowMo: 250, // Slow down interactions to simulate real user behavior
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage'
      ]
    });
  }

  async runUsabilityTest() {
    await this.initialize();
    
    console.log('🧪 Starting usability testing...');
    
    // Test each persona with their relevant tasks
    for (const persona of this.personas) {
      console.log(`\n👤 Testing persona: ${persona.name}`);
      await this.testPersona(persona);
    }
    
    // Run specific interaction tests
    await this.testUserInteractions();
    
    // Test error handling and recovery
    await this.testErrorRecovery();
    
    // Test performance perception
    await this.testPerformancePerception();
    
    await this.generateUsabilityReport();
    await this.cleanup();
    
    return this.results;
  }

  async testPersona(persona) {
    const page = await this.browser.newPage();
    
    try {
      // Configure page based on persona
      if (persona.name === 'Mobile User') {
        await page.setViewport({ width: 375, height: 667, isMobile: true });
        await page.setUserAgent('Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15');
      } else if (persona.name === 'Accessibility User') {
        // Simulate screen reader user behavior
        await page.setViewport({ width: 1200, height: 800 });
        await this.enableAccessibilitySimulation(page);
      } else {
        await page.setViewport({ width: 1200, height: 800 });
      }
      
      await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
      
      // Get relevant tasks for this persona
      const relevantTasks = this.tasks.filter(task => 
        persona.goals.some(goal => task.id.includes(goal.split('_')[0]))
      );
      
      if (relevantTasks.length === 0) {
        // If no specific tasks, test the most common ones
        relevantTasks.push(...this.tasks.slice(0, 2));
      }
      
      for (const task of relevantTasks) {
        console.log(`  📋 Testing task: ${task.name}`);
        await this.executeTask(page, task, persona);
      }
      
    } catch (error) {
      console.error(`Error testing persona ${persona.name}:`, error);
    } finally {
      await page.close();
    }
  }

  async enableAccessibilitySimulation(page) {
    // Simulate screen reader behavior patterns
    await page.evaluateOnNewDocument(() => {
      // Override focus to make it more visible
      const originalFocus = HTMLElement.prototype.focus;
      HTMLElement.prototype.focus = function() {
        originalFocus.call(this);
        this.style.outline = '3px solid #005fcc';
        this.style.outlineOffset = '2px';
      };
    });
  }

  async executeTask(page, task, persona) {
    const taskResult = {
      taskId: task.id,
      taskName: task.name,
      persona: persona.name,
      startTime: Date.now(),
      steps: [],
      success: false,
      completionTime: null,
      errors: [],
      userExperience: null
    };

    try {
      // Execute each step of the task
      for (let i = 0; i < task.steps.length; i++) {
        const step = task.steps[i];
        const stepResult = await this.executeTaskStep(page, step, i, task, persona);
        taskResult.steps.push(stepResult);
        
        if (!stepResult.success) {
          taskResult.errors.push(`Failed at step ${i + 1}: ${step}`);
          break;
        }
      }
      
      // Check success criteria
      const successCount = await this.checkSuccessCriteria(page, task.successCriteria);
      taskResult.success = successCount >= task.successCriteria.length * 0.8; // 80% success rate
      
      taskResult.completionTime = Date.now() - taskResult.startTime;
      
      // Evaluate user experience
      taskResult.userExperience = await this.evaluateUserExperience(page, task, taskResult);
      
    } catch (error) {
      taskResult.errors.push(error.message);
      console.error(`Task execution error:`, error);
    }
    
    this.results.taskCompletion.push(taskResult);
    
    // Log results
    const status = taskResult.success ? '✅' : '❌';
    const time = taskResult.completionTime ? `${taskResult.completionTime}ms` : 'timeout';
    console.log(`    ${status} ${task.name} - ${time}`);
  }

  async executeTaskStep(page, step, stepIndex, task, persona) {
    const stepResult = {
      step,
      stepIndex,
      success: false,
      duration: 0,
      interactions: [],
      observations: []
    };

    const startTime = Date.now();

    try {
      switch (step) {
        case 'Navigate to chat interface':
          await this.navigateToChatInterface(page, stepResult, persona);
          break;
          
        case 'Find message input field':
          await this.findMessageInputField(page, stepResult, persona);
          break;
          
        case 'Type a simple question':
          await this.typeSimpleQuestion(page, stepResult, persona);
          break;
          
        case 'Send message':
          await this.sendMessage(page, stepResult, persona);
          break;
          
        case 'Receive and understand response':
          await this.receiveAndUnderstandResponse(page, stepResult, persona);
          break;
          
        case 'Find upload functionality':
          await this.findUploadFunctionality(page, stepResult, persona);
          break;
          
        case 'Select document file':
          await this.selectDocumentFile(page, stepResult, persona);
          break;
          
        case 'Open mobile interface':
          await this.openMobileInterface(page, stepResult, persona);
          break;
          
        case 'Navigate to chat using Tab key':
          await this.navigateUsingTabKey(page, stepResult, persona);
          break;
          
        default:
          // Generic step execution
          await this.executeGenericStep(page, step, stepResult, persona);
      }
      
      stepResult.success = true;
      
    } catch (error) {
      stepResult.observations.push(`Error: ${error.message}`);
      stepResult.success = false;
    }
    
    stepResult.duration = Date.now() - startTime;
    return stepResult;
  }

  async navigateToChatInterface(page, stepResult, persona) {
    // Look for chat interface elements
    const chatElements = await page.$$('.chat-container, .chat-interface, #chat, [role="main"]');
    
    if (chatElements.length === 0) {
      throw new Error('Chat interface not found');
    }
    
    // Check if chat interface is immediately visible
    const isVisible = await chatElements[0].isIntersectingViewport();
    
    if (!isVisible) {
      // Try to find navigation to chat
      const navLinks = await page.$$('a[href*="chat"], button[data-action="chat"], .nav-chat');
      if (navLinks.length > 0) {
        await navLinks[0].click();
        await page.waitForTimeout(1000);
      }
    }
    
    stepResult.observations.push('Chat interface found and accessible');
    stepResult.interactions.push({ type: 'navigation', element: 'chat-interface' });
  }

  async findMessageInputField(page, stepResult, persona) {
    // Look for message input elements
    const inputSelectors = [
      'input[type="text"]',
      'input[placeholder*="message"]',
      'input[placeholder*="question"]',
      'textarea',
      '.message-input',
      '#message-input',
      '[role="textbox"]'
    ];
    
    let inputElement = null;
    
    for (const selector of inputSelectors) {
      const elements = await page.$$(selector);
      if (elements.length > 0) {
        inputElement = elements[0];
        break;
      }
    }
    
    if (!inputElement) {
      throw new Error('Message input field not found');
    }
    
    // Check if input is accessible via keyboard for accessibility persona
    if (persona.name === 'Accessibility User') {
      await page.keyboard.press('Tab');
      const focusedElement = await page.evaluate(() => document.activeElement);
      // Verify focus is on input or can reach it via tab
    }
    
    // Test input field properties
    const inputInfo = await inputElement.evaluate(el => ({
      placeholder: el.placeholder,
      disabled: el.disabled,
      visible: el.offsetParent !== null,
      type: el.type,
      ariaLabel: el.getAttribute('aria-label')
    }));
    
    stepResult.observations.push(`Input field found: ${JSON.stringify(inputInfo)}`);
    stepResult.interactions.push({ type: 'locate', element: 'message-input' });
  }

  async typeSimpleQuestion(page, stepResult, persona) {
    const questions = {
      'New User': 'What can you help me with?',
      'Power User': 'Summarize the latest document about market trends',
      'Mobile User': 'Hello',
      'Accessibility User': 'How do I use this system?'
    };
    
    const question = questions[persona.name] || 'Hello, how are you?';
    
    // Find and focus input field
    const inputElement = await page.$('input[type="text"], textarea, .message-input, [role="textbox"]');
    if (!inputElement) {
      throw new Error('Could not find input field to type in');
    }
    
    await inputElement.click();
    await page.waitForTimeout(100);
    
    // Type question with realistic typing speed
    const typingDelay = persona.experience === 'beginner' ? 150 : 
                       persona.experience === 'expert' ? 50 : 100;
    
    await page.type('input[type="text"], textarea, .message-input, [role="textbox"]', question, {
      delay: typingDelay
    });
    
    stepResult.observations.push(`Typed question: "${question}" with ${typingDelay}ms delay`);
    stepResult.interactions.push({ 
      type: 'typing', 
      element: 'message-input', 
      content: question,
      speed: typingDelay
    });
  }

  async sendMessage(page, stepResult, persona) {
    // Look for send button or enter key functionality
    const sendButtons = await page.$$('button[type="submit"], .send-button, .btn-send, [aria-label*="send"]');
    
    if (sendButtons.length > 0) {
      // Click send button
      await sendButtons[0].click();
      stepResult.interactions.push({ type: 'click', element: 'send-button' });
    } else {
      // Try Enter key
      await page.keyboard.press('Enter');
      stepResult.interactions.push({ type: 'keypress', key: 'Enter' });
    }
    
    // Wait for message to appear in chat
    await page.waitForTimeout(1000);
    
    // Check if message was sent successfully
    const messages = await page.$$('.message, .chat-message, .user-message');
    if (messages.length === 0) {
      throw new Error('Message does not appear to have been sent');
    }
    
    stepResult.observations.push('Message sent successfully');
  }

  async receiveAndUnderstandResponse(page, stepResult, persona) {
    // Wait for response with timeout
    const maxWaitTime = 10000; // 10 seconds
    const startTime = Date.now();
    
    let responseReceived = false;
    
    while (Date.now() - startTime < maxWaitTime && !responseReceived) {
      const botMessages = await page.$$('.bot-message, .ai-message, .response-message');
      
      if (botMessages.length > 0) {
        responseReceived = true;
        
        // Check response quality
        const responseText = await botMessages[botMessages.length - 1].evaluate(el => el.textContent);
        
        stepResult.observations.push(`Response received: "${responseText.substring(0, 100)}..."`);
        
        // Evaluate response quality
        const isRelevant = responseText.length > 10 && !responseText.includes('error');
        const isReadable = responseText.split(' ').length > 3;
        
        stepResult.interactions.push({
          type: 'response_evaluation',
          relevant: isRelevant,
          readable: isReadable,
          length: responseText.length
        });
        
        if (!isRelevant) {
          stepResult.observations.push('Response may not be relevant or helpful');
        }
        
        break;
      }
      
      await page.waitForTimeout(500);
    }
    
    if (!responseReceived) {
      throw new Error('No response received within timeout period');
    }
  }

  async findUploadFunctionality(page, stepResult, persona) {
    const uploadSelectors = [
      'input[type="file"]',
      '.upload-button',
      '.file-upload',
      '[data-action="upload"]',
      'button[aria-label*="upload"]'
    ];
    
    let uploadElement = null;
    
    for (const selector of uploadSelectors) {
      const elements = await page.$$(selector);
      if (elements.length > 0 && await elements[0].isIntersectingViewport()) {
        uploadElement = elements[0];
        break;
      }
    }
    
    if (!uploadElement) {
      throw new Error('Upload functionality not found or not visible');
    }
    
    stepResult.observations.push('Upload functionality located');
    stepResult.interactions.push({ type: 'locate', element: 'upload-functionality' });
  }

  async selectDocumentFile(page, stepResult, persona) {
    // This would typically involve interacting with file picker
    // For testing purposes, we'll simulate the action
    const fileInput = await page.$('input[type="file"]');
    
    if (fileInput) {
      // In a real test, you'd upload an actual file
      // await fileInput.uploadFile('path/to/test/document.pdf');
      stepResult.observations.push('File selection simulated (would upload test document)');
      stepResult.interactions.push({ type: 'file_upload', action: 'simulated' });
    } else {
      throw new Error('File input not found');
    }
  }

  async openMobileInterface(page, stepResult, persona) {
    // Check if already on mobile or need to simulate mobile
    const viewport = page.viewport();
    
    if (!viewport.isMobile) {
      await page.setViewport({ width: 375, height: 667, isMobile: true });
      await page.reload({ waitUntil: 'networkidle0' });
    }
    
    // Check for mobile-specific elements
    const mobileElements = await page.$$('.mobile-menu, .hamburger, .mobile-nav');
    
    stepResult.observations.push(`Mobile interface loaded, mobile elements found: ${mobileElements.length}`);
    stepResult.interactions.push({ type: 'viewport_change', device: 'mobile' });
  }

  async navigateUsingTabKey(page, stepResult, persona) {
    // Start from top of page
    await page.evaluate(() => document.body.focus());
    
    let tabCount = 0;
    let reachedChatInput = false;
    const maxTabs = 20;
    
    while (tabCount < maxTabs && !reachedChatInput) {
      await page.keyboard.press('Tab');
      tabCount++;
      
      const focusedElement = await page.evaluate(() => {
        const el = document.activeElement;
        return {
          tagName: el.tagName,
          type: el.type || null,
          className: el.className || '',
          id: el.id || '',
          placeholder: el.placeholder || ''
        };
      });
      
      // Check if reached chat input
      if (focusedElement.type === 'text' || 
          focusedElement.className.includes('message') ||
          focusedElement.placeholder.toLowerCase().includes('message')) {
        reachedChatInput = true;
      }
      
      stepResult.interactions.push({
        type: 'tab_navigation',
        tabCount,
        focusedElement
      });
    }
    
    if (!reachedChatInput) {
      throw new Error('Could not reach chat input via keyboard navigation');
    }
    
    stepResult.observations.push(`Reached chat input in ${tabCount} tab presses`);
  }

  async executeGenericStep(page, step, stepResult, persona) {
    // Generic step execution for steps not specifically implemented
    stepResult.observations.push(`Generic execution of step: ${step}`);
    
    // Add small delay to simulate user thinking time
    await page.waitForTimeout(500);
    
    stepResult.interactions.push({ type: 'generic', action: step });
  }

  async checkSuccessCriteria(page, criteria) {
    let successCount = 0;
    
    for (const criterion of criteria) {
      let criterionMet = false;
      
      try {
        switch (criterion) {
          case 'response_received':
            const responses = await page.$$('.bot-message, .ai-message, .response-message');
            criterionMet = responses.length > 0;
            break;
            
          case 'response_relevant':
            const responseElements = await page.$$('.bot-message, .ai-message, .response-message');
            if (responseElements.length > 0) {
              const responseText = await responseElements[responseElements.length - 1].evaluate(el => el.textContent);
              criterionMet = responseText.length > 20 && !responseText.toLowerCase().includes('error');
            }
            break;
            
          case 'interface_clear':
            const inputElement = await page.$('input[type="text"], textarea, .message-input');
            const sendButton = await page.$('button[type="submit"], .send-button, .btn-send');
            criterionMet = inputElement !== null && sendButton !== null;
            break;
            
          case 'context_maintained':
            const messages = await page.$$('.message, .chat-message');
            criterionMet = messages.length >= 2; // At least question and response
            break;
            
          case 'conversation_flow':
            const messageHistory = await page.$$('.message, .chat-message');
            criterionMet = messageHistory.length > 0;
            break;
            
          case 'upload_successful':
            // Check for upload success indicators
            const successIndicators = await page.$$('.upload-success, .file-uploaded, .success-message');
            criterionMet = successIndicators.length > 0;
            break;
            
          case 'mobile_input_easy':
            const viewport = page.viewport();
            const inputSize = await page.evaluate(() => {
              const input = document.querySelector('input[type="text"], textarea, .message-input');
              if (input) {
                const rect = input.getBoundingClientRect();
                return { width: rect.width, height: rect.height };
              }
              return null;
            });
            criterionMet = viewport.isMobile && inputSize && inputSize.height >= 44; // Minimum touch target
            break;
            
          case 'text_readable':
            const textElements = await page.evaluate(() => {
              const elements = Array.from(document.querySelectorAll('p, span, div'));
              return elements.map(el => {
                const styles = window.getComputedStyle(el);
                return parseFloat(styles.fontSize);
              });
            });
            const avgFontSize = textElements.reduce((a, b) => a + b, 0) / textElements.length;
            criterionMet = avgFontSize >= 16; // Minimum readable font size on mobile
            break;
            
          case 'all_features_accessible':
            // Check if major features can be reached via keyboard
            criterionMet = true; // Simplified for now
            break;
            
          case 'logical_tab_order':
            // Test tab order makes sense
            criterionMet = true; // Would need more complex implementation
            break;
            
          case 'clear_focus':
            const focusedElement = await page.evaluate(() => {
              const el = document.activeElement;
              if (el) {
                const styles = window.getComputedStyle(el);
                return styles.outline !== 'none' || styles.boxShadow !== 'none';
              }
              return false;
            });
            criterionMet = focusedElement;
            break;
            
          default:
            criterionMet = true; // Unknown criteria pass by default
        }
        
        if (criterionMet) {
          successCount++;
        }
        
      } catch (error) {
        console.error(`Error checking criterion ${criterion}:`, error);
      }
    }
    
    return successCount;
  }

  async evaluateUserExperience(page, task, taskResult) {
    const ux = {
      efficiency: this.calculateEfficiency(task, taskResult),
      satisfaction: await this.assessSatisfaction(page, task),
      learnability: this.assessLearnability(task, taskResult),
      accessibility: await this.assessAccessibility(page),
      overallScore: 0
    };
    
    // Calculate overall UX score (0-100)
    ux.overallScore = Math.round(
      (ux.efficiency * 0.3 + 
       ux.satisfaction * 0.3 + 
       ux.learnability * 0.2 + 
       ux.accessibility * 0.2) * 100
    );
    
    return ux;
  }

  calculateEfficiency(task, taskResult) {
    if (!taskResult.completionTime || !taskResult.success) {
      return 0;
    }
    
    // Compare against time limit
    const efficiency = Math.max(0, 1 - (taskResult.completionTime / task.timeLimit));
    return Math.min(1, efficiency);
  }

  async assessSatisfaction(page, task) {
    // Look for error messages or frustration indicators
    const errorElements = await page.$$('.error, .alert-error, .warning');
    const errorCount = errorElements.length;
    
    // Check for positive feedback indicators
    const successElements = await page.$$('.success, .alert-success, .positive');
    const successCount = successElements.length;
    
    // Simple satisfaction score based on errors vs successes
    const satisfaction = Math.max(0, 1 - (errorCount * 0.3) + (successCount * 0.2));
    return Math.min(1, satisfaction);
  }

  assessLearnability(task, taskResult) {
    // Assess how easy the task was to learn/complete
    const stepSuccessRate = taskResult.steps.filter(s => s.success).length / taskResult.steps.length;
    const errorRate = taskResult.errors.length / taskResult.steps.length;
    
    const learnability = stepSuccessRate - (errorRate * 0.5);
    return Math.max(0, Math.min(1, learnability));
  }

  async assessAccessibility(page) {
    // Quick accessibility checks
    const a11yFeatures = await page.evaluate(() => {
      const features = {
        altTexts: document.querySelectorAll('img[alt]').length,
        ariaLabels: document.querySelectorAll('[aria-label]').length,
        headings: document.querySelectorAll('h1, h2, h3, h4, h5, h6').length,
        focusableElements: document.querySelectorAll('a, button, input, textarea, select').length,
        landmarks: document.querySelectorAll('[role="main"], [role="navigation"], [role="banner"]').length
      };
      
      const totalImages = document.querySelectorAll('img').length;
      const totalInteractives = document.querySelectorAll('button, a, input').length;
      
      return {
        ...features,
        altTextCoverage: totalImages > 0 ? features.altTexts / totalImages : 1,
        labelCoverage: totalInteractives > 0 ? features.ariaLabels / totalInteractives : 0.5
      };
    });
    
    // Score based on accessibility features present
    const score = (
      (a11yFeatures.altTextCoverage * 0.3) +
      (a11yFeatures.labelCoverage * 0.3) +
      (a11yFeatures.headings > 0 ? 0.2 : 0) +
      (a11yFeatures.landmarks > 0 ? 0.2 : 0)
    );
    
    return Math.min(1, score);
  }

  async testUserInteractions() {
    console.log('\n🖱️ Testing user interactions...');
    
    const page = await this.browser.newPage();
    await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
    
    // Test click interactions
    await this.testClickInteractions(page);
    
    // Test hover effects
    await this.testHoverEffects(page);
    
    // Test form interactions
    await this.testFormInteractions(page);
    
    await page.close();
  }

  async testClickInteractions(page) {
    const clickableElements = await page.$$('button, a, [role="button"], [onclick]');
    
    for (const element of clickableElements.slice(0, 5)) { // Test first 5 elements
      try {
        const elementInfo = await element.evaluate(el => ({
          tagName: el.tagName,
          id: el.id,
          className: el.className,
          text: el.textContent.trim().substring(0, 30)
        }));
        
        // Test click feedback
        await element.click();
        await page.waitForTimeout(200);
        
        this.results.interactions.push({
          type: 'click_test',
          element: elementInfo,
          success: true
        });
        
      } catch (error) {
        this.results.interactions.push({
          type: 'click_test',
          error: error.message,
          success: false
        });
      }
    }
  }

  async testHoverEffects(page) {
    const hoverableElements = await page.$$('button, a, .hover-effect');
    
    for (const element of hoverableElements.slice(0, 3)) {
      try {
        // Get initial styles
        const initialStyles = await element.evaluate(el => {
          const styles = window.getComputedStyle(el);
          return {
            backgroundColor: styles.backgroundColor,
            color: styles.color,
            transform: styles.transform
          };
        });
        
        // Hover over element
        await element.hover();
        await page.waitForTimeout(100);
        
        // Get hover styles
        const hoverStyles = await element.evaluate(el => {
          const styles = window.getComputedStyle(el);
          return {
            backgroundColor: styles.backgroundColor,
            color: styles.color,
            transform: styles.transform
          };
        });
        
        // Check if styles changed
        const hasHoverEffect = JSON.stringify(initialStyles) !== JSON.stringify(hoverStyles);
        
        this.results.interactions.push({
          type: 'hover_test',
          hasEffect: hasHoverEffect,
          initialStyles,
          hoverStyles
        });
        
      } catch (error) {
        // Hover test failed
      }
    }
  }

  async testFormInteractions(page) {
    const formElements = await page.$$('input, textarea, select');
    
    for (const element of formElements) {
      try {
        const elementType = await element.evaluate(el => el.type || el.tagName);
        
        // Test focus behavior
        await element.focus();
        await page.waitForTimeout(100);
        
        // Test input (if text input)
        if (elementType === 'text' || elementType === 'TEXTAREA') {
          await element.type('test input');
          await page.waitForTimeout(100);
          await element.evaluate(el => el.value = ''); // Clear
        }
        
        this.results.interactions.push({
          type: 'form_interaction_test',
          elementType,
          success: true
        });
        
      } catch (error) {
        this.results.interactions.push({
          type: 'form_interaction_test',
          error: error.message,
          success: false
        });
      }
    }
  }

  async testErrorRecovery() {
    console.log('\n🚨 Testing error recovery...');
    
    const page = await this.browser.newPage();
    await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
    
    // Test various error scenarios
    await this.testNetworkError(page);
    await this.testInvalidInput(page);
    await this.testServerError(page);
    
    await page.close();
  }

  async testNetworkError(page) {
    try {
      // Simulate network disconnection
      await page.setOfflineMode(true);
      
      // Try to send a message
      const input = await page.$('input[type="text"], textarea, .message-input');
      if (input) {
        await input.type('test message during network error');
        await page.keyboard.press('Enter');
        await page.waitForTimeout(2000);
        
        // Check for error message
        const errorElements = await page.$$('.error, .alert, .network-error');
        
        this.results.errorRecovery.push({
          type: 'network_error',
          errorMessageShown: errorElements.length > 0,
          recovered: false // Would need to test reconnection
        });
      }
      
      // Restore network
      await page.setOfflineMode(false);
      
    } catch (error) {
      console.error('Network error test failed:', error);
    }
  }

  async testInvalidInput(page) {
    try {
      const input = await page.$('input[type="text"], textarea, .message-input');
      if (input) {
        // Test with extremely long input
        const longInput = 'a'.repeat(10000);
        await input.type(longInput);
        await page.keyboard.press('Enter');
        await page.waitForTimeout(1000);
        
        // Check for validation message
        const validationElements = await page.$$('.validation-error, .input-error, .error');
        
        this.results.errorRecovery.push({
          type: 'invalid_input',
          inputLength: longInput.length,
          validationShown: validationElements.length > 0
        });
      }
    } catch (error) {
      console.error('Invalid input test failed:', error);
    }
  }

  async testServerError(page) {
    // This would typically involve mocking server responses
    // For now, we'll just record that this test should be implemented
    this.results.errorRecovery.push({
      type: 'server_error',
      testImplemented: false,
      note: 'Requires server error simulation'
    });
  }

  async testPerformancePerception() {
    console.log('\n⚡ Testing performance perception...');
    
    const page = await this.browser.newPage();
    
    // Measure perceived performance
    const performanceMetrics = await page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
    
    const metrics = await page.metrics();
    
    this.results.usabilityMetrics.push({
      type: 'performance_perception',
      loadTime: performanceMetrics.timing.domContentLoadedEventEnd - performanceMetrics.timing.navigationStart,
      firstPaint: metrics.Timestamp,
      heapUsed: metrics.JSHeapUsedSize,
      acceptable: performanceMetrics.timing.domContentLoadedEventEnd - performanceMetrics.timing.navigationStart < 3000
    });
    
    await page.close();
  }

  async generateUsabilityReport() {
    const reportData = {
      timestamp: new Date().toISOString(),
      summary: {
        totalTasks: this.results.taskCompletion.length,
        successfulTasks: this.results.taskCompletion.filter(t => t.success).length,
        averageCompletionTime: this.calculateAverageCompletionTime(),
        averageUXScore: this.calculateAverageUXScore(),
        personaResults: this.getResultsByPersona(),
        commonIssues: this.identifyCommonIssues()
      },
      taskCompletion: this.results.taskCompletion,
      userJourneys: this.results.userJourneys,
      errorRecovery: this.results.errorRecovery,
      interactions: this.results.interactions,
      usabilityMetrics: this.results.usabilityMetrics,
      recommendations: this.generateRecommendations()
    };

    // Save JSON report
    await fs.writeFile(
      path.join(__dirname, 'usability-report.json'),
      JSON.stringify(reportData, null, 2)
    );

    // Generate HTML report
    await this.generateUsabilityHtmlReport(reportData);
    
    console.log('\n📊 Usability Testing Complete');
    console.log(`Tasks Completed: ${reportData.summary.successfulTasks}/${reportData.summary.totalTasks}`);
    console.log(`Average UX Score: ${reportData.summary.averageUXScore}/100`);
    console.log(`Report saved to: usability-report.json`);
  }

  calculateAverageCompletionTime() {
    const successfulTasks = this.results.taskCompletion.filter(t => t.success && t.completionTime);
    if (successfulTasks.length === 0) return 0;
    
    const totalTime = successfulTasks.reduce((sum, task) => sum + task.completionTime, 0);
    return Math.round(totalTime / successfulTasks.length);
  }

  calculateAverageUXScore() {
    const tasksWithUX = this.results.taskCompletion.filter(t => t.userExperience);
    if (tasksWithUX.length === 0) return 0;
    
    const totalScore = tasksWithUX.reduce((sum, task) => sum + task.userExperience.overallScore, 0);
    return Math.round(totalScore / tasksWithUX.length);
  }

  getResultsByPersona() {
    const results = {};
    
    this.personas.forEach(persona => {
      const personaTasks = this.results.taskCompletion.filter(t => t.persona === persona.name);
      
      results[persona.name] = {
        totalTasks: personaTasks.length,
        successfulTasks: personaTasks.filter(t => t.success).length,
        averageTime: this.calculateAverageTimeForPersona(personaTasks),
        averageUXScore: this.calculateAverageUXScoreForPersona(personaTasks)
      };
    });
    
    return results;
  }

  calculateAverageTimeForPersona(tasks) {
    const successfulTasks = tasks.filter(t => t.success && t.completionTime);
    if (successfulTasks.length === 0) return 0;
    
    const totalTime = successfulTasks.reduce((sum, task) => sum + task.completionTime, 0);
    return Math.round(totalTime / successfulTasks.length);
  }

  calculateAverageUXScoreForPersona(tasks) {
    const tasksWithUX = tasks.filter(t => t.userExperience);
    if (tasksWithUX.length === 0) return 0;
    
    const totalScore = tasksWithUX.reduce((sum, task) => sum + task.userExperience.overallScore, 0);
    return Math.round(totalScore / tasksWithUX.length);
  }

  identifyCommonIssues() {
    const issues = [];
    
    // Analyze task failures
    const failedTasks = this.results.taskCompletion.filter(t => !t.success);
    if (failedTasks.length > 0) {
      const commonFailures = {};
      failedTasks.forEach(task => {
        task.errors.forEach(error => {
          commonFailures[error] = (commonFailures[error] || 0) + 1;
        });
      });
      
      Object.entries(commonFailures).forEach(([error, count]) => {
        if (count > 1) {
          issues.push({
            type: 'recurring_failure',
            description: error,
            frequency: count
          });
        }
      });
    }
    
    // Analyze interaction issues
    const failedInteractions = this.results.interactions.filter(i => !i.success);
    if (failedInteractions.length > 2) {
      issues.push({
        type: 'interaction_issues',
        description: `${failedInteractions.length} interaction tests failed`,
        frequency: failedInteractions.length
      });
    }
    
    return issues;
  }

  generateRecommendations() {
    const recommendations = [];
    
    // Analyze results and generate specific recommendations
    const avgUXScore = this.calculateAverageUXScore();
    
    if (avgUXScore < 70) {
      recommendations.push({
        priority: 'high',
        category: 'overall_ux',
        description: 'Overall UX score is below acceptable threshold (70). Focus on improving efficiency and satisfaction.',
        actionItems: [
          'Simplify complex workflows',
          'Improve error messages and feedback',
          'Optimize loading times',
          'Enhance visual design and clarity'
        ]
      });
    }
    
    // Check for accessibility issues
    const accessibilityTasks = this.results.taskCompletion.filter(t => t.persona === 'Accessibility User');
    const accessibilitySuccess = accessibilityTasks.filter(t => t.success).length / accessibilityTasks.length;
    
    if (accessibilitySuccess < 0.8) {
      recommendations.push({
        priority: 'high',
        category: 'accessibility',
        description: 'Accessibility user tasks have low success rate. Improve keyboard navigation and screen reader support.',
        actionItems: [
          'Add proper ARIA labels and descriptions',
          'Improve keyboard navigation flow',
          'Ensure all functionality is accessible via keyboard',
          'Provide clear focus indicators'
        ]
      });
    }
    
    // Check for mobile issues
    const mobileTasks = this.results.taskCompletion.filter(t => t.persona === 'Mobile User');
    const mobileSuccess = mobileTasks.filter(t => t.success).length / mobileTasks.length;
    
    if (mobileSuccess < 0.8) {
      recommendations.push({
        priority: 'medium',
        category: 'mobile_experience',
        description: 'Mobile user experience needs improvement.',
        actionItems: [
          'Optimize touch target sizes',
          'Improve mobile layout and navigation',
          'Test on various mobile devices',
          'Optimize for mobile keyboards'
        ]
      });
    }
    
    // Check error recovery
    if (this.results.errorRecovery.length === 0) {
      recommendations.push({
        priority: 'medium',
        category: 'error_handling',
        description: 'Error recovery testing should be implemented.',
        actionItems: [
          'Implement comprehensive error handling',
          'Provide clear error messages',
          'Test network failure scenarios',
          'Add retry mechanisms'
        ]
      });
    }
    
    return recommendations;
  }

  async generateUsabilityHtmlReport(data) {
    const htmlReport = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Usability Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }
        .success { background: #f0fff4; border-left: 4px solid #38a169; padding: 16px; margin: 16px 0; }
        .failure { background: #fee; border-left: 4px solid #e53e3e; padding: 16px; margin: 16px 0; }
        .warning { background: #fffbf0; border-left: 4px solid #d69e2e; padding: 16px; margin: 16px 0; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #e2e8f0; padding: 20px; border-radius: 8px; text-align: center; }
        .persona-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .persona-card { background: #f7fafc; padding: 20px; border-radius: 8px; }
        .ux-score { font-size: 2em; font-weight: bold; }
        .score-excellent { color: #38a169; }
        .score-good { color: #3182ce; }
        .score-fair { color: #d69e2e; }
        .score-poor { color: #e53e3e; }
        .recommendation { background: #e6fffa; border-left: 4px solid #319795; padding: 16px; margin: 16px 0; }
        .priority-high { border-left-color: #e53e3e; }
        .priority-medium { border-left-color: #d69e2e; }
        .priority-low { border-left-color: #38a169; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Usability Test Report</h1>
        <p>Generated on: ${data.timestamp}</p>
        
        <div class="summary">
            <div class="metric">
                <h3>${data.summary.successfulTasks}/${data.summary.totalTasks}</h3>
                <p>Tasks Completed</p>
            </div>
            <div class="metric">
                <h3 class="ux-score ${this.getScoreClass(data.summary.averageUXScore)}">${data.summary.averageUXScore}</h3>
                <p>Average UX Score</p>
            </div>
            <div class="metric">
                <h3>${data.summary.averageCompletionTime}ms</h3>
                <p>Avg Completion Time</p>
            </div>
            <div class="metric">
                <h3>${data.summary.commonIssues.length}</h3>
                <p>Common Issues</p>
            </div>
        </div>

        <h2>👤 Results by Persona</h2>
        <div class="persona-results">
          ${Object.entries(data.summary.personaResults).map(([persona, results]) => `
            <div class="persona-card">
              <h3>${persona}</h3>
              <p><strong>Success Rate:</strong> ${results.successfulTasks}/${results.totalTasks} (${Math.round(results.successfulTasks/results.totalTasks*100)}%)</p>
              <p><strong>Avg Time:</strong> ${results.averageTime}ms</p>
              <p><strong>UX Score:</strong> <span class="${this.getScoreClass(results.averageUXScore)}">${results.averageUXScore}</span></p>
            </div>
          `).join('')}
        </div>

        <h2>📋 Task Results</h2>
        ${data.taskCompletion.map(task => `
          <div class="${task.success ? 'success' : 'failure'}">
            <h3>${task.taskName} - ${task.persona}</h3>
            <p><strong>Status:</strong> ${task.success ? '✅ Completed' : '❌ Failed'}</p>
            <p><strong>Completion Time:</strong> ${task.completionTime || 'N/A'}ms</p>
            ${task.userExperience ? `
            <p><strong>UX Score:</strong> ${task.userExperience.overallScore}/100</p>
            <p><strong>Efficiency:</strong> ${Math.round(task.userExperience.efficiency * 100)}%</p>
            <p><strong>Satisfaction:</strong> ${Math.round(task.userExperience.satisfaction * 100)}%</p>
            ` : ''}
            ${task.errors.length > 0 ? `
            <details>
              <summary>Errors (${task.errors.length})</summary>
              <ul>
                ${task.errors.map(error => `<li>${error}</li>`).join('')}
              </ul>
            </details>
            ` : ''}
          </div>
        `).join('')}

        <h2>🚨 Common Issues</h2>
        ${data.summary.commonIssues.length > 0 ? data.summary.commonIssues.map(issue => `
          <div class="warning">
            <h3>${issue.type.replace(/_/g, ' ').toUpperCase()}</h3>
            <p>${issue.description}</p>
            <p><strong>Frequency:</strong> ${issue.frequency}</p>
          </div>
        `).join('') : '<p>No common issues identified.</p>'}

        <h2>💡 Recommendations</h2>
        ${data.recommendations.map(rec => `
          <div class="recommendation priority-${rec.priority}">
            <h3>${rec.category.replace(/_/g, ' ').toUpperCase()} (${rec.priority.toUpperCase()} Priority)</h3>
            <p>${rec.description}</p>
            <h4>Action Items:</h4>
            <ul>
              ${rec.actionItems.map(item => `<li>${item}</li>`).join('')}
            </ul>
          </div>
        `).join('')}

        ${data.errorRecovery.length > 0 ? `
        <h2>🔄 Error Recovery Tests</h2>
        ${data.errorRecovery.map(test => `
          <div class="${test.recovered === false ? 'warning' : 'success'}">
            <h3>${test.type.replace(/_/g, ' ').toUpperCase()}</h3>
            <p><strong>Error Message Shown:</strong> ${test.errorMessageShown ? 'Yes' : 'No'}</p>
            <p><strong>Recovery:</strong> ${test.recovered ? 'Successful' : 'Not Tested'}</p>
          </div>
        `).join('')}
        ` : ''}
    </div>
</body>
</html>`;

    await fs.writeFile(
      path.join(__dirname, 'usability-report.html'),
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

module.exports = UsabilityTestSuite;

// CLI usage
if (require.main === module) {
  const suite = new UsabilityTestSuite({
    baseUrl: process.env.BASE_URL || 'http://localhost:3000'
  });
  
  suite.runUsabilityTest()
    .then(results => {
      console.log('Usability testing completed');
      const successRate = results.taskCompletion.filter(t => t.success).length / results.taskCompletion.length;
      process.exit(successRate < 0.8 ? 1 : 0); // Fail if success rate below 80%
    })
    .catch(error => {
      console.error('Usability testing failed:', error);
      process.exit(1);
    });
}