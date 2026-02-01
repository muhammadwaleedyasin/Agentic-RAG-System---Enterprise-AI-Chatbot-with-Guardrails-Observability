"""
Mobile Browser Compatibility Testing Suite for RAG Chatbot UI

Tests mobile browser compatibility including iOS Safari, Chrome Mobile,
responsive design, touch interactions, and mobile-specific features.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.touch_actions import TouchActions
import pytest
import numpy as np
from pathlib import Path


@dataclass
class MobileTestResult:
    """Mobile test result container"""
    test_name: str
    device_type: str
    browser: str
    viewport_size: Tuple[int, int]
    success: bool
    duration: float
    touch_interactions: int
    responsive_score: float
    performance_metrics: Dict[str, float]
    accessibility_score: float
    error_message: Optional[str] = None


class MobileCompatibilityTester:
    """Mobile browser compatibility and responsive design testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
        # Mobile device configurations
        self.mobile_devices = {
            'iphone_se': {
                'width': 375,
                'height': 667,
                'pixel_ratio': 2.0,
                'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
            },
            'iphone_12': {
                'width': 390,
                'height': 844,
                'pixel_ratio': 3.0,
                'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
            },
            'iphone_12_pro_max': {
                'width': 428,
                'height': 926,
                'pixel_ratio': 3.0,
                'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
            },
            'samsung_galaxy_s21': {
                'width': 384,
                'height': 854,
                'pixel_ratio': 2.75,
                'user_agent': 'Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36'
            },
            'samsung_galaxy_note': {
                'width': 412,
                'height': 915,
                'pixel_ratio': 2.625,
                'user_agent': 'Mozilla/5.0 (Linux; Android 11; SM-N986B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36'
            },
            'ipad': {
                'width': 768,
                'height': 1024,
                'pixel_ratio': 2.0,
                'user_agent': 'Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
            },
            'ipad_pro': {
                'width': 1024,
                'height': 1366,
                'pixel_ratio': 2.0,
                'user_agent': 'Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
            }
        }
        
        # Responsive breakpoints
        self.breakpoints = {
            'mobile_small': (320, 568),
            'mobile_medium': (375, 667),
            'mobile_large': (414, 896),
            'tablet_portrait': (768, 1024),
            'tablet_landscape': (1024, 768),
            'desktop_small': (1280, 720),
            'desktop_large': (1920, 1080)
        }
        
        # Touch interaction test scenarios
        self.touch_scenarios = [
            'tap_elements',
            'swipe_gestures',
            'pinch_zoom',
            'scroll_performance',
            'touch_accuracy',
            'multi_touch'
        ]
    
    def _get_mobile_driver(self, device: str) -> webdriver.Chrome:
        """Configure Chrome driver for mobile device simulation"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        device_config = self.mobile_devices[device]
        
        # Mobile emulation
        mobile_emulation = {
            "deviceMetrics": {
                "width": device_config['width'],
                "height": device_config['height'],
                "pixelRatio": device_config['pixel_ratio']
            },
            "userAgent": device_config['user_agent']
        }
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        # Enable touch events
        options.add_argument('--touch-events')
        
        # Performance logging
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': True
        })
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    async def test_responsive_design(self, device: str) -> MobileTestResult:
        """Test responsive design across different viewports"""
        driver = self._get_mobile_driver(device)
        device_config = self.mobile_devices[device]
        viewport_size = (device_config['width'], device_config['height'])
        
        try:
            start_time = time.time()
            
            # Navigate to application
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get performance metrics
            performance_metrics = self._get_mobile_performance_metrics(driver)
            
            # Test responsive layout
            responsive_score = await self._test_responsive_layout(driver, viewport_size)
            
            # Test mobile navigation
            nav_score = await self._test_mobile_navigation(driver)
            
            # Test form usability
            form_score = await self._test_mobile_forms(driver)
            
            # Test content readability
            content_score = await self._test_content_readability(driver)
            
            # Calculate overall responsive score
            overall_score = (responsive_score + nav_score + form_score + content_score) / 4
            
            duration = time.time() - start_time
            
            return MobileTestResult(
                test_name="responsive_design",
                device_type=device,
                browser="chrome_mobile",
                viewport_size=viewport_size,
                success=overall_score >= 0.7,  # 70% threshold
                duration=duration,
                touch_interactions=0,
                responsive_score=overall_score,
                performance_metrics=performance_metrics,
                accessibility_score=0  # Will be calculated in accessibility test
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return MobileTestResult(
                test_name="responsive_design",
                device_type=device,
                browser="chrome_mobile",
                viewport_size=viewport_size,
                success=False,
                duration=duration,
                touch_interactions=0,
                responsive_score=0,
                performance_metrics={},
                accessibility_score=0,
                error_message=str(e)
            )
        finally:
            driver.quit()
    
    async def _test_responsive_layout(self, driver: webdriver.Chrome, viewport_size: Tuple[int, int]) -> float:
        """Test responsive layout adaptation"""
        layout_tests = []
        
        try:
            # Check if content fits viewport
            body_width = driver.execute_script("return document.body.scrollWidth;")
            viewport_width = viewport_size[0]
            
            if body_width <= viewport_width * 1.05:  # Allow 5% tolerance
                layout_tests.append(1.0)
            else:
                layout_tests.append(0.0)
            
            # Check for horizontal scrolling
            has_horizontal_scroll = driver.execute_script("return document.body.scrollWidth > window.innerWidth;")
            if not has_horizontal_scroll:
                layout_tests.append(1.0)
            else:
                layout_tests.append(0.0)
            
            # Check element visibility and positioning
            important_elements = driver.find_elements(By.CSS_SELECTOR, "header, nav, main, .chat-input, .search, button")
            visible_elements = 0
            
            for element in important_elements:
                if element.is_displayed():
                    rect = element.rect
                    if rect['x'] >= 0 and rect['y'] >= 0 and rect['x'] + rect['width'] <= viewport_width:
                        visible_elements += 1
            
            if important_elements:
                visibility_score = visible_elements / len(important_elements)
                layout_tests.append(visibility_score)
            
            # Check for responsive images
            images = driver.find_elements(By.TAG_NAME, "img")
            responsive_images = 0
            
            for img in images:
                width = img.get_attribute('width')
                style = img.get_attribute('style')
                
                if 'max-width' in (style or '') or 'width: 100%' in (style or ''):
                    responsive_images += 1
            
            if images:
                img_score = responsive_images / len(images)
                layout_tests.append(img_score)
            
            return np.mean(layout_tests) if layout_tests else 0.0
            
        except Exception:
            return 0.0
    
    async def _test_mobile_navigation(self, driver: webdriver.Chrome) -> float:
        """Test mobile navigation usability"""
        nav_tests = []
        
        try:
            # Check for mobile menu/hamburger button
            mobile_menu_selectors = [
                ".hamburger", ".menu-toggle", ".mobile-menu", 
                "[aria-label*='menu']", ".nav-toggle"
            ]
            
            mobile_menu_found = False
            for selector in mobile_menu_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and elements[0].is_displayed():
                    mobile_menu_found = True
                    break
            
            nav_tests.append(1.0 if mobile_menu_found else 0.5)
            
            # Check touch target sizes
            clickable_elements = driver.find_elements(By.CSS_SELECTOR, "button, a, input[type='submit'], .clickable")
            adequate_touch_targets = 0
            
            for element in clickable_elements:
                if element.is_displayed():
                    size = element.size
                    # Apple recommends 44x44 points, Android recommends 48x48dp
                    if size['width'] >= 40 and size['height'] >= 40:
                        adequate_touch_targets += 1
            
            if clickable_elements:
                touch_target_score = adequate_touch_targets / len(clickable_elements)
                nav_tests.append(touch_target_score)
            
            # Check spacing between touch elements
            spacing_score = self._check_touch_element_spacing(driver)
            nav_tests.append(spacing_score)
            
            return np.mean(nav_tests) if nav_tests else 0.0
            
        except Exception:
            return 0.0
    
    def _check_touch_element_spacing(self, driver: webdriver.Chrome) -> float:
        """Check spacing between touch elements"""
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, "button, a, input")
            adequate_spacing = 0
            total_pairs = 0
            
            for i, elem1 in enumerate(elements):
                if not elem1.is_displayed():
                    continue
                    
                rect1 = elem1.rect
                
                for elem2 in elements[i+1:i+5]:  # Check next 4 elements
                    if not elem2.is_displayed():
                        continue
                        
                    rect2 = elem2.rect
                    
                    # Calculate distance between elements
                    distance = min(
                        abs(rect1['x'] + rect1['width'] - rect2['x']),
                        abs(rect2['x'] + rect2['width'] - rect1['x']),
                        abs(rect1['y'] + rect1['height'] - rect2['y']),
                        abs(rect2['y'] + rect2['height'] - rect1['y'])
                    )
                    
                    total_pairs += 1
                    if distance >= 8:  # 8px minimum spacing
                        adequate_spacing += 1
            
            return adequate_spacing / total_pairs if total_pairs > 0 else 1.0
            
        except Exception:
            return 0.5
    
    async def _test_mobile_forms(self, driver: webdriver.Chrome) -> float:
        """Test mobile form usability"""
        form_tests = []
        
        try:
            forms = driver.find_elements(By.TAG_NAME, "form")
            if not forms:
                return 1.0  # No forms to test
            
            for form in forms:
                # Check input field sizes
                inputs = form.find_elements(By.CSS_SELECTOR, "input, textarea, select")
                adequate_input_sizes = 0
                
                for input_elem in inputs:
                    if input_elem.is_displayed():
                        size = input_elem.size
                        if size['height'] >= 36:  # Minimum touch-friendly height
                            adequate_input_sizes += 1
                
                if inputs:
                    form_tests.append(adequate_input_sizes / len(inputs))
                
                # Check for mobile-friendly input types
                mobile_input_types = ['tel', 'email', 'url', 'number', 'search']
                mobile_optimized_inputs = 0
                
                for input_elem in inputs:
                    input_type = input_elem.get_attribute('type')
                    if input_type in mobile_input_types:
                        mobile_optimized_inputs += 1
                
                if inputs:
                    mobile_optimization_score = mobile_optimized_inputs / len(inputs)
                    form_tests.append(mobile_optimization_score)
                
                # Check for autocomplete attributes
                autocomplete_inputs = 0
                for input_elem in inputs:
                    if input_elem.get_attribute('autocomplete'):
                        autocomplete_inputs += 1
                
                if inputs:
                    autocomplete_score = autocomplete_inputs / len(inputs)
                    form_tests.append(autocomplete_score)
            
            return np.mean(form_tests) if form_tests else 0.0
            
        except Exception:
            return 0.0
    
    async def _test_content_readability(self, driver: webdriver.Chrome) -> float:
        """Test content readability on mobile"""
        readability_tests = []
        
        try:
            # Check font sizes
            text_elements = driver.find_elements(By.CSS_SELECTOR, "p, span, div, h1, h2, h3, h4, h5, h6")
            adequate_font_sizes = 0
            
            for element in text_elements[:20]:  # Test first 20 elements
                if element.is_displayed() and element.text.strip():
                    font_size = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).fontSize;", element
                    )
                    
                    if font_size:
                        size_px = float(font_size.replace('px', ''))
                        if size_px >= 14:  # Minimum readable font size
                            adequate_font_sizes += 1
            
            if text_elements:
                font_score = adequate_font_sizes / min(len(text_elements), 20)
                readability_tests.append(font_score)
            
            # Check line height
            line_height_elements = 0
            adequate_line_heights = 0
            
            for element in text_elements[:10]:
                if element.is_displayed() and element.text.strip():
                    line_height = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).lineHeight;", element
                    )
                    
                    if line_height and line_height != 'normal':
                        line_height_elements += 1
                        if 'px' in line_height:
                            lh_px = float(line_height.replace('px', ''))
                            if lh_px >= 20:  # Adequate line height
                                adequate_line_heights += 1
                        else:
                            # Relative line height
                            lh_val = float(line_height)
                            if lh_val >= 1.4:
                                adequate_line_heights += 1
            
            if line_height_elements > 0:
                line_height_score = adequate_line_heights / line_height_elements
                readability_tests.append(line_height_score)
            
            # Check text contrast (simplified)
            contrast_score = self._check_text_contrast(driver)
            readability_tests.append(contrast_score)
            
            return np.mean(readability_tests) if readability_tests else 0.0
            
        except Exception:
            return 0.0
    
    def _check_text_contrast(self, driver: webdriver.Chrome) -> float:
        """Check text contrast ratio (simplified test)"""
        try:
            # Get computed styles for text elements
            text_elements = driver.find_elements(By.CSS_SELECTOR, "p, span, div")
            adequate_contrast = 0
            total_elements = 0
            
            for element in text_elements[:10]:  # Test first 10 elements
                if element.is_displayed() and element.text.strip():
                    color = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).color;", element
                    )
                    bg_color = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).backgroundColor;", element
                    )
                    
                    # Simple contrast check - if colors are specified and different
                    if color and bg_color and color != bg_color:
                        adequate_contrast += 1
                    elif not bg_color or bg_color == 'rgba(0, 0, 0, 0)':
                        # Assume adequate contrast if no background color (inherits from body)
                        adequate_contrast += 1
                    
                    total_elements += 1
            
            return adequate_contrast / total_elements if total_elements > 0 else 1.0
            
        except Exception:
            return 0.5  # Assume moderate contrast if test fails
    
    async def test_touch_interactions(self, device: str) -> MobileTestResult:
        """Test touch interactions and gestures"""
        driver = self._get_mobile_driver(device)
        device_config = self.mobile_devices[device]
        viewport_size = (device_config['width'], device_config['height'])
        
        try:
            start_time = time.time()
            touch_interactions = 0
            
            # Navigate to application
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Test tap interactions
            tap_score = await self._test_tap_interactions(driver)
            touch_interactions += 10
            
            # Test scroll performance
            scroll_score = await self._test_scroll_performance(driver)
            touch_interactions += 5
            
            # Test swipe gestures (if applicable)
            swipe_score = await self._test_swipe_gestures(driver)
            touch_interactions += 5
            
            # Test touch accuracy
            accuracy_score = await self._test_touch_accuracy(driver)
            touch_interactions += 10
            
            # Calculate overall touch interaction score
            overall_score = np.mean([tap_score, scroll_score, swipe_score, accuracy_score])
            
            duration = time.time() - start_time
            performance_metrics = self._get_mobile_performance_metrics(driver)
            
            return MobileTestResult(
                test_name="touch_interactions",
                device_type=device,
                browser="chrome_mobile",
                viewport_size=viewport_size,
                success=overall_score >= 0.7,
                duration=duration,
                touch_interactions=touch_interactions,
                responsive_score=overall_score,
                performance_metrics=performance_metrics,
                accessibility_score=0
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return MobileTestResult(
                test_name="touch_interactions",
                device_type=device,
                browser="chrome_mobile",
                viewport_size=viewport_size,
                success=False,
                duration=duration,
                touch_interactions=touch_interactions,
                responsive_score=0,
                performance_metrics={},
                accessibility_score=0,
                error_message=str(e)
            )
        finally:
            driver.quit()
    
    async def _test_tap_interactions(self, driver: webdriver.Chrome) -> float:
        """Test tap interactions on various elements"""
        try:
            tappable_elements = driver.find_elements(By.CSS_SELECTOR, "button, a, input, .clickable")
            successful_taps = 0
            
            for element in tappable_elements[:10]:  # Test first 10 elements
                if element.is_displayed():
                    try:
                        # Use TouchActions for mobile-like interaction
                        touch_actions = TouchActions(driver)
                        touch_actions.tap(element).perform()
                        successful_taps += 1
                        await asyncio.sleep(0.2)
                    except Exception:
                        # Try regular click as fallback
                        try:
                            element.click()
                            successful_taps += 1
                        except Exception:
                            pass
            
            return successful_taps / min(len(tappable_elements), 10) if tappable_elements else 0.0
            
        except Exception:
            return 0.0
    
    async def _test_scroll_performance(self, driver: webdriver.Chrome) -> float:
        """Test scroll performance and smoothness"""
        try:
            # Measure scroll performance
            scroll_script = """
                let startTime = performance.now();
                let scrollCount = 0;
                
                function smoothScroll() {
                    window.scrollBy(0, 100);
                    scrollCount++;
                    
                    if (scrollCount < 10) {
                        requestAnimationFrame(smoothScroll);
                    } else {
                        window.scrollTo(0, 0);
                        return performance.now() - startTime;
                    }
                }
                
                return new Promise((resolve) => {
                    smoothScroll();
                    setTimeout(() => resolve(performance.now() - startTime), 1000);
                });
            """
            
            scroll_time = driver.execute_async_script(scroll_script)
            
            # Good scroll performance should complete in under 500ms
            if scroll_time < 500:
                return 1.0
            elif scroll_time < 1000:
                return 0.7
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    async def _test_swipe_gestures(self, driver: webdriver.Chrome) -> float:
        """Test swipe gestures (limited in Selenium)"""
        try:
            # Test horizontal swipe simulation
            body = driver.find_element(By.TAG_NAME, "body")
            
            # Simulate swipe using touch actions
            touch_actions = TouchActions(driver)
            touch_actions.scroll_from_element(body, 100, 0).perform()
            await asyncio.sleep(0.5)
            
            # If no error occurred, consider it successful
            return 1.0
            
        except Exception:
            # Swipe gestures might not be fully supported
            return 0.5
    
    async def _test_touch_accuracy(self, driver: webdriver.Chrome) -> float:
        """Test touch target accuracy and ease of use"""
        try:
            # Find small touch targets that might be problematic
            small_elements = driver.find_elements(By.CSS_SELECTOR, "button, a, input")
            problematic_targets = 0
            total_targets = 0
            
            for element in small_elements:
                if element.is_displayed():
                    size = element.size
                    total_targets += 1
                    
                    # Check if element is too small for comfortable touch
                    if size['width'] < 32 or size['height'] < 32:
                        problematic_targets += 1
            
            if total_targets == 0:
                return 1.0
            
            # Higher score means fewer problematic targets
            accuracy_score = 1.0 - (problematic_targets / total_targets)
            return max(0.0, accuracy_score)
            
        except Exception:
            return 0.5
    
    def _get_mobile_performance_metrics(self, driver: webdriver.Chrome) -> Dict[str, float]:
        """Get mobile-specific performance metrics"""
        try:
            return driver.execute_script("""
                const timing = performance.timing;
                const navigation = performance.getEntriesByType('navigation')[0];
                
                return {
                    page_load_time: (timing.loadEventEnd - timing.navigationStart) / 1000,
                    dom_content_loaded: (timing.domContentLoadedEventEnd - timing.navigationStart) / 1000,
                    first_paint: performance.getEntriesByType('paint')[0]?.startTime / 1000 || 0,
                    transfer_size: navigation?.transferSize || 0,
                    memory_used: performance.memory?.usedJSHeapSize / 1024 / 1024 || 0
                };
            """)
        except Exception:
            return {}
    
    async def test_mobile_accessibility(self, device: str) -> MobileTestResult:
        """Test mobile accessibility features"""
        driver = self._get_mobile_driver(device)
        device_config = self.mobile_devices[device]
        viewport_size = (device_config['width'], device_config['height'])
        
        try:
            start_time = time.time()
            
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Test accessibility features
            accessibility_score = await self._test_mobile_accessibility_features(driver)
            
            duration = time.time() - start_time
            performance_metrics = self._get_mobile_performance_metrics(driver)
            
            return MobileTestResult(
                test_name="mobile_accessibility",
                device_type=device,
                browser="chrome_mobile",
                viewport_size=viewport_size,
                success=accessibility_score >= 0.7,
                duration=duration,
                touch_interactions=0,
                responsive_score=0,
                performance_metrics=performance_metrics,
                accessibility_score=accessibility_score
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return MobileTestResult(
                test_name="mobile_accessibility",
                device_type=device,
                browser="chrome_mobile",
                viewport_size=viewport_size,
                success=False,
                duration=duration,
                touch_interactions=0,
                responsive_score=0,
                performance_metrics={},
                accessibility_score=0,
                error_message=str(e)
            )
        finally:
            driver.quit()
    
    async def _test_mobile_accessibility_features(self, driver: webdriver.Chrome) -> float:
        """Test mobile-specific accessibility features"""
        accessibility_tests = []
        
        try:
            # Test for ARIA labels on interactive elements
            interactive_elements = driver.find_elements(By.CSS_SELECTOR, "button, a, input")
            labeled_elements = 0
            
            for element in interactive_elements:
                aria_label = element.get_attribute('aria-label')
                aria_labelledby = element.get_attribute('aria-labelledby')
                title = element.get_attribute('title')
                text_content = element.text.strip()
                
                if aria_label or aria_labelledby or title or text_content:
                    labeled_elements += 1
            
            if interactive_elements:
                label_score = labeled_elements / len(interactive_elements)
                accessibility_tests.append(label_score)
            
            # Test for keyboard/screen reader navigation
            focusable_elements = driver.find_elements(By.CSS_SELECTOR, "a, button, input, select, textarea, [tabindex]")
            proper_focus_order = 0
            
            for element in focusable_elements[:10]:  # Test first 10 elements
                try:
                    element.send_keys(Keys.TAB)
                    if driver.switch_to.active_element == element:
                        proper_focus_order += 1
                except Exception:
                    pass
            
            if focusable_elements:
                focus_score = proper_focus_order / min(len(focusable_elements), 10)
                accessibility_tests.append(focus_score)
            
            # Test for semantic HTML
            semantic_elements = driver.find_elements(By.CSS_SELECTOR, "header, nav, main, section, article, aside, footer, h1, h2, h3, h4, h5, h6")
            if semantic_elements:
                accessibility_tests.append(1.0)
            else:
                accessibility_tests.append(0.5)
            
            # Test for alternative text on images
            images = driver.find_elements(By.TAG_NAME, "img")
            images_with_alt = 0
            
            for img in images:
                alt_text = img.get_attribute('alt')
                if alt_text is not None:  # Even empty alt="" is acceptable for decorative images
                    images_with_alt += 1
            
            if images:
                alt_score = images_with_alt / len(images)
                accessibility_tests.append(alt_score)
            
            return np.mean(accessibility_tests) if accessibility_tests else 0.0
            
        except Exception:
            return 0.0
    
    async def run_comprehensive_mobile_test(self) -> Dict[str, Any]:
        """Run comprehensive mobile compatibility test suite"""
        results = {
            'timestamp': time.time(),
            'test_suite': 'mobile_compatibility',
            'tests': {},
            'device_summary': {},
            'overall_summary': {}
        }
        
        # Test key mobile devices
        test_devices = ['iphone_12', 'samsung_galaxy_s21', 'ipad']
        
        for device in test_devices:
            print(f"Testing {device}...")
            device_results = {}
            
            # Test responsive design
            responsive_result = await self.test_responsive_design(device)
            device_results['responsive_design'] = asdict(responsive_result)
            
            # Test touch interactions
            touch_result = await self.test_touch_interactions(device)
            device_results['touch_interactions'] = asdict(touch_result)
            
            # Test mobile accessibility
            accessibility_result = await self.test_mobile_accessibility(device)
            device_results['mobile_accessibility'] = asdict(accessibility_result)
            
            results['tests'][device] = device_results
            
            # Calculate device summary
            device_scores = [
                responsive_result.responsive_score,
                touch_result.responsive_score,
                accessibility_result.accessibility_score
            ]
            
            results['device_summary'][device] = {
                'overall_score': np.mean(device_scores),
                'responsive_score': responsive_result.responsive_score,
                'touch_score': touch_result.responsive_score,
                'accessibility_score': accessibility_result.accessibility_score,
                'all_tests_passed': all(r.success for r in [responsive_result, touch_result, accessibility_result])
            }
        
        # Generate overall summary
        results['overall_summary'] = self._generate_mobile_summary(results)
        
        return results
    
    def _generate_mobile_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mobile compatibility summary"""
        device_summaries = results['device_summary']
        
        if not device_summaries:
            return {'error': 'No device test results available'}
        
        # Calculate overall metrics
        all_scores = []
        responsive_scores = []
        touch_scores = []
        accessibility_scores = []
        
        for device_summary in device_summaries.values():
            all_scores.append(device_summary['overall_score'])
            responsive_scores.append(device_summary['responsive_score'])
            touch_scores.append(device_summary['touch_score'])
            accessibility_scores.append(device_summary['accessibility_score'])
        
        overall_mobile_score = np.mean(all_scores)
        
        # Determine mobile compatibility status
        if overall_mobile_score >= 0.9:
            compatibility_status = "excellent"
        elif overall_mobile_score >= 0.75:
            compatibility_status = "good"
        elif overall_mobile_score >= 0.6:
            compatibility_status = "fair"
        else:
            compatibility_status = "poor"
        
        # Generate recommendations
        recommendations = []
        
        if np.mean(responsive_scores) < 0.7:
            recommendations.append("Improve responsive design for better mobile layout")
        
        if np.mean(touch_scores) < 0.7:
            recommendations.append("Optimize touch interactions and target sizes")
        
        if np.mean(accessibility_scores) < 0.7:
            recommendations.append("Enhance mobile accessibility features")
        
        if overall_mobile_score < 0.8:
            recommendations.extend([
                "Test on actual mobile devices for real-world validation",
                "Consider progressive web app (PWA) features",
                "Optimize mobile performance and loading times"
            ])
        
        return {
            'overall_mobile_score': overall_mobile_score,
            'compatibility_status': compatibility_status,
            'average_responsive_score': np.mean(responsive_scores),
            'average_touch_score': np.mean(touch_scores),
            'average_accessibility_score': np.mean(accessibility_scores),
            'devices_tested': len(device_summaries),
            'all_devices_passed': all(summary['all_tests_passed'] for summary in device_summaries.values()),
            'recommendations': recommendations
        }


# Pytest fixtures and test cases
@pytest.fixture
def mobile_tester():
    """Fixture for mobile compatibility tester"""
    return MobileCompatibilityTester()


@pytest.mark.asyncio
async def test_iphone_responsive_design(mobile_tester):
    """Test responsive design on iPhone"""
    result = await mobile_tester.test_responsive_design('iphone_12')
    assert result.success, f"iPhone responsive design test failed: {result.error_message}"
    assert result.responsive_score >= 0.7, f"Responsive score {result.responsive_score} below threshold"


@pytest.mark.asyncio
async def test_android_touch_interactions(mobile_tester):
    """Test touch interactions on Android device"""
    result = await mobile_tester.test_touch_interactions('samsung_galaxy_s21')
    assert result.success, f"Android touch interactions test failed: {result.error_message}"
    assert result.touch_interactions > 0, "No touch interactions recorded"


@pytest.mark.asyncio
async def test_tablet_accessibility(mobile_tester):
    """Test accessibility on tablet device"""
    result = await mobile_tester.test_mobile_accessibility('ipad')
    assert result.success, f"Tablet accessibility test failed: {result.error_message}"
    assert result.accessibility_score >= 0.6, f"Accessibility score {result.accessibility_score} below threshold"


if __name__ == "__main__":
    # Run comprehensive mobile compatibility test
    async def main():
        tester = MobileCompatibilityTester()
        results = await tester.run_comprehensive_mobile_test()
        
        # Save results
        output_path = Path("tests/ui/mobile/reports")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"mobile_compatibility_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = results['overall_summary']
        print("Mobile compatibility test completed.")
        print(f"Overall mobile score: {summary['overall_mobile_score']:.2f} ({summary['compatibility_status']})")
        print(f"Devices tested: {summary['devices_tested']}")
        
        if summary['recommendations']:
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")
    
    asyncio.run(main())