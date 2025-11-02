# adaptive_agent.py - SELF-LEARNING WEB AGENT WITH GUARANTEED RESULTS
from playwright.sync_api import sync_playwright
import anthropic
import base64
import time
import random
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional

client = anthropic.Anthropic()

# ============ LEARNING DATABASE ============
def init_learning_db():
    """Initialize learning database - stores what works and what doesn't"""
    conn = sqlite3.connect('agent_learning.db')
    cursor = conn.cursor()
    
    # Store successful action patterns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS success_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT,
            website_domain TEXT,
            action_sequence TEXT,
            success_rate REAL DEFAULT 1.0,
            times_used INTEGER DEFAULT 1,
            avg_steps INTEGER,
            last_used TEXT,
            notes TEXT
        )
    ''')
    
    # Store failed attempts to learn from
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT,
            website_domain TEXT,
            attempted_action TEXT,
            error_type TEXT,
            timestamp TEXT,
            context TEXT
        )
    ''')
    
    # Store extracted data/results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            task TEXT,
            result_type TEXT,
            result_data TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    
    # Store website patterns learned
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS site_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT,
            element_pattern TEXT,
            pattern_type TEXT,
            selector TEXT,
            reliability REAL,
            last_verified TEXT
        )
    ''')
    
    # Agent's memory/context across sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            context TEXT,
            updated TEXT
        )
    ''')
    
    conn.commit()
    return conn

def learn_from_success(conn, task_type: str, domain: str, actions: List[str], steps: int):
    """Record successful pattern for future use"""
    cursor = conn.cursor()
    action_seq = json.dumps(actions)
    
    # Check if pattern exists
    cursor.execute('''
        SELECT id, success_rate, times_used FROM success_patterns 
        WHERE task_type = ? AND website_domain = ? AND action_sequence = ?
    ''', (task_type, domain, action_seq))
    
    existing = cursor.fetchone()
    
    if existing:
        # Update existing pattern
        pattern_id, success_rate, times_used = existing
        new_times = times_used + 1
        new_success_rate = (success_rate * times_used + 1.0) / new_times
        
        cursor.execute('''
            UPDATE success_patterns 
            SET success_rate = ?, times_used = ?, avg_steps = ?, last_used = ?
            WHERE id = ?
        ''', (new_success_rate, new_times, steps, datetime.now().isoformat(), pattern_id))
    else:
        # Insert new pattern
        cursor.execute('''
            INSERT INTO success_patterns 
            (task_type, website_domain, action_sequence, success_rate, times_used, avg_steps, last_used)
            VALUES (?, ?, ?, 1.0, 1, ?, ?)
        ''', (task_type, domain, action_seq, steps, datetime.now().isoformat()))
    
    conn.commit()

def learn_from_failure(conn, task_type: str, domain: str, action: str, error: str, context: str):
    """Record what didn't work"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO failures (task_type, website_domain, attempted_action, error_type, timestamp, context)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (task_type, domain, action, error, datetime.now().isoformat(), context))
    conn.commit()

def get_learned_strategies(conn, task_type: str, domain: str) -> List[Dict]:
    """Retrieve proven successful strategies"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT action_sequence, success_rate, times_used, avg_steps, notes
        FROM success_patterns
        WHERE task_type = ? AND website_domain = ?
        ORDER BY success_rate DESC, times_used DESC
        LIMIT 3
    ''', (task_type, domain))
    
    strategies = []
    for row in cursor.fetchall():
        strategies.append({
            'actions': json.loads(row[0]),
            'success_rate': row[1],
            'times_used': row[2],
            'avg_steps': row[3],
            'notes': row[4]
        })
    
    return strategies

def save_result(conn, session_id: str, task: str, result_data: any, confidence: float):
    """Save final results with confidence score"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (session_id, task, result_type, result_data, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, task, 'completion', json.dumps(result_data), confidence, datetime.now().isoformat()))
    conn.commit()

# ============ INTELLIGENT ELEMENT DETECTION ============
def detect_elements_smart(page):
    """Smart element detection with pattern recognition"""
    
    elements = page.evaluate("""
        () => {
            const elements = [];
            let id = 1;
            
            // Detect ALL interactive elements
            const selectors = [
                'a', 'button', 'input', 'textarea', 'select',
                '[role="button"]', '[role="link"]', '[role="tab"]',
                '[role="menuitem"]', '[onclick]', '[role="slider"]',
                '.clickable', '[data-action]', '[data-testid]'
            ];
            
            const allEls = document.querySelectorAll(selectors.join(','));
            
            allEls.forEach(el => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                
                if (rect.width > 0 && rect.height > 0 && 
                    style.display !== 'none' && 
                    style.visibility !== 'hidden' &&
                    rect.top < window.innerHeight + 300 &&
                    rect.bottom > -300) {
                    
                    // Extract comprehensive data
                    const text = (el.innerText || el.value || el.placeholder || 
                                 el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
                    
                    elements.push({
                        id: id++,
                        tag: el.tagName.toLowerCase(),
                        text: text.substring(0, 150),
                        type: el.type || '',
                        role: el.getAttribute('role') || '',
                        className: Array.from(el.classList).join(' ').substring(0, 100),
                        href: el.href || '',
                        dataAttributes: JSON.stringify(Object.fromEntries(
                            Array.from(el.attributes)
                                .filter(attr => attr.name.startsWith('data-'))
                                .map(attr => [attr.name, attr.value])
                        )),
                        x: Math.round(rect.left + rect.width / 2),
                        y: Math.round(rect.top + rect.height / 2),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                        top: Math.round(rect.top),
                        left: Math.round(rect.left),
                        visible: rect.top >= 0 && rect.top <= window.innerHeight
                    });
                }
            });
            
            return elements;
        }
    """)
    
    return elements

def extract_structured_data(page, data_type: str = 'auto'):
    """Extract structured data from any page intelligently"""
    
    data = page.evaluate("""
        (dataType) => {
            const extracted = {
                products: [],
                listings: [],
                forms: [],
                tables: [],
                metadata: {}
            };
            
            // Auto-detect page type
            const pageType = dataType === 'auto' ? 
                document.querySelector('[data-testid*="product"]') ? 'product' :
                document.querySelector('.search-result, .product-grid') ? 'search' :
                document.querySelector('form') ? 'form' :
                'general' : dataType;
            
            // Extract products/items
            const itemSelectors = [
                '[data-item-id]',
                '[data-product-id]',
                '[data-asin]',
                '.product-item',
                '.search-result-item',
                '[itemtype*="Product"]',
                'article',
                '.listing-item'
            ];
            
            for (const selector of itemSelectors) {
                const items = document.querySelectorAll(selector);
                if (items.length > 0) {
                    items.forEach((item, idx) => {
                        const product = {
                            index: idx,
                            name: '',
                            price: null,
                            rating: null,
                            reviews: null,
                            url: '',
                            image: '',
                            attributes: {}
                        };
                        
                        // Name extraction
                        const nameEls = item.querySelectorAll('h1, h2, h3, h4, [class*="title"], [class*="name"], a[title]');
                        for (const el of nameEls) {
                            const text = (el.textContent || el.getAttribute('title') || '').trim();
                            if (text && text.length > 5 && text.length < 200) {
                                product.name = text;
                                break;
                            }
                        }
                        
                        // Price extraction
                        const priceEls = item.querySelectorAll('[class*="price"], [itemprop="price"], [data-price]');
                        for (const el of priceEls) {
                            const priceText = el.textContent || el.getAttribute('content') || el.getAttribute('data-price') || '';
                            const match = priceText.match(/[\d,]+\.?\d{0,2}/);
                            if (match) {
                                product.price = parseFloat(match[0].replace(',', ''));
                                break;
                            }
                        }
                        
                        // Rating extraction
                        const ratingEls = item.querySelectorAll('[class*="rating"], [class*="star"], [aria-label*="star"]');
                        for (const el of ratingEls) {
                            const ratingText = el.getAttribute('aria-label') || el.textContent || '';
                            const match = ratingText.match(/(\d+\.?\d*)\s*(out of|stars?)/i);
                            if (match) {
                                product.rating = parseFloat(match[1]);
                                break;
                            }
                        }
                        
                        // Review count extraction
                        const reviewEls = item.querySelectorAll('[class*="review"], [aria-label*="review"]');
                        for (const el of reviewEls) {
                            const reviewText = el.textContent || el.getAttribute('aria-label') || '';
                            const match = reviewText.match(/([\d,]+)\s*review/i);
                            if (match) {
                                product.reviews = parseInt(match[1].replace(',', ''));
                                break;
                            }
                        }
                        
                        // URL extraction
                        const link = item.querySelector('a[href]');
                        if (link) {
                            product.url = link.href;
                        }
                        
                        // Image extraction
                        const img = item.querySelector('img[src]');
                        if (img) {
                            product.image = img.src;
                        }
                        
                        // Only add if we got meaningful data
                        if (product.name || product.price) {
                            extracted.products.push(product);
                        }
                    });
                    break; // Found products, stop looking
                }
            }
            
            // Extract forms
            document.querySelectorAll('form').forEach((form, idx) => {
                const formData = {
                    index: idx,
                    action: form.action,
                    method: form.method,
                    fields: []
                };
                
                form.querySelectorAll('input, select, textarea').forEach(field => {
                    formData.fields.push({
                        name: field.name,
                        type: field.type,
                        placeholder: field.placeholder,
                        required: field.required
                    });
                });
                
                extracted.forms.push(formData);
            });
            
            // Extract metadata
            extracted.metadata = {
                title: document.title,
                url: window.location.href,
                domain: window.location.hostname,
                pageType: pageType,
                hasProducts: extracted.products.length > 0,
                hasForms: extracted.forms.length > 0
            };
            
            return extracted;
        }
    """, data_type)
    
    return data

# ============ REFLECTION & ADAPTATION ============
class AgentReflection:
    """Agent's ability to reflect on actions and adapt"""
    
    def __init__(self, conn):
        self.conn = conn
        self.action_history = []
        self.stuck_threshold = 5  # If same action 5 times, we're stuck
        self.progress_metrics = {
            'data_extracted': 0,
            'pages_visited': 0,
            'successful_actions': 0,
            'failed_actions': 0
        }
    
    def record_action(self, action: str, success: bool, result: any = None):
        """Record each action and outcome"""
        self.action_history.append({
            'action': action,
            'success': success,
            'result': result,
            'timestamp': time.time()
        })
        
        if success:
            self.progress_metrics['successful_actions'] += 1
        else:
            self.progress_metrics['failed_actions'] += 1
    
    def is_stuck(self) -> Tuple[bool, str]:
        """Detect if agent is stuck in a loop"""
        if len(self.action_history) < self.stuck_threshold:
            return False, ""
        
        recent = self.action_history[-self.stuck_threshold:]
        actions = [a['action'] for a in recent]
        
        # Check for repetitive actions
        if len(set(actions)) <= 2:
            return True, f"Repeating same actions: {set(actions)}"
        
        # Check for no progress
        recent_successes = [a['success'] for a in recent]
        if sum(recent_successes) == 0:
            return True, "No successful actions in recent steps"
        
        return False, ""
    
    def suggest_alternative(self, current_strategy: str) -> str:
        """Suggest different approach when stuck"""
        suggestions = {
            'clicking': 'Try extracting data directly without clicking individual items',
            'scrolling': 'Try using search or filters instead of scrolling',
            'typing': 'Try using buttons or navigation instead',
            'navigation': 'Try going back to homepage and using different path'
        }
        
        return suggestions.get(current_strategy, 'Try a completely different approach')
    
    def get_progress_summary(self) -> str:
        """Summarize current progress"""
        return f"""
📊 PROGRESS METRICS:
   ✓ Successful actions: {self.progress_metrics['successful_actions']}
   ✗ Failed actions: {self.progress_metrics['failed_actions']}
   📄 Pages visited: {self.progress_metrics['pages_visited']}
   📦 Data extracted: {self.progress_metrics['data_extracted']} items
   
   Success rate: {self.progress_metrics['successful_actions'] / max(1, len(self.action_history)) * 100:.1f}%
"""

# ============ VISUAL HELPERS ============
def draw_labels(page, elements):
    """Enhanced labels with priorities"""
    page.evaluate("""
        (elements) => {
            const old = document.getElementById('ai-overlay');
            if (old) old.remove();
            
            const container = document.createElement('div');
            container.id = 'ai-overlay';
            container.style.cssText = 'position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 999999;';
            
            elements.forEach(el => {
                // Color code by element type
                let color = '#00ff00';
                if (el.tag === 'input') color = '#00ffff';
                if (el.tag === 'button' || el.role === 'button') color = '#ffff00';
                if (el.tag === 'a') color = '#ff00ff';
                
                const box = document.createElement('div');
                box.style.cssText = `position: absolute; left: ${el.left}px; top: ${el.top}px; width: ${el.width}px; height: ${el.height}px; border: 2px solid ${color}; background: ${color}22; box-sizing: border-box;`;
                
                const label = document.createElement('div');
                label.textContent = el.id;
                label.style.cssText = `position: absolute; left: ${el.left}px; top: ${el.top - 22}px; background: ${color}; color: #000; padding: 2px 8px; font-size: 14px; font-weight: bold; font-family: Arial; border-radius: 3px;`;
                
                container.appendChild(box);
                container.appendChild(label);
            });
            
            document.body.appendChild(container);
        }
    """, elements)

def remove_labels(page):
    try:
        page.evaluate("() => { const o = document.getElementById('ai-overlay'); if (o) o.remove(); }")
    except:
        pass

# ============ MAIN ADAPTIVE AGENT ============
def adaptive_agent(task: str):
    """Self-learning agent that adapts and guarantees results"""
    
    # Initialize learning database
    learning_db = init_learning_db()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    reflection = AgentReflection(learning_db)
    
    print(f"🧠 ADAPTIVE WEB AGENT - Session: {session_id}")
    print(f"🎯 TASK: {task}\n")
    
    # Extract task type and domain
    task_type = "search" if "search" in task.lower() or "find" in task.lower() else "navigate"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        
        conversation_history = []
        collected_data = []
        last_url = ""
        current_domain = ""
        action_sequence = []
        
        MAX_STEPS = 40
        
        for step in range(MAX_STEPS):
            print(f"\n{'='*70}\nSTEP {step + 1}/{MAX_STEPS}\n{'='*70}")
            
            time.sleep(random.uniform(1.0, 1.8))
            
            # Check if URL changed
            current_url = page.url
            if current_url != last_url:
                current_domain = page.evaluate("() => window.location.hostname")
                print(f"📍 URL: {current_url[:80]}")
                last_url = current_url
                reflection.progress_metrics['pages_visited'] += 1
                
                # Auto-extract data from new page
                extracted = extract_structured_data(page)
                if extracted['products']:
                    print(f"🔍 Auto-extracted {len(extracted['products'])} items from page")
                    collected_data.extend(extracted['products'])
                    reflection.progress_metrics['data_extracted'] = len(collected_data)
            
            # Check if stuck
            is_stuck, stuck_reason = reflection.is_stuck()
            if is_stuck:
                print(f"⚠️ STUCK DETECTED: {stuck_reason}")
                print(f"💡 Suggestion: {reflection.suggest_alternative('navigation')}")
            
            # Detect elements
            elements = detect_elements_smart(page)
            print(f"🔍 Detected {len(elements)} interactive elements")
            
            # Get learned strategies
            strategies = get_learned_strategies(learning_db, task_type, current_domain)
            strategy_text = ""
            if strategies:
                strategy_text = "\n\n🎓 LEARNED STRATEGIES for this site:\n"
                for i, s in enumerate(strategies, 1):
                    strategy_text += f"{i}. {' → '.join(s['actions'][:3])} (Success: {s['success_rate']*100:.0f}%, Used: {s['times_used']}x)\n"
            
            # Draw labels
            draw_labels(page, elements)
            time.sleep(0.4)
            
            # Screenshot
            try:
                screenshot = page.screenshot()
                screenshot_b64 = base64.b64encode(screenshot).decode()
            except:
                break
            
            remove_labels(page)
            
            # Build focused element list (prioritize visible, important elements)
            visible_elements = [e for e in elements if e['visible']][:30]
            elem_list = []
            for el in visible_elements:
                desc = f"[{el['id']}] {el['tag']}"
                if el['type']: desc += f" type={el['type']}"
                if el['role']: desc += f" role={el['role']}"
                if el['text']: desc += f": {el['text'][:60]}"
                elem_list.append(desc)
            
            # Build results summary
            results_summary = ""
            if collected_data:
                results_summary = f"\n\n📦 COLLECTED DATA ({len(collected_data)} items):\n"
                for i, item in enumerate(collected_data[:5], 1):
                    results_summary += f"{i}. {item.get('name', 'Unknown')[:50]}"
                    if item.get('price'): results_summary += f" - ${item['price']}"
                    if item.get('rating'): results_summary += f" ⭐{item['rating']}"
                    if item.get('reviews'): results_summary += f" ({item['reviews']} reviews)"
                    results_summary += "\n"
                if len(collected_data) > 5:
                    results_summary += f"... and {len(collected_data) - 5} more items\n"
            
            # Enhanced prompt with learning and reflection
            prompt = f"""You are an ADAPTIVE web agent that LEARNS and DELIVERS RESULTS.

🎯 TASK: {task}

📍 Current URL: {page.url}
🌐 Domain: {current_domain}

{reflection.get_progress_summary()}{results_summary}{strategy_text}

🔍 VISIBLE ELEMENTS (color-coded on screenshot):
{chr(10).join(elem_list)}

AVAILABLE ACTIONS:
• goto <url> - Navigate to URL
• click <id> - Click element
• type <text> - Type in search/input (auto-finds input)
• extract - Extract ALL data from current page intelligently
• analyze - Analyze collected data and generate final results
• done - Task complete with results

🧠 INTELLIGENCE GUIDELINES:
1. **Learn from context**: Use EXTRACTED DATA above to avoid redundant clicks
2. **Be efficient**: Use 'extract' to gather data, don't click every item
3. **Adapt if stuck**: If same action isn't working, try different approach
4. **Validate results**: Before 'done', ensure you have meaningful data
5. **Think strategically**: What's the fastest path to complete results?

⚡ CRITICAL: Your goal is RESULTS, not just actions. Don't say "done" without deliverables!

What's your next intelligent action?

ACTION: [goto/click/type/extract/analyze/done]
DETAILS: [specific details]
REASON: [strategic reasoning - why this moves us toward RESULTS]"""

            messages = conversation_history + [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Get Claude's decision
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                messages=messages
            )
            
            answer = response.content[0].text
            print(f"\n🤖 AGENT DECISION:\n{answer}\n")
            
            conversation_history.append({"role": "assistant", "content": answer})
            
            # Parse action
            action = None
            details = None
            for line in answer.split('\n'):
                upper = line.strip().upper()
                if upper.startswith('ACTION:'):
                    action = line.split(':', 1)[1].strip().lower()
                if upper.startswith('DETAILS:'):
                    details = line.split(':', 1)[1].strip()
            
            if not action:
                reflection.record_action('parse_error', False)
                continue
            
            action_sequence.append(action)
            print(f"⚡ EXECUTING: {action.upper()}")
            if details:
                print(f"   Details: {details}")
            
            # Execute action
            success = False
            try:
                if action == "done":
                    if not collected_data:
                        print("❌ REJECTED: Cannot complete without results!")
                        print("   Continuing to gather data...")
                        reflection.record_action('done_without_results', False)
                        continue
                    
                    print("\n" + "="*70)
                    print("✅ TASK COMPLETED WITH RESULTS")
                    print("="*70)
                    
                    # Save results
                    save_result(learning_db, session_id, task, collected_data, 0.95)
                    learn_from_success(learning_db, task_type, current_domain, action_sequence, step + 1)
                    
                    # Display final results
                    print(f"\n📊 FINAL RESULTS ({len(collected_data)} items):")
                    for i, item in enumerate(collected_data, 1):
                        print(f"\n{i}. {item.get('name', 'Unknown')}")
                        if item.get('price'): print(f"   💰 ${item['price']}")
                        if item.get('rating'): print(f"   ⭐ {item['rating']}/5")
                        if item.get('reviews'): print(f"   💬 {item['reviews']:,} reviews")
                        if item.get('url'): print(f"   🔗 {item['url'][:60]}...")
                    
                    success = True
                    reflection.record_action('done', True, collected_data)
                    break
                
                elif action == "extract":
                    extracted = extract_structured_data(page)
                    new_items = extracted['products']
                    
                    if new_items:
                        # Avoid duplicates
                        existing_urls = {item.get('url') for item in collected_data}
                        new_items = [item for item in new_items if item.get('url') not in existing_urls]
                        
                        collected_data.extend(new_items)
                        print(f"✓ Extracted {len(new_items)} new items (Total: {len(collected_data)})")
                        reflection.progress_metrics['data_extracted'] = len(collected_data)
                        success = True
                    else:
                        print("⚠️ No data extracted - may need different approach")
                        learn_from_failure(learning_db, task_type, current_domain, 'extract', 
                                         'no_data', json.dumps({'url': current_url}))
                    
                    reflection.record_action('extract', success, len(new_items) if success else 0)
                    time.sleep(1)
                
                elif action == "analyze":
                    if not collected_data:
                        print("❌ No data to analyze yet!")
                        reflection.record_action('analyze_no_data', False)
                        continue
                    
                    print("\n" + "="*70)
                    print("📊 DATA ANALYSIS")
                    print("="*70)
                    print(f"Total items collected: {len(collected_data)}\n")
                    
                    # Sort by relevance
                    sorted_data = sorted(collected_data, 
                                       key=lambda x: (x.get('rating', 0), x.get('reviews', 0)), 
                                       reverse=True)
                    
                    print("🏆 TOP MATCHES:")
                    for i, item in enumerate(sorted_data[:10], 1):
                        print(f"\n{i}. {item.get('name', 'Unknown')}")
                        if item.get('price'): print(f"   💰 ${item['price']}")
                        if item.get('rating'): print(f"   ⭐ {item['rating']}/5")
                        if item.get('reviews'): print(f"   💬 {item['reviews']:,} reviews")
                    
                    success = True
                    reflection.record_action('analyze', True)
                    time.sleep(2)
                
                elif action == "goto":
                    page.goto(details, wait_until='domcontentloaded', timeout=30000)
                    success = True
                    reflection.record_action('goto', True)
                    time.sleep(random.uniform(1.5, 2.5))
                
                elif action == "type":
                    inputs = [e for e in elements if e['tag'] == 'input' and e['type'] in ['text', 'search', '']]
                    if inputs:
                        el = inputs[0]
                        page.mouse.click(el['x'], el['y'])
                        time.sleep(0.3)
                        page.keyboard.press('Control+A')
                        page.keyboard.type(details, delay=random.randint(60, 120))
                        time.sleep(0.3)
                        page.keyboard.press('Enter')
                        print(f"✓ Typed: {details}")
                        success = True
                        reflection.record_action('type', True)
                    else:
                        print("✗ No input field found")
                        reflection.record_action('type_no_input', False)
                    time.sleep(random.uniform(2, 3))
                
                elif action == "click":
                    elem_id = int(details.strip())
                    target = next((e for e in elements if e['id'] == elem_id), None)
                    
                    if target:
                        # Scroll to element if needed
                        if not target['visible']:
                            page.evaluate(f"window.scrollTo({{top: {target['top'] - 300}, behavior: 'smooth'}})")
                            time.sleep(1)
                        
                        page.mouse.click(target['x'], target['y'])
                        print(f"✓ Clicked [{elem_id}]: {target['text'][:40]}")
                        success = True
                        reflection.record_action('click', True)
                    else:
                        print(f"✗ Element {elem_id} not found")
                        reflection.record_action('click_not_found', False)
                    
                    time.sleep(random.uniform(1.5, 2.5))
                
            except Exception as e:
                error_msg = str(e)[:100]
                print(f"✗ Error: {error_msg}")
                learn_from_failure(learning_db, task_type, current_domain, action, 
                                 error_msg, json.dumps({'step': step, 'url': current_url}))
                reflection.record_action(action, False)
                time.sleep(1)
        
        # End of steps
        if step == MAX_STEPS - 1:
            print("\n⚠️ Reached maximum steps")
            if collected_data:
                print(f"✓ But successfully collected {len(collected_data)} items")
                save_result(learning_db, session_id, task, collected_data, 0.8)
            else:
                print("❌ No results collected - task incomplete")
        
        print(f"\n{reflection.get_progress_summary()}")
        
        input("\n\nPress Enter to close browser...")
        browser.close()
    
    learning_db.close()
    print(f"\n💾 Learning data saved to: agent_learning.db")
    print(f"📊 Session: {session_id}")

if __name__ == "__main__":
    task = input("What should I do? ")
    if not task:
        task = "go to walmart.com and find me queen bed frames under $250 with at least 1500 reviews and 4+ stars"
    
    adaptive_agent(task)