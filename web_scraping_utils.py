"""
Web Scraping Utilities - Enhanced capabilities for adaptive_agent.py

Features:
- Visual debugging (draw labels on elements)
- Auto data extraction (smart product/form/table detection)
- Pattern recognition for e-commerce sites
"""

from typing import Dict, List, Any


# ============================================================================
# VISUAL DEBUGGING - Label drawing for debugging
# ============================================================================

def draw_labels(page, elements: List[Dict]):
    """
    Draw color-coded labels on web page elements

    Colors:
    - Green: General elements
    - Cyan: Input fields
    - Yellow: Buttons
    - Magenta: Links

    Args:
        page: Playwright page object
        elements: List of element dicts with positions
    """
    page.evaluate("""
        (elements) => {
            // Remove old overlay if exists
            const old = document.getElementById('ai-overlay');
            if (old) old.remove();

            // Create overlay container
            const container = document.createElement('div');
            container.id = 'ai-overlay';
            container.style.cssText = 'position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 999999;';

            elements.forEach(el => {
                // Color code by element type
                let color = '#00ff00';  // Green default
                if (el.tag === 'input') color = '#00ffff';  // Cyan for inputs
                if (el.tag === 'button' || el.role === 'button') color = '#ffff00';  // Yellow for buttons
                if (el.tag === 'a') color = '#ff00ff';  // Magenta for links

                // Draw box around element
                const box = document.createElement('div');
                box.style.cssText = `
                    position: absolute;
                    left: ${el.left}px;
                    top: ${el.top}px;
                    width: ${el.width}px;
                    height: ${el.height}px;
                    border: 2px solid ${color};
                    background: ${color}22;
                    box-sizing: border-box;
                `;

                // Draw label above element
                const label = document.createElement('div');
                label.textContent = el.id;
                label.style.cssText = `
                    position: absolute;
                    left: ${el.left}px;
                    top: ${el.top - 22}px;
                    background: ${color};
                    color: #000;
                    padding: 2px 8px;
                    font-size: 14px;
                    font-weight: bold;
                    font-family: Arial;
                    border-radius: 3px;
                `;

                container.appendChild(box);
                container.appendChild(label);
            });

            document.body.appendChild(container);
        }
    """, elements)


def remove_labels(page):
    """Remove visual labels from page"""
    try:
        page.evaluate("() => { const o = document.getElementById('ai-overlay'); if (o) o.remove(); }")
    except:
        pass


# ============================================================================
# AUTO DATA EXTRACTION - Smart extraction for e-commerce and forms
# ============================================================================

def extract_structured_data(page, data_type: str = 'auto') -> Dict[str, Any]:
    """
    Extract structured data from any page intelligently

    Auto-detects:
    - Products (name, price, rating, reviews, images)
    - Forms (fields, actions, methods)
    - Tables (headers, rows)
    - Metadata (title, URL, page type)

    Args:
        page: Playwright page object
        data_type: 'auto', 'product', 'form', 'table', or 'general'

    Returns:
        Dict with extracted data:
        {
            'products': [...],
            'listings': [...],
            'forms': [...],
            'tables': [...],
            'metadata': {...}
        }
    """

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

            // ================================================================
            // PRODUCT EXTRACTION - Multiple selector strategies
            // ================================================================
            const itemSelectors = [
                '[data-item-id]',
                '[data-product-id]',
                '[data-asin]',  // Amazon
                '.product-item',
                '.search-result-item',
                '[itemtype*="Product"]',  // Schema.org markup
                'article',
                '.listing-item',
                '[data-component-type*="product"]'
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
                            availability: '',
                            attributes: {}
                        };

                        // Name extraction (multiple strategies)
                        const nameEls = item.querySelectorAll('h1, h2, h3, h4, [class*="title"], [class*="name"], a[title]');
                        for (const el of nameEls) {
                            const text = (el.textContent || el.getAttribute('title') || '').trim();
                            if (text && text.length > 5 && text.length < 200) {
                                product.name = text;
                                break;
                            }
                        }

                        // Price extraction (handles multiple formats)
                        const priceEls = item.querySelectorAll('[class*="price"], [itemprop="price"], [data-price], [aria-label*="price"]');
                        for (const el of priceEls) {
                            const priceText = el.textContent || el.getAttribute('content') || el.getAttribute('data-price') || el.getAttribute('aria-label') || '';
                            // Match $123.45, 123.45, $123, etc.
                            const match = priceText.match(/[\$£€]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)/);
                            if (match) {
                                product.price = parseFloat(match[1].replace(',', ''));
                                break;
                            }
                        }

                        // Rating extraction (multiple formats)
                        const ratingEls = item.querySelectorAll('[class*="rating"], [class*="star"], [aria-label*="star"], [itemprop="ratingValue"]');
                        for (const el of ratingEls) {
                            const ratingText = el.getAttribute('aria-label') || el.textContent || el.getAttribute('content') || '';
                            // Match "4.5 out of 5", "4.5 stars", etc.
                            const match = ratingText.match(/(\d+\.?\d*)\s*(?:out of|stars?|\/)/i);
                            if (match) {
                                product.rating = parseFloat(match[1]);
                                break;
                            }
                        }

                        // Review count extraction
                        const reviewEls = item.querySelectorAll('[class*="review"], [aria-label*="review"], [itemprop="reviewCount"]');
                        for (const el of reviewEls) {
                            const reviewText = el.textContent || el.getAttribute('aria-label') || el.getAttribute('content') || '';
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

                        // Availability extraction
                        const availEls = item.querySelectorAll('[class*="stock"], [class*="availability"], [itemprop="availability"]');
                        for (const el of availEls) {
                            const availText = el.textContent || el.getAttribute('content') || '';
                            if (availText.toLowerCase().includes('in stock')) {
                                product.availability = 'in_stock';
                                break;
                            } else if (availText.toLowerCase().includes('out of stock')) {
                                product.availability = 'out_of_stock';
                                break;
                            }
                        }

                        // Only add if we got meaningful data
                        if (product.name || product.price) {
                            extracted.products.push(product);
                        }
                    });
                    break; // Found products, stop looking
                }
            }

            // ================================================================
            // FORM EXTRACTION
            // ================================================================
            document.querySelectorAll('form').forEach((form, idx) => {
                const formData = {
                    index: idx,
                    action: form.action,
                    method: form.method,
                    id: form.id,
                    name: form.name,
                    fields: []
                };

                form.querySelectorAll('input, select, textarea').forEach(field => {
                    formData.fields.push({
                        name: field.name,
                        type: field.type,
                        id: field.id,
                        placeholder: field.placeholder,
                        required: field.required,
                        value: field.value
                    });
                });

                extracted.forms.push(formData);
            });

            // ================================================================
            // TABLE EXTRACTION
            // ================================================================
            document.querySelectorAll('table').forEach((table, idx) => {
                const tableData = {
                    index: idx,
                    headers: [],
                    rows: []
                };

                // Extract headers
                table.querySelectorAll('th').forEach(th => {
                    tableData.headers.push(th.textContent.trim());
                });

                // Extract rows
                table.querySelectorAll('tr').forEach(tr => {
                    const cells = [];
                    tr.querySelectorAll('td').forEach(td => {
                        cells.push(td.textContent.trim());
                    });
                    if (cells.length > 0) {
                        tableData.rows.push(cells);
                    }
                });

                if (tableData.headers.length > 0 || tableData.rows.length > 0) {
                    extracted.tables.push(tableData);
                }
            });

            // ================================================================
            // METADATA EXTRACTION
            // ================================================================
            extracted.metadata = {
                title: document.title,
                url: window.location.href,
                domain: window.location.hostname,
                pageType: pageType,
                hasProducts: extracted.products.length > 0,
                hasForms: extracted.forms.length > 0,
                hasTables: extracted.tables.length > 0,
                timestamp: new Date().toISOString()
            };

            return extracted;
        }
    """, data_type)

    return data


def deduplicate_products(products: List[Dict], key: str = 'url') -> List[Dict]:
    """
    Remove duplicate products based on key field

    Args:
        products: List of product dicts
        key: Field to use for de-duplication ('url', 'name', etc.)

    Returns:
        De-duplicated list of products
    """
    seen = set()
    unique_products = []

    for product in products:
        identifier = product.get(key)
        if identifier and identifier not in seen:
            seen.add(identifier)
            unique_products.append(product)

    return unique_products


def filter_products(
    products: List[Dict],
    min_price: float = None,
    max_price: float = None,
    min_rating: float = None,
    min_reviews: int = None
) -> List[Dict]:
    """
    Filter products by criteria

    Args:
        products: List of product dicts
        min_price: Minimum price
        max_price: Maximum price
        min_rating: Minimum rating (e.g., 4.0)
        min_reviews: Minimum number of reviews

    Returns:
        Filtered list of products
    """
    filtered = products

    if min_price is not None:
        filtered = [p for p in filtered if p.get('price') and p['price'] >= min_price]

    if max_price is not None:
        filtered = [p for p in filtered if p.get('price') and p['price'] <= max_price]

    if min_rating is not None:
        filtered = [p for p in filtered if p.get('rating') and p['rating'] >= min_rating]

    if min_reviews is not None:
        filtered = [p for p in filtered if p.get('reviews') and p['reviews'] >= min_reviews]

    return filtered


# ============================================================================
# SMART ELEMENT DETECTION
# ============================================================================

def detect_elements_smart(page) -> List[Dict]:
    """
    Smart element detection with pattern recognition

    Detects ALL interactive elements including:
    - Links, buttons, inputs
    - Role-based elements (ARIA)
    - Clickable elements (onclick handlers)
    - Data attributes

    Returns:
        List of element dicts with comprehensive metadata
    """

    elements = page.evaluate("""
        () => {
            const elements = [];
            let id = 1;

            // Detect ALL interactive elements
            const selectors = [
                'a', 'button', 'input', 'textarea', 'select',
                '[role="button"]', '[role="link"]', '[role="tab"]',
                '[role="menuitem"]', '[onclick]', '[role="slider"]',
                '.clickable', '[data-action]', '[data-testid]',
                '[data-cy]',  // Cypress test IDs
                '[data-qa]'   // QA test IDs
            ];

            const allEls = document.querySelectorAll(selectors.join(','));

            allEls.forEach(el => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                // Check if element is visible and in viewport (with buffer)
                if (rect.width > 0 && rect.height > 0 &&
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    rect.top < window.innerHeight + 300 &&
                    rect.bottom > -300) {

                    // Extract comprehensive data
                    const text = (el.innerText || el.value || el.placeholder ||
                                 el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();

                    // Get all data attributes
                    const dataAttributes = {};
                    Array.from(el.attributes)
                        .filter(attr => attr.name.startsWith('data-'))
                        .forEach(attr => {
                            dataAttributes[attr.name] = attr.value;
                        });

                    elements.push({
                        id: id++,
                        tag: el.tagName.toLowerCase(),
                        text: text.substring(0, 150),
                        type: el.type || '',
                        role: el.getAttribute('role') || '',
                        className: Array.from(el.classList).join(' ').substring(0, 100),
                        href: el.href || '',
                        dataAttributes: dataAttributes,
                        x: Math.round(rect.left + rect.width / 2),
                        y: Math.round(rect.top + rect.height / 2),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                        top: Math.round(rect.top),
                        left: Math.round(rect.left),
                        visible: rect.top >= 0 && rect.top <= window.innerHeight,
                        zIndex: parseInt(style.zIndex) || 0
                    });
                }
            });

            return elements;
        }
    """)

    return elements
