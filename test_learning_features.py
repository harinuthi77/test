"""
Test script for new learning features

Tests:
- LearningDatabase (SQLite learning)
- AgentReflection (stuck detection)
- Web scraping utilities
"""

import sys
import time
from agent_framework import LearningDatabase, AgentReflection

def test_learning_database():
    """Test learning database functionality"""
    print("="*70)
    print("TESTING: LearningDatabase")
    print("="*70)

    db = LearningDatabase('test_agent_learning.db')

    # Test 1: Learn from success
    db.learn_from_success(
        task_type="search",
        domain="amazon.com",
        actions=["goto", "type", "click", "extract"],
        steps=4,
        notes="Found products successfully"
    )
    print("✓ Test 1: learn_from_success()")

    # Test 2: Get learned strategies
    strategies = db.get_learned_strategies("search", "amazon.com")
    assert len(strategies) == 1, "Should have 1 strategy"
    assert strategies[0]['success_rate'] == 1.0, "Success rate should be 1.0"
    assert strategies[0]['times_used'] == 1, "Times used should be 1"
    print("✓ Test 2: get_learned_strategies()")
    print(f"  Strategy: {strategies[0]['actions']}")
    print(f"  Success rate: {strategies[0]['success_rate']}")
    print(f"  Times used: {strategies[0]['times_used']}")

    # Test 3: Learn from failure
    db.learn_from_failure(
        task_type="search",
        domain="example.com",
        action="click_button",
        error="element_not_found",
        context='{"url": "https://example.com/search"}'
    )
    print("✓ Test 3: learn_from_failure()")

    # Test 4: Save result
    db.save_result(
        session_id="test_session_001",
        task="find products",
        result_data={"products": [{"name": "Test Product", "price": 29.99}]},
        confidence=0.95
    )
    print("✓ Test 4: save_result()")

    # Test 5: Memory operations
    db.set_memory("last_search_query", "queen bed frames", "search context")
    memory = db.get_memory("last_search_query")
    assert memory == "queen bed frames", "Memory should match"
    print("✓ Test 5: Memory operations")
    print(f"  Stored and retrieved: {memory}")

    # Test 6: Update existing strategy (should increment times_used)
    db.learn_from_success(
        task_type="search",
        domain="amazon.com",
        actions=["goto", "type", "click", "extract"],
        steps=5
    )
    strategies = db.get_learned_strategies("search", "amazon.com")
    assert strategies[0]['times_used'] == 2, "Times used should be 2"
    print("✓ Test 6: Strategy updates correctly")
    print(f"  Times used: {strategies[0]['times_used']}")
    print(f"  Success rate: {strategies[0]['success_rate']:.2f}")

    db.close()

    print("\n" + "="*70)
    print("✅ ALL LEARNING DATABASE TESTS PASSED")
    print("="*70)


def test_agent_reflection():
    """Test agent reflection functionality"""
    print("\n" + "="*70)
    print("TESTING: AgentReflection")
    print("="*70)

    reflection = AgentReflection()

    # Test 1: Record actions
    reflection.record_action("goto", True, None, {"url": "example.com"})
    reflection.record_action("type", True, None, {"text": "search query"})
    reflection.record_action("click", False, None, {"element": "button"})
    print("✓ Test 1: record_action()")
    print(f"  Successful actions: {reflection.progress_metrics['successful_actions']}")
    print(f"  Failed actions: {reflection.progress_metrics['failed_actions']}")

    # Test 2: Not stuck (only 3 actions)
    is_stuck, reason = reflection.is_stuck()
    assert not is_stuck, "Should not be stuck with only 3 actions"
    print("✓ Test 2: is_stuck() - not stuck yet")

    # Test 3: Simulate stuck state (repetitive actions)
    for i in range(5):
        reflection.record_action("click", False, None, {"url": "example.com"})
    is_stuck, reason = reflection.is_stuck()
    assert is_stuck, "Should detect stuck state"
    print("✓ Test 3: is_stuck() - detects repetitive actions")
    print(f"  Reason: {reason}")

    # Test 4: Suggest alternative
    suggestion = reflection.suggest_alternative("clicking")
    print("✓ Test 4: suggest_alternative()")
    print(f"  Suggestion: {suggestion}")

    # Test 5: Progress summary
    summary = reflection.get_progress_summary()
    print("✓ Test 5: get_progress_summary()")
    print(summary)

    # Test 6: Action summary
    action_summary = reflection.get_action_summary(last_n=3)
    print("✓ Test 6: get_action_summary()")
    print(action_summary)

    # Test 7: Reset
    reflection.reset()
    assert len(reflection.action_history) == 0, "History should be empty after reset"
    assert reflection.progress_metrics['successful_actions'] == 0, "Metrics should be reset"
    print("✓ Test 7: reset()")

    print("\n" + "="*70)
    print("✅ ALL AGENT REFLECTION TESTS PASSED")
    print("="*70)


def test_web_scraping_utils():
    """Test web scraping utilities (mock tests without browser)"""
    print("\n" + "="*70)
    print("TESTING: Web Scraping Utilities")
    print("="*70)

    from web_scraping_utils import deduplicate_products, filter_products

    # Test data
    products = [
        {"name": "Product A", "price": 29.99, "rating": 4.5, "reviews": 1500, "url": "url1"},
        {"name": "Product B", "price": 49.99, "rating": 4.0, "reviews": 800, "url": "url2"},
        {"name": "Product A Duplicate", "price": 29.99, "rating": 4.5, "reviews": 1500, "url": "url1"},  # Duplicate
        {"name": "Product C", "price": 199.99, "rating": 3.5, "reviews": 200, "url": "url3"},
        {"name": "Product D", "price": 89.99, "rating": 4.8, "reviews": 2000, "url": "url4"},
    ]

    # Test 1: Deduplicate
    unique = deduplicate_products(products, key='url')
    assert len(unique) == 4, f"Should have 4 unique products, got {len(unique)}"
    print("✓ Test 1: deduplicate_products()")
    print(f"  Original: {len(products)} products")
    print(f"  Unique: {len(unique)} products")

    # Test 2: Filter by price range
    filtered = filter_products(products, min_price=30, max_price=100)
    assert len(filtered) == 2, f"Should have 2 products in range, got {len(filtered)}"
    print("✓ Test 2: filter_products() by price")
    print(f"  Products $30-$100: {len(filtered)}")

    # Test 3: Filter by rating
    filtered = filter_products(products, min_rating=4.5)
    assert len(filtered) == 2, f"Should have 2 products with 4.5+ rating, got {len(filtered)}"
    print("✓ Test 3: filter_products() by rating")
    print(f"  Products ≥4.5 stars: {len(filtered)}")

    # Test 4: Filter by reviews
    filtered = filter_products(products, min_reviews=1000)
    assert len(filtered) == 3, f"Should have 3 products with 1000+ reviews, got {len(filtered)}"
    print("✓ Test 4: filter_products() by reviews")
    print(f"  Products ≥1000 reviews: {len(filtered)}")

    # Test 5: Combined filters
    filtered = filter_products(
        products,
        max_price=100,
        min_rating=4.0,
        min_reviews=1000
    )
    assert len(filtered) == 2, f"Should have 2 products matching all criteria, got {len(filtered)}"
    print("✓ Test 5: filter_products() combined")
    print(f"  Products matching all criteria: {len(filtered)}")
    for p in filtered:
        print(f"    - {p['name']}: ${p['price']}, {p['rating']}⭐, {p['reviews']} reviews")

    print("\n" + "="*70)
    print("✅ ALL WEB SCRAPING UTILS TESTS PASSED")
    print("="*70)


def main():
    """Run all tests"""
    print("\n🧪 TESTING NEW LEARNING FEATURES\n")

    try:
        test_learning_database()
        test_agent_reflection()
        test_web_scraping_utils()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70)
        print("\n💡 New features ready:")
        print("   ✓ LearningDatabase - Cross-session learning")
        print("   ✓ AgentReflection - Stuck detection & adaptation")
        print("   ✓ Web Scraping Utils - Smart extraction & filtering")
        print("\n📝 Database created: test_agent_learning.db")
        print("   (Can be deleted after testing)\n")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
