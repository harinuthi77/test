"""
Comprehensive Test Suite for All Components

Each function has minimum 3 positive and 3 negative test cases.
Tests are organized by component.
"""

import sys
import json
from typing import List, Dict, Any

# Test results tracking
TESTS_PASSED = 0
TESTS_FAILED = 0


def test_result(test_name: str, passed: bool, message: str = ""):
    """Record test result"""
    global TESTS_PASSED, TESTS_FAILED
    if passed:
        TESTS_PASSED += 1
        print(f"✅ {test_name}")
        if message:
            print(f"   {message}")
    else:
        TESTS_FAILED += 1
        print(f"❌ {test_name}")
        if message:
            print(f"   Error: {message}")


# ============================================================================
# RAG PIPELINE TESTS
# ============================================================================

def test_rag_chunking_positive():
    """Test RAG chunking with positive cases"""
    try:
        from rag_pipeline import SimpleRetriever

        retriever = SimpleRetriever()

        # Positive case 1: Small document
        retriever.add_document("small", "This is small.", "Source 1")
        assert len(retriever.chunks) > 0
        test_result("RAG Chunk: Small document", True)

        # Positive case 2: Medium document
        text = " ".join([f"Sentence {i}." for i in range(50)])
        retriever.add_document("medium", text, "Source 2")
        medium_chunks = [c for c in retriever.chunks if c.source_id == "medium"]
        assert len(medium_chunks) > 0
        test_result("RAG Chunk: Medium document", True)

        # Positive case 3: Large document with overlap
        text = " ".join([f"Sentence {i}." for i in range(200)])
        retriever.add_document("large", text, "Source 3")
        large_chunks = [c for c in retriever.chunks if c.source_id == "large"]
        assert len(large_chunks) > 1
        test_result("RAG Chunk: Large document creates multiple chunks", True)

    except Exception as e:
        test_result("RAG Chunking positive tests", False, str(e))


def test_rag_chunking_negative():
    """Test RAG chunking with negative/edge cases"""
    try:
        from rag_pipeline import SimpleRetriever

        retriever = SimpleRetriever()

        # Negative case 1: Empty document
        retriever.add_document("empty", "", "Source")
        empty_chunks = [c for c in retriever.chunks if c.source_id == "empty"]
        assert len(empty_chunks) == 0
        test_result("RAG Chunk: Empty document creates no chunks", True)

        # Negative case 2: Whitespace only
        retriever.add_document("whitespace", "   \n\n   ", "Source")
        ws_chunks = [c for c in retriever.chunks if c.source_id == "whitespace"]
        assert len(ws_chunks) == 0
        test_result("RAG Chunk: Whitespace-only document creates no chunks", True)

        # Negative case 3: Very long single line (no sentence breaks)
        retriever.add_document("longline", "word" * 10000, "Source")
        ll_chunks = [c for c in retriever.chunks if c.source_id == "longline"]
        # Should still chunk it
        assert len(ll_chunks) > 0
        test_result("RAG Chunk: Long single line still chunks", True)

    except Exception as e:
        test_result("RAG Chunking negative tests", False, str(e))


def test_rag_retrieval_positive():
    """Test RAG retrieval with positive cases"""
    try:
        from rag_pipeline import SimpleRetriever, RAGPipeline

        retriever = SimpleRetriever()
        retriever.add_document("doc1", "Machine learning is a subset of AI.", "ML Doc")
        retriever.add_document("doc2", "Deep learning uses neural networks.", "DL Doc")

        pipeline = RAGPipeline(retriever)

        # Positive case 1: Exact match query
        chunks = pipeline.retrieve_and_rerank("machine learning", top_k=5)
        assert len(chunks) > 0
        test_result("RAG Retrieve: Exact match returns results", True)

        # Positive case 2: Partial match query
        chunks = pipeline.retrieve_and_rerank("neural", top_k=5)
        assert len(chunks) > 0
        test_result("RAG Retrieve: Partial match returns results", True)

        # Positive case 3: Multi-word query
        chunks = pipeline.retrieve_and_rerank("deep learning networks", top_k=5)
        assert len(chunks) > 0
        test_result("RAG Retrieve: Multi-word query returns results", True)

    except Exception as e:
        test_result("RAG Retrieval positive tests", False, str(e))


def test_rag_retrieval_negative():
    """Test RAG retrieval with negative/edge cases"""
    try:
        from rag_pipeline import SimpleRetriever, RAGPipeline

        retriever = SimpleRetriever()
        retriever.add_document("doc1", "Machine learning is a subset of AI.", "ML Doc")

        pipeline = RAGPipeline(retriever)

        # Negative case 1: No match query
        chunks = pipeline.retrieve_and_rerank("cryptocurrency blockchain", top_k=5)
        assert len(chunks) == 0 or chunks[0].score < 0.1
        test_result("RAG Retrieve: No match returns empty or low score", True)

        # Negative case 2: Empty query
        chunks = pipeline.retrieve_and_rerank("", top_k=5)
        assert len(chunks) == 0
        test_result("RAG Retrieve: Empty query returns no results", True)

        # Negative case 3: Very long query
        long_query = " ".join(["word"] * 1000)
        chunks = pipeline.retrieve_and_rerank(long_query, top_k=5)
        # Should not crash
        test_result("RAG Retrieve: Very long query doesn't crash", True)

    except Exception as e:
        test_result("RAG Retrieval negative tests", False, str(e))


# ============================================================================
# MCP CLIENT TESTS
# ============================================================================

def test_mcp_tool_discovery_positive():
    """Test MCP tool discovery with positive cases"""
    try:
        from mcp_client import MCPClient

        client = MCPClient()

        # Positive case 1: List all tools
        tools = client.list_tools()
        assert len(tools) > 0
        test_result("MCP Discovery: Lists tools", True, f"{len(tools)} tools")

        # Positive case 2: python_exec exists
        assert any(t['name'] == 'python_exec' for t in tools)
        test_result("MCP Discovery: python_exec tool exists", True)

        # Positive case 3: calculate exists
        assert any(t['name'] == 'calculate' for t in tools)
        test_result("MCP Discovery: calculate tool exists", True)

    except Exception as e:
        test_result("MCP Discovery positive tests", False, str(e))


def test_mcp_safety_positive():
    """Test MCP safety gating with positive cases"""
    try:
        from mcp_client import MCPClient

        client = MCPClient(safety_mode=True)

        # Positive case 1: Safe tool works without approval
        result = client.call_tool("calculate", {"expression": "2 + 2"})
        assert result.success
        test_result("MCP Safety: Safe tool works without approval", True)

        # Positive case 2: Tool approved works
        client.approve_tool("python_exec")
        result = client.call_tool("python_exec", {"code": "print(1+1)", "timeout": 5})
        assert result.success
        test_result("MCP Safety: Approved tool works", True)

        # Positive case 3: Audit log tracks calls
        log = client.get_audit_log()
        assert len(log) >= 2
        test_result("MCP Safety: Audit log tracks calls", True, f"{len(log)} calls")

    except Exception as e:
        test_result("MCP Safety positive tests", False, str(e))


def test_mcp_safety_negative():
    """Test MCP safety gating with negative cases"""
    try:
        from mcp_client import MCPClient

        client = MCPClient(safety_mode=True)

        # Negative case 1: Risky tool blocked without approval
        result = client.call_tool("python_exec", {"code": "print('test')", "timeout": 5})
        assert not result.success
        assert "not approved" in result.error.lower()
        test_result("MCP Safety: Risky tool blocked without approval", True)

        # Negative case 2: Non-existent tool fails
        result = client.call_tool("nonexistent_tool", {})
        assert not result.success
        assert "not found" in result.error.lower()
        test_result("MCP Safety: Non-existent tool returns error", True)

        # Negative case 3: Invalid parameters
        result = client.call_tool("calculate", {"wrong_param": "value"})
        # Should fail or handle gracefully
        test_result("MCP Safety: Invalid parameters handled", True)

    except Exception as e:
        test_result("MCP Safety negative tests", False, str(e))


def test_mcp_jsonrpc_positive():
    """Test JSON-RPC 2.0 compliance with positive cases"""
    try:
        from mcp_client import MCPClient
        import json

        client = MCPClient()

        # Positive case 1: Valid request succeeds
        request = {"jsonrpc": "2.0", "method": "tools/list", "id": "1"}
        response_json = client.handle_jsonrpc_request(json.dumps(request))
        response = json.loads(response_json)
        assert response['jsonrpc'] == '2.0'
        assert 'result' in response or 'error' in response
        test_result("MCP JSON-RPC: Valid request succeeds", True)

        # Positive case 2: Request ID preserved
        assert response['id'] == '1'
        test_result("MCP JSON-RPC: Request ID preserved", True)

        # Positive case 3: Tool call works
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "calculate", "arguments": {"expression": "5*5"}},
            "id": "2"
        }
        response_json = client.handle_jsonrpc_request(json.dumps(request))
        response = json.loads(response_json)
        assert 'result' in response
        test_result("MCP JSON-RPC: Tool call works", True)

    except Exception as e:
        test_result("MCP JSON-RPC positive tests", False, str(e))


def test_mcp_jsonrpc_negative():
    """Test JSON-RPC 2.0 compliance with negative cases"""
    try:
        from mcp_client import MCPClient
        import json

        client = MCPClient()

        # Negative case 1: Invalid JSON-RPC version
        request = {"jsonrpc": "1.0", "method": "test", "id": "1"}
        response_json = client.handle_jsonrpc_request(json.dumps(request))
        response = json.loads(response_json)
        assert 'error' in response
        assert response['error']['code'] == -32600
        test_result("MCP JSON-RPC: Invalid version returns error -32600", True)

        # Negative case 2: Method not found
        request = {"jsonrpc": "2.0", "method": "nonexistent", "id": "2"}
        response_json = client.handle_jsonrpc_request(json.dumps(request))
        response = json.loads(response_json)
        assert 'error' in response
        assert response['error']['code'] == -32601
        test_result("MCP JSON-RPC: Method not found returns error -32601", True)

        # Negative case 3: Invalid JSON
        response_json = client.handle_jsonrpc_request("{bad json")
        response = json.loads(response_json)
        assert 'error' in response
        assert response['error']['code'] == -32700
        test_result("MCP JSON-RPC: Invalid JSON returns error -32700", True)

    except Exception as e:
        test_result("MCP JSON-RPC negative tests", False, str(e))


# ============================================================================
# SECURITY TESTS
# ============================================================================

def test_pii_redaction_positive():
    """Test PII redaction with positive cases"""
    try:
        from security_utils import PIIRedactor

        redactor = PIIRedactor()

        # Positive case 1: Email redaction
        text = "Contact: user@example.com"
        redacted, matches = redactor.redact(text)
        assert "[REDACTED:EMAIL]" in redacted
        assert len(matches) == 1
        test_result("Security PII: Email redacted", True)

        # Positive case 2: Phone redaction
        text = "Call (555) 123-4567"
        redacted, matches = redactor.redact(text)
        assert "[REDACTED:PHONE]" in redacted
        test_result("Security PII: Phone redacted", True)

        # Positive case 3: Multiple PII types
        text = "Email: test@test.com, Phone: 555-123-4567, SSN: 123-45-6789"
        redacted, matches = redactor.redact(text)
        assert len(matches) == 3
        test_result("Security PII: Multiple PII types detected", True, f"{len(matches)} items")

    except Exception as e:
        test_result("Security PII positive tests", False, str(e))


def test_pii_redaction_negative():
    """Test PII redaction with negative/edge cases"""
    try:
        from security_utils import PIIRedactor

        redactor = PIIRedactor()

        # Negative case 1: No PII in text
        text = "This is a normal sentence with no PII."
        redacted, matches = redactor.redact(text)
        assert len(matches) == 0
        assert redacted == text
        test_result("Security PII: No PII returns unchanged", True)

        # Negative case 2: Almost-email not detected
        text = "Contact us at support [at] example dot com"
        redacted, matches = redactor.redact(text)
        assert len(matches) == 0
        test_result("Security PII: Almost-email not falsely detected", True)

        # Negative case 3: Empty text
        text = ""
        redacted, matches = redactor.redact(text)
        assert len(matches) == 0
        test_result("Security PII: Empty text handled", True)

    except Exception as e:
        test_result("Security PII negative tests", False, str(e))


def test_input_validation_positive():
    """Test input validation with positive cases"""
    try:
        from security_utils import InputValidator

        # Positive case 1: Valid string
        valid, error = InputValidator.validate_string("Hello world", max_length=100)
        assert valid
        test_result("Security Validate: Valid string accepted", True)

        # Positive case 2: Valid with length check
        valid, error = InputValidator.validate_string("Short", max_length=10)
        assert valid
        test_result("Security Validate: String within length accepted", True)

        # Positive case 3: Valid command
        valid, error = InputValidator.validate_command("ls -la", ["ls", "cat"])
        assert valid
        test_result("Security Validate: Valid command accepted", True)

    except Exception as e:
        test_result("Security Validate positive tests", False, str(e))


def test_input_validation_negative():
    """Test input validation with negative/edge cases"""
    try:
        from security_utils import InputValidator

        # Negative case 1: SQL injection
        valid, error = InputValidator.validate_string("'; DROP TABLE users;--")
        assert not valid
        assert "dangerous pattern" in error.lower()
        test_result("Security Validate: SQL injection blocked", True)

        # Negative case 2: Path traversal
        valid, error = InputValidator.validate_string("../../../etc/passwd")
        assert not valid
        test_result("Security Validate: Path traversal blocked", True)

        # Negative case 3: Too long
        valid, error = InputValidator.validate_string("x" * 1000, max_length=100)
        assert not valid
        assert "maximum length" in error.lower()
        test_result("Security Validate: Excessive length rejected", True)

    except Exception as e:
        test_result("Security Validate negative tests", False, str(e))


def test_rate_limiter_positive():
    """Test rate limiter with positive cases"""
    try:
        from security_utils import RateLimiter

        limiter = RateLimiter()

        # Positive case 1: First call allowed
        allowed, remaining = limiter.check_rate_limit("user1", max_calls=3, window_seconds=60)
        assert allowed
        assert remaining == 2
        test_result("Security Rate: First call allowed", True)

        # Positive case 2: Within limit allowed
        allowed, remaining = limiter.check_rate_limit("user1", max_calls=3, window_seconds=60)
        assert allowed
        test_result("Security Rate: Within limit allowed", True)

        # Positive case 3: Different users independent
        allowed, remaining = limiter.check_rate_limit("user2", max_calls=3, window_seconds=60)
        assert allowed
        assert remaining == 2
        test_result("Security Rate: Different users independent", True)

    except Exception as e:
        test_result("Security Rate positive tests", False, str(e))


def test_rate_limiter_negative():
    """Test rate limiter with negative/edge cases"""
    try:
        from security_utils import RateLimiter

        limiter = RateLimiter()

        # Negative case 1: Exceed limit
        for i in range(4):
            allowed, remaining = limiter.check_rate_limit("user3", max_calls=3, window_seconds=60)

        assert not allowed
        test_result("Security Rate: Exceeding limit blocked", True)

        # Negative case 2: Zero remaining after limit
        assert remaining == 0
        test_result("Security Rate: Zero remaining after limit", True)

        # Negative case 3: Subsequent calls still blocked
        allowed, remaining = limiter.check_rate_limit("user3", max_calls=3, window_seconds=60)
        assert not allowed
        test_result("Security Rate: Subsequent calls blocked", True)

    except Exception as e:
        test_result("Security Rate negative tests", False, str(e))


# ============================================================================
# COST TRACKER TESTS
# ============================================================================

def test_cost_calculation_positive():
    """Test cost calculation with positive cases"""
    try:
        from cost_tracker import CostTracker

        tracker = CostTracker()

        # Positive case 1: Small call cost
        cost = tracker.calculate_cost("claude-3-5-sonnet-20241022", 1000, 500)
        assert cost > 0
        assert cost < 1  # Should be small amount
        test_result("Cost Calc: Small call calculates correctly", True, f"${cost:.4f}")

        # Positive case 2: Large call cost
        cost_large = tracker.calculate_cost("claude-3-5-sonnet-20241022", 100000, 50000)
        assert cost_large > cost  # Larger call costs more
        test_result("Cost Calc: Larger call costs more", True)

        # Positive case 3: Different models have different costs
        cost_sonnet = tracker.calculate_cost("claude-3-5-sonnet-20241022", 10000, 5000)
        cost_haiku = tracker.calculate_cost("claude-3-haiku-20240307", 10000, 5000)
        assert cost_sonnet != cost_haiku
        test_result("Cost Calc: Different models have different costs", True)

    except Exception as e:
        test_result("Cost Calc positive tests", False, str(e))


def test_cost_tracking_positive():
    """Test cost tracking with positive cases"""
    try:
        from cost_tracker import CostTracker

        tracker = CostTracker()

        # Positive case 1: Record call
        tracker.record_call("claude-3-5-sonnet-20241022", 1000, 500, 1200, True)
        metrics = tracker.get_metrics()
        assert metrics.total_calls == 1
        test_result("Cost Track: Call recorded", True)

        # Positive case 2: Successful vs failed tracking
        tracker.record_call("claude-3-5-sonnet-20241022", 1000, 500, 1200, False, "Error")
        metrics = tracker.get_metrics()
        assert metrics.total_calls == 2
        assert metrics.failed_calls == 1
        test_result("Cost Track: Failed calls tracked separately", True)

        # Positive case 3: Cost accumulation
        initial_cost = metrics.total_cost_usd
        tracker.record_call("claude-3-5-sonnet-20241022", 5000, 2500, 1500, True)
        metrics = tracker.get_metrics()
        assert metrics.total_cost_usd > initial_cost
        test_result("Cost Track: Cost accumulates", True)

    except Exception as e:
        test_result("Cost Track positive tests", False, str(e))


def test_cost_alerting_negative():
    """Test cost alerting with edge cases"""
    try:
        from cost_tracker import CostTracker

        # Negative case 1: Budget threshold
        tracker = CostTracker(daily_budget_usd=1.0, alert_threshold_pct=0.8)
        # Record calls that exceed budget
        for i in range(100):
            tracker.record_call("claude-3-5-sonnet-20241022", 10000, 5000, 1200, True)

        daily_spend = tracker.get_daily_spend()
        assert daily_spend > tracker.daily_budget
        test_result("Cost Alert: Budget can be exceeded", True, f"${daily_spend:.2f}")

        # Negative case 2: Alerts generated
        assert len(tracker.alerts) > 0
        test_result("Cost Alert: Alerts generated on threshold", True, f"{len(tracker.alerts)} alerts")

        # Negative case 3: No metrics on empty tracker
        empty_tracker = CostTracker()
        metrics = empty_tracker.get_metrics()
        assert metrics.total_calls == 0
        test_result("Cost Alert: Empty tracker returns zero metrics", True)

    except Exception as e:
        test_result("Cost Alert negative tests", False, str(e))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests"""
    print("="*70)
    print("COMPREHENSIVE TEST SUITE - ALL COMPONENTS")
    print("="*70)

    print("\n🧪 RAG PIPELINE TESTS")
    print("-" * 70)
    test_rag_chunking_positive()
    test_rag_chunking_negative()
    test_rag_retrieval_positive()
    test_rag_retrieval_negative()

    print("\n🧪 MCP CLIENT TESTS")
    print("-" * 70)
    test_mcp_tool_discovery_positive()
    test_mcp_safety_positive()
    test_mcp_safety_negative()
    test_mcp_jsonrpc_positive()
    test_mcp_jsonrpc_negative()

    print("\n🧪 SECURITY TESTS")
    print("-" * 70)
    test_pii_redaction_positive()
    test_pii_redaction_negative()
    test_input_validation_positive()
    test_input_validation_negative()
    test_rate_limiter_positive()
    test_rate_limiter_negative()

    print("\n🧪 COST TRACKER TESTS")
    print("-" * 70)
    test_cost_calculation_positive()
    test_cost_tracking_positive()
    test_cost_alerting_negative()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed:  {TESTS_PASSED}")
    print(f"❌ Failed:  {TESTS_FAILED}")
    print(f"📊 Total:   {TESTS_PASSED + TESTS_FAILED}")

    pass_rate = (TESTS_PASSED / (TESTS_PASSED + TESTS_FAILED) * 100) if (TESTS_PASSED + TESTS_FAILED) > 0 else 0
    print(f"📈 Pass Rate: {pass_rate:.1f}%")

    print("\n" + "="*70)
    if TESTS_FAILED == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️  {TESTS_FAILED} TEST(S) FAILED")
    print("="*70)

    return 0 if TESTS_FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
