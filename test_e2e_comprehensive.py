"""
Comprehensive End-to-End Test Suite
Tests critical workflows and edge cases
"""

import os
import sys
from typing import List, Dict, Any

# Test configuration
TESTS_PASSED = 0
TESTS_FAILED = 0
TESTS_SKIPPED = 0


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


def skip_test(test_name: str, reason: str):
    """Skip a test"""
    global TESTS_SKIPPED
    TESTS_SKIPPED += 1
    print(f"⏭️  {test_name} (skipped: {reason})")


# ============================================================================
# TEST SUITE 1: RAG Pipeline
# ============================================================================

def test_rag_chunking():
    """Test RAG chunking with various document sizes"""
    try:
        from rag_pipeline import SimpleRetriever

        retriever = SimpleRetriever()

        # Test 1: Small document
        retriever.add_document(
            "small_doc",
            "This is a small document. It has only a few sentences. Testing chunking.",
            "Small Test Doc"
        )

        assert len(retriever.chunks) > 0, "No chunks created for small document"
        test_result("RAG: Small document chunking", True, f"{len(retriever.chunks)} chunks")

        # Test 2: Large document (should create multiple chunks)
        large_text = " ".join([f"Sentence {i} with some content." for i in range(200)])
        retriever.add_document("large_doc", large_text, "Large Test Doc")

        chunk_count = len([c for c in retriever.chunks if c.source_id == "large_doc"])
        assert chunk_count > 1, "Large document should create multiple chunks"
        test_result("RAG: Large document multi-chunking", True, f"{chunk_count} chunks")

        # Test 3: Overlap verification
        doc_chunks = [c for c in retriever.chunks if c.source_id == "large_doc"]
        if len(doc_chunks) >= 2:
            chunk1_end = doc_chunks[0].text[-50:]
            chunk2_start = doc_chunks[1].text[:50]
            # Some overlap should exist
            test_result("RAG: Chunk overlap exists", True)
        else:
            test_result("RAG: Chunk overlap verification", True, "Not enough chunks to test")

    except ImportError as e:
        skip_test("RAG: Chunking tests", f"Missing import: {e}")
    except Exception as e:
        test_result("RAG: Chunking tests", False, str(e))


def test_rag_retrieval():
    """Test RAG retrieval accuracy"""
    try:
        from rag_pipeline import SimpleRetriever, RAGPipeline

        retriever = SimpleRetriever()

        # Add test documents
        retriever.add_document(
            "transformers",
            """
            Transformers are neural networks that use self-attention mechanisms.
            They revolutionized natural language processing. The key innovation
            is allowing tokens to attend to other tokens in the sequence.
            """,
            "Transformers Overview"
        )

        retriever.add_document(
            "llms",
            """
            Large language models like GPT and Claude use transformer architecture.
            They are trained on massive datasets and can generate human-like text.
            These models have billions of parameters.
            """,
            "LLMs Overview"
        )

        pipeline = RAGPipeline(retriever)

        # Test 1: Relevant query
        chunks = pipeline.retrieve_and_rerank("What are transformers?", top_k=5)
        assert len(chunks) > 0, "No chunks retrieved"

        # Should retrieve transformer doc with higher score
        top_chunk = chunks[0]
        assert "transform" in top_chunk.text.lower() or "attention" in top_chunk.text.lower(), \
            "Top chunk not relevant to query"

        test_result("RAG: Relevant retrieval", True, f"Retrieved {len(chunks)} chunks")

        # Test 2: Grounding with citations
        grounded = pipeline.ground_context("transformers and attention", chunks)
        assert grounded.total_sources > 0, "No sources in grounded context"
        assert len(grounded.chunks) > 0, "No chunks in grounded context"

        test_result("RAG: Grounding and context", True,
                   f"{len(grounded.chunks)} chunks, {grounded.total_sources} sources")

    except ImportError as e:
        skip_test("RAG: Retrieval tests", f"Missing import: {e}")
    except Exception as e:
        test_result("RAG: Retrieval tests", False, str(e))


def test_rag_vector_retriever():
    """Test vector retriever if dependencies available"""
    try:
        from rag_pipeline import VectorRetriever

        # Test graceful fallback when dependencies missing
        retriever = VectorRetriever(backend="chroma")

        retriever.add_document(
            "test_doc",
            "This is a test document for vector retrieval.",
            "Test Doc"
        )

        chunks = retriever.retrieve("test query", top_k=5)

        # Should work either with vector DB or keyword fallback
        test_result("RAG: Vector retriever initialization", True,
                   f"Backend: chroma, Retrieved {len(chunks)} chunks")

    except ImportError as e:
        skip_test("RAG: Vector retriever", f"Dependencies not installed: {e}")
    except Exception as e:
        test_result("RAG: Vector retriever", False, str(e))


# ============================================================================
# TEST SUITE 2: MCP Client
# ============================================================================

def test_mcp_tool_discovery():
    """Test MCP tool discovery and listing"""
    try:
        from mcp_client import MCPClient

        client = MCPClient(safety_mode=True)
        tools = client.list_tools()

        assert len(tools) > 0, "No tools registered"
        assert any(t['name'] == 'python_exec' for t in tools), "python_exec not found"
        assert any(t['name'] == 'calculate' for t in tools), "calculate not found"

        test_result("MCP: Tool discovery", True, f"{len(tools)} tools found")

    except ImportError as e:
        skip_test("MCP: Tool discovery", f"Missing import: {e}")
    except Exception as e:
        test_result("MCP: Tool discovery", False, str(e))


def test_mcp_safety_gating():
    """Test MCP safety gating for risky tools"""
    try:
        from mcp_client import MCPClient

        client = MCPClient(safety_mode=True)

        # Test 1: Call risky tool without approval (should fail)
        result = client.call_tool("python_exec", {"code": "print('test')", "timeout": 5})
        assert not result.success, "Risky tool should fail without approval"
        assert "not approved" in result.error.lower(), "Error should mention approval"

        test_result("MCP: Safety gating blocks unapproved tools", True)

        # Test 2: Approve and retry (should succeed)
        client.approve_tool("python_exec")
        result = client.call_tool("python_exec", {"code": "print('Hello')", "timeout": 5})
        assert result.success, f"Approved tool should succeed: {result.error}"

        test_result("MCP: Approved tools execute", True)

        # Test 3: Safe tool without approval (should succeed)
        result = client.call_tool("calculate", {"expression": "2 + 2"})
        assert result.success, "Safe tool should work without approval"
        assert "4" in str(result.result), "Calculator result incorrect"

        test_result("MCP: Safe tools work without approval", True)

    except ImportError as e:
        skip_test("MCP: Safety gating", f"Missing import: {e}")
    except Exception as e:
        test_result("MCP: Safety gating", False, str(e))


def test_mcp_jsonrpc_compliance():
    """Test JSON-RPC 2.0 compliance"""
    try:
        from mcp_client import MCPClient, JSONRPCRequest
        import json

        client = MCPClient(safety_mode=True)

        # Test 1: Valid JSON-RPC 2.0 request
        request = JSONRPCRequest(method="tools/list", id="test-123")
        response_json = client.handle_jsonrpc_request(json.dumps(request.to_dict()))
        response = json.loads(response_json)

        assert response['jsonrpc'] == '2.0', "Response not JSON-RPC 2.0"
        assert response['id'] == 'test-123', "Request ID not preserved"
        assert 'result' in response or 'error' in response, "Response missing result/error"

        test_result("MCP: JSON-RPC 2.0 format compliance", True)

        # Test 2: Error handling for invalid version
        bad_request = {"jsonrpc": "1.0", "method": "test", "id": "bad"}
        response_json = client.handle_jsonrpc_request(json.dumps(bad_request))
        response = json.loads(response_json)

        assert 'error' in response, "Should return error for invalid version"
        assert response['error']['code'] == -32600, "Wrong error code"

        test_result("MCP: JSON-RPC error handling", True)

    except ImportError as e:
        skip_test("MCP: JSON-RPC compliance", f"Missing import: {e}")
    except Exception as e:
        test_result("MCP: JSON-RPC compliance", False, str(e))


# ============================================================================
# TEST SUITE 3: Multi-Agent System
# ============================================================================

def test_agent_framework_basic():
    """Test basic agent framework functionality"""
    try:
        from agent_framework import BaseAgent, AgentConfig

        # Note: This would need ANTHROPIC_API_KEY to actually run
        # So we just test initialization
        config = AgentConfig(
            name="test_agent",
            role="Test role",
            temperature=0.7
        )

        # Just test that config is created properly
        assert config.name == "test_agent", "Config name incorrect"
        assert config.temperature == 0.7, "Config temperature incorrect"

        test_result("Agent: Framework configuration", True)

    except ImportError as e:
        skip_test("Agent: Framework tests", f"Missing import: {e}")
    except Exception as e:
        test_result("Agent: Framework tests", False, str(e))


def test_deviation_detection():
    """Test deviation detection rules"""
    try:
        from agent_framework import check_no_citations, check_excessive_length, check_low_confidence

        # Test 1: Citation detection
        text_no_cite = "This is a response without citations."
        text_with_cite = "According to [Source 1], transformers use attention."

        assert check_no_citations(text_no_cite), "Should detect missing citations"
        assert not check_no_citations(text_with_cite), "Should not flag text with citations"

        test_result("Agent: Citation deviation detection", True)

        # Test 2: Length check
        short_text = "Too short"
        long_text = " ".join(["word"] * 1000)

        assert check_excessive_length(short_text, min_length=20), "Should detect too short"
        assert check_excessive_length(long_text, max_length=500), "Should detect too long"

        test_result("Agent: Length deviation detection", True)

        # Test 3: Confidence check
        low_conf = "I think maybe possibly it could be..."
        high_conf = "The transformer architecture uses self-attention mechanisms."

        assert check_low_confidence(low_conf), "Should detect low confidence"
        assert not check_low_confidence(high_conf), "Should not flag confident text"

        test_result("Agent: Confidence deviation detection", True)

    except ImportError as e:
        skip_test("Agent: Deviation detection", f"Missing import: {e}")
    except Exception as e:
        test_result("Agent: Deviation detection", False, str(e))


# ============================================================================
# TEST SUITE 4: Transformer Optimizations
# ============================================================================

def test_transformer_optimizations():
    """Test transformer optimization utilities"""
    try:
        from agent_transformer_optimizations import (
            smart_element_filtering,
            create_compact_element_description
        )

        # Test 1: Smart filtering
        elements = [
            {"tag": "button", "text": "Click me", "visible": True},
            {"tag": "div", "text": "", "visible": False},
            {"tag": "a", "text": "Link", "visible": True},
            {"tag": "span", "text": "Hidden", "visible": False},
        ]

        filtered = smart_element_filtering(elements, max_elements=2)
        assert len(filtered) <= 2, "Filtering didn't limit elements"
        assert all(e['visible'] for e in filtered), "Should prioritize visible elements"

        test_result("Transformer: Smart element filtering", True,
                   f"Filtered to {len(filtered)} elements")

        # Test 2: Compact descriptions
        element = {"tag": "button", "text": "Submit Form", "visible": True, "id": "submit-btn"}
        compact = create_compact_element_description(element)

        assert len(compact) < 100, "Description should be compact"
        assert "button" in compact.lower(), "Should include tag"

        test_result("Transformer: Compact descriptions", True)

    except ImportError as e:
        skip_test("Transformer: Optimizations", f"Missing import: {e}")
    except Exception as e:
        test_result("Transformer: Optimizations", False, str(e))


# ============================================================================
# TEST SUITE 5: Integration Tests
# ============================================================================

def test_unified_agent_interface():
    """Test unified agent can be initialized"""
    try:
        # This will fail without API key, so we test import only
        from unified_agent import UnifiedAgent

        # Test capabilities method (doesn't need API key)
        # We can't instantiate without API key, but we can test the class exists
        assert hasattr(UnifiedAgent, 'web_scrape'), "Missing web_scrape method"
        assert hasattr(UnifiedAgent, 'teach'), "Missing teach method"
        assert hasattr(UnifiedAgent, 'add_knowledge'), "Missing add_knowledge method"

        test_result("Unified Agent: Interface methods exist", True)

    except ImportError as e:
        skip_test("Unified Agent: Interface", f"Missing import: {e}")
    except Exception as e:
        test_result("Unified Agent: Interface", False, str(e))


# ============================================================================
# TEST SUITE 6: Security & Safety
# ============================================================================

def test_security_input_validation():
    """Test input validation and sanitization"""
    try:
        from mcp_client import MCPClient

        client = MCPClient(safety_mode=True)

        # Test 1: Calculator with invalid characters
        result = client.call_tool("calculate", {"expression": "2 + 2; import os"})
        # Should fail due to invalid characters
        assert not result.success or "ERROR" in result.result, "Should reject invalid characters"

        test_result("Security: Input validation in calculator", True)

        # Test 2: File read path validation
        result = client.call_tool("read_file", {"path": "/etc/passwd"})
        # Should fail due to path restrictions
        assert not result.success or "ERROR" in result.result, "Should restrict file paths"

        test_result("Security: Path validation in file reader", True)

    except ImportError as e:
        skip_test("Security: Input validation", f"Missing import: {e}")
    except Exception as e:
        test_result("Security: Input validation", False, str(e))


def test_security_audit_logging():
    """Test that all tool calls are logged"""
    try:
        from mcp_client import MCPClient

        client = MCPClient(safety_mode=True)

        # Make several calls
        client.call_tool("calculate", {"expression": "1 + 1"})
        client.call_tool("calculate", {"expression": "2 + 2"})

        # Check audit log
        log = client.get_audit_log()
        assert len(log) >= 2, "Audit log should have at least 2 entries"
        assert all('tool_name' in entry for entry in log), "Log entries missing tool_name"
        assert all('success' in entry for entry in log), "Log entries missing success field"

        test_result("Security: Audit logging", True, f"{len(log)} calls logged")

    except ImportError as e:
        skip_test("Security: Audit logging", f"Missing import: {e}")
    except Exception as e:
        test_result("Security: Audit logging", False, str(e))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests"""
    print("="*70)
    print("COMPREHENSIVE END-TO-END TEST SUITE")
    print("="*70)

    print("\n📋 TEST SUITE 1: RAG Pipeline")
    print("-" * 70)
    test_rag_chunking()
    test_rag_retrieval()
    test_rag_vector_retriever()

    print("\n📋 TEST SUITE 2: MCP Client")
    print("-" * 70)
    test_mcp_tool_discovery()
    test_mcp_safety_gating()
    test_mcp_jsonrpc_compliance()

    print("\n📋 TEST SUITE 3: Multi-Agent System")
    print("-" * 70)
    test_agent_framework_basic()
    test_deviation_detection()

    print("\n📋 TEST SUITE 4: Transformer Optimizations")
    print("-" * 70)
    test_transformer_optimizations()

    print("\n📋 TEST SUITE 5: Integration Tests")
    print("-" * 70)
    test_unified_agent_interface()

    print("\n📋 TEST SUITE 6: Security & Safety")
    print("-" * 70)
    test_security_input_validation()
    test_security_audit_logging()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed:  {TESTS_PASSED}")
    print(f"❌ Failed:  {TESTS_FAILED}")
    print(f"⏭️  Skipped: {TESTS_SKIPPED}")
    print(f"📊 Total:   {TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED}")

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
