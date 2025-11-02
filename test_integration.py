"""
Integration Test - Verify all components work together
Tests both web scraping and tutoring modes
"""

def test_imports():
    """Test all modules can be imported"""
    print("Testing imports...")

    try:
        from agent_framework import BaseAgent, MultiAgentOrchestrator
        print("✓ agent_framework")

        from rag_pipeline import RAGPipeline, SimpleRetriever
        print("✓ rag_pipeline")

        from mcp_client import MCPClient
        print("✓ mcp_client")

        from tutoring_orchestrator import TutoringOrchestrator
        print("✓ tutoring_orchestrator")

        from agent_transformer_optimizations import smart_element_filtering
        print("✓ agent_transformer_optimizations")

        from unified_agent import UnifiedAgent
        print("✓ unified_agent")

        return True

    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_rag_pipeline():
    """Test RAG pipeline standalone"""
    print("\nTesting RAG pipeline...")

    from rag_pipeline import RAGPipeline, SimpleRetriever

    retriever = SimpleRetriever()
    retriever.add_document(
        "test_doc",
        "This is a test document about transformers and neural networks.",
        "Test Source"
    )

    pipeline = RAGPipeline(retriever)
    chunks = pipeline.retrieve_and_rerank("transformers", top_k=5)

    if chunks:
        print(f"✓ Retrieved {len(chunks)} chunks")
        return True
    else:
        print("✗ No chunks retrieved")
        return False


def test_mcp_client():
    """Test MCP client standalone"""
    print("\nTesting MCP client...")

    from mcp_client import MCPClient

    client = MCPClient(safety_mode=True)

    # Test safe tool
    result = client.call_tool("calculate", {"expression": "2 + 2"})

    if result.success and "4" in str(result.result):
        print("✓ Calculator tool works")
        return True
    else:
        print(f"✗ Calculator failed: {result.error}")
        return False


def test_agent_framework():
    """Test agent framework (without API key)"""
    print("\nTesting agent framework...")

    from agent_framework import AgentConfig, AgentRole, MultiAgentOrchestrator
    import anthropic

    # Can't make API calls without key, but can test structure
    try:
        # Mock client
        class MockClient:
            pass

        client = MockClient()
        orch = MultiAgentOrchestrator(client)

        # Test spawning (won't call API)
        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            system_prompt="Test"
        )

        print("✓ Agent framework structure OK")
        return True

    except Exception as e:
        print(f"✗ Agent framework error: {e}")
        return False


def test_unified_agent_structure():
    """Test unified agent can be instantiated"""
    print("\nTesting unified agent structure...")

    import os
    os.environ['ANTHROPIC_API_KEY'] = 'test-key-placeholder'

    try:
        from unified_agent import UnifiedAgent

        agent = UnifiedAgent()

        # Test capability listing
        caps = agent.get_capabilities()

        if "modes" in caps and "web_scraping" in caps["modes"]:
            print("✓ Unified agent structure OK")
            print(f"  Modes: {list(caps['modes'].keys())}")
            return True
        else:
            print("✗ Missing capabilities")
            return False

    except Exception as e:
        print(f"✗ Unified agent error: {e}")
        return False


def test_transformer_optimizations():
    """Test transformer optimization functions"""
    print("\nTesting transformer optimizations...")

    try:
        from agent_transformer_optimizations import (
            smart_element_filtering,
            create_compact_element_description,
            count_tokens_estimate
        )

        # Test element filtering
        elements = [
            {"id": 1, "tag": "button", "text": "Click me", "visible": True},
            {"id": 2, "tag": "div", "text": "", "visible": False},
            {"id": 3, "tag": "a", "text": "Link", "visible": True}
        ]

        filtered = smart_element_filtering(elements, max_elements=2)

        if len(filtered) == 2:
            print(f"✓ Smart filtering works ({len(filtered)} elements)")
            return True
        else:
            print(f"✗ Filtering failed: got {len(filtered)} elements")
            return False

    except Exception as e:
        print(f"✗ Optimization error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("INTEGRATION TEST SUITE")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("RAG Pipeline", test_rag_pipeline),
        ("MCP Client", test_mcp_client),
        ("Agent Framework", test_agent_framework),
        ("Transformer Opts", test_transformer_optimizations),
        ("Unified Agent", test_unified_agent_structure)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System ready to use!")
        return 0
    else:
        print("\n⚠️  Some tests failed - check errors above")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
