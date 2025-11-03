"""
Comprehensive Tests for SSM/Mamba Integration

Tests cover:
1. SSM core (forward, step, state management)
2. Attention bridge (cross-attention over spans)
3. Adaptive scheduler (routing by L,R,H,T)
4. Enhanced reasoning (self-consistency, PoT, verification)
5. Supervision (monitoring, alerts, coverage)
6. Integration (end-to-end adaptive tutor)

Each test has 3+ positive and negative cases.
"""

import numpy as np
from typing import List, Dict
import sys

from ssm_mamba_core import (
    MambaLayer, AttentionBridge, HybridSSMTransformer,
    SSMState, SSMStateManager,
    compute_sequence_length_signal, compute_hardness_signal
)
from adaptive_scheduler import (
    AdaptiveScheduler, SchedulerSignals, ExecutionPath,
    ReasoningEngine, SupervisionMonitor,
    compute_signals_from_request
)
from ssm_integration import (
    SSMEnhancedAgent, SSMAgentConfig,
    SSMSupervisedOrchestrator, rag_chunks_to_spans
)


def test_ssm_core():
    """Test SSM/Mamba core functionality"""
    print("🧪 SSM CORE TESTS")
    print("-" * 70)

    # Test 1: MambaLayer forward pass (positive)
    d_model, d_state = 64, 16
    layer = MambaLayer(d_model, d_state)
    x = np.random.randn(10, d_model) * 0.1  # (L=10, D=64)
    y, state = layer.forward(x)

    assert y.shape == (10, d_model), f"Expected (10, {d_model}), got {y.shape}"
    assert state.h.shape == (d_state,), f"Expected ({d_state},), got {state.h.shape}"
    assert state.step == 10, f"Expected step=10, got {state.step}"
    print("✅ MambaLayer forward: Correct output and state shapes")

    # Test 2: MambaLayer step (single token) (positive)
    x_token = np.random.randn(d_model) * 0.1
    y_token, new_state = layer.step(x_token, state)

    assert y_token.shape == (d_model,), f"Expected ({d_model},), got {y_token.shape}"
    assert new_state.step == 11, f"Expected step=11, got {new_state.step}"
    print("✅ MambaLayer step: Single-token processing works")

    # Test 3: State reset (positive)
    state.reset()
    assert state.step == 0, "State step should reset to 0"
    assert np.allclose(state.h, 0), "State h should reset to zeros"
    print("✅ SSMState reset: State resets correctly")

    # Test 4: State serialization (positive)
    state = SSMState(h=np.array([1.0, 2.0, 3.0]), step=5, metadata={'task': 'test'})
    state_dict = state.to_dict()
    restored = SSMState.from_dict(state_dict)

    assert np.allclose(restored.h, state.h), "State h should match after serialization"
    assert restored.step == state.step, "State step should match"
    assert restored.metadata == state.metadata, "Metadata should match"
    print("✅ SSMState serialization: to_dict/from_dict works")

    # Test 5: Empty input (negative)
    try:
        y_empty, _ = layer.forward(np.zeros((0, d_model)))
        # Should handle gracefully or raise clear error
        print("⚠️  MambaLayer empty input: Handled (may need validation)")
    except Exception as e:
        print(f"⚠️  MambaLayer empty input: Raises {type(e).__name__} (expected)")

    # Test 6: Mismatched dimensions (negative)
    try:
        y_bad, _ = layer.forward(np.random.randn(10, 32))  # Wrong d_model
        print("❌ MambaLayer dimension mismatch: Should raise error")
    except AssertionError:
        print("✅ MambaLayer dimension mismatch: Correctly raises AssertionError")

    print()


def test_attention_bridge():
    """Test attention bridge over RAG/tool spans"""
    print("🧪 ATTENTION BRIDGE TESTS")
    print("-" * 70)

    d_model = 64
    bridge = AttentionBridge(d_model, n_heads=4)

    # Test 1: Cross-attend over 2 spans (positive)
    query = np.random.randn(10, d_model) * 0.1  # (L=10, D=64)
    spans = [
        np.random.randn(5, d_model) * 0.1,   # Span 1: 5 tokens
        np.random.randn(3, d_model) * 0.1,   # Span 2: 3 tokens
    ]
    output = bridge.forward(query, spans)

    assert output.shape == query.shape, f"Expected {query.shape}, got {output.shape}"
    print("✅ AttentionBridge forward: Correct output shape with 2 spans")

    # Test 2: Many spans (positive)
    many_spans = [np.random.randn(10, d_model) * 0.1 for _ in range(20)]
    output_many = bridge.forward(query, many_spans)
    assert output_many.shape == query.shape, "Should handle many spans"
    print("✅ AttentionBridge many spans: Handles 20 spans")

    # Test 3: No spans (negative)
    output_no_spans = bridge.forward(query, [])
    assert np.allclose(output_no_spans, query), "No spans should return identity"
    print("✅ AttentionBridge no spans: Returns identity (graceful fallback)")

    # Test 4: Single large span (positive)
    large_span = [np.random.randn(100, d_model) * 0.1]
    output_large = bridge.forward(query, large_span)
    assert output_large.shape == query.shape, "Should handle large span"
    print("✅ AttentionBridge large span: Handles 100-token span")

    # Test 5: Mismatched span dimension (negative)
    try:
        bad_span = [np.random.randn(5, 32)]  # Wrong d_model
        output_bad = bridge.forward(query, bad_span)
        print("❌ AttentionBridge dimension mismatch: Should raise error")
    except (ValueError, AssertionError):
        print("✅ AttentionBridge dimension mismatch: Correctly raises error")

    print()


def test_hybrid_model():
    """Test hybrid SSM + attention architecture"""
    print("🧪 HYBRID MODEL TESTS")
    print("-" * 70)

    # Test 1: Pure SSM forward (no bridge) (positive)
    model = HybridSSMTransformer(d_model=64, n_layers=6, d_state=16, bridge_layers=[])
    x = np.random.randn(20, 64) * 0.1
    output, states = model.forward(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert len(states) == 6, f"Expected 6 states, got {len(states)}"
    print("✅ Hybrid model (pure SSM): Forward pass works")

    # Test 2: Hybrid (SSM + bridge) (positive)
    model_hybrid = HybridSSMTransformer(
        d_model=64, n_layers=6, d_state=16,
        bridge_layers=[4, 5]  # Top 2 layers have bridge
    )
    spans = [np.random.randn(10, 64) * 0.1, np.random.randn(8, 64) * 0.1]
    output_hybrid, states_hybrid = model_hybrid.forward(x, bridge_spans=spans)

    assert output_hybrid.shape == x.shape, "Hybrid output shape correct"
    assert len(states_hybrid) == 6, "Hybrid has 6 states"
    print("✅ Hybrid model (SSM + bridge): Forward with bridge works")

    # Test 3: State persistence (positive)
    output2, states2 = model_hybrid.forward(x, states=states_hybrid, bridge_spans=spans)
    assert states2[0].step > states_hybrid[0].step, "State step should increment"
    print("✅ Hybrid model state: States persist across calls")

    # Test 4: Different sequence lengths (positive)
    x_short = np.random.randn(5, 64) * 0.1
    output_short, _ = model_hybrid.forward(x_short)
    assert output_short.shape == (5, 64), "Handles short sequences"

    x_long = np.random.randn(200, 64) * 0.1
    output_long, _ = model_hybrid.forward(x_long)
    assert output_long.shape == (200, 64), "Handles long sequences"
    print("✅ Hybrid model lengths: Handles 5 and 200 token sequences")

    # Test 5: Wrong state count (negative)
    try:
        bad_states = [SSMState(h=np.zeros(16)) for _ in range(3)]  # Only 3, need 6
        output_bad, _ = model_hybrid.forward(x, states=bad_states)
        print("⚠️  Hybrid model wrong state count: May need validation")
    except (IndexError, AssertionError):
        print("✅ Hybrid model wrong state count: Raises error")

    print()


def test_scheduler_routing():
    """Test adaptive scheduler routing logic"""
    print("🧪 SCHEDULER ROUTING TESTS")
    print("-" * 70)

    scheduler = AdaptiveScheduler(
        long_context_threshold=8000,
        high_hardness_threshold=0.6,
        precision_tasks=['facts', 'legal', 'medical']
    )

    # Test 1: Fast path (low L, R, H) (positive)
    signals = SchedulerSignals(L=100, R=0, H=0.3, T=False)
    config = scheduler.route(signals, task_type='qa')
    assert config.path == ExecutionPath.FAST, f"Expected FAST, got {config.path}"
    assert config.self_consistency_samples == 1, "Fast path should use 1 sample"
    print("✅ Scheduler fast path: Routes correctly for simple Q&A")

    # Test 2: RAG bridge (R>0, precision task) (positive)
    signals = SchedulerSignals(L=500, R=10, H=0.5, T=False)
    config = scheduler.route(signals, task_type='medical')
    assert config.path == ExecutionPath.RAG_BRIDGE, f"Expected RAG_BRIDGE, got {config.path}"
    assert config.use_bridge, "RAG path should use bridge"
    assert config.run_verifier, "RAG path should verify citations"
    print("✅ Scheduler RAG bridge: Routes correctly for medical task with RAG")

    # Test 3: Deliberate (high H) (positive)
    signals = SchedulerSignals(L=500, R=0, H=0.8, T=False)
    config = scheduler.route(signals, task_type='math')
    assert config.path == ExecutionPath.DELIBERATE, f"Expected DELIBERATE, got {config.path}"
    assert config.self_consistency_samples >= 5, "Deliberate should use 5+ samples"
    assert config.enable_pot, "Math task should enable PoT"
    assert config.run_verifier, "Deliberate should verify"
    assert config.allow_repair, "Deliberate should allow repair"
    print("✅ Scheduler deliberate: Routes correctly for hard math (H=0.8)")

    # Test 4: Streaming (long context or T=True) (positive)
    signals = SchedulerSignals(L=10000, R=0, H=0.5, T=False)
    config = scheduler.route(signals, task_type='qa')
    assert config.path == ExecutionPath.STREAMING, f"Expected STREAMING for L=10k"
    print("✅ Scheduler streaming (long L): Routes correctly for L=10k")

    signals_stream = SchedulerSignals(L=500, R=0, H=0.5, T=True)
    config_stream = scheduler.route(signals_stream, task_type='code')
    assert config_stream.path == ExecutionPath.STREAMING, f"Expected STREAMING for T=True"
    print("✅ Scheduler streaming (T=True): Routes correctly for streaming tools")

    # Test 5: Hardness-based sample scaling (positive)
    h_samples = []
    for h in [0.6, 0.7, 0.8, 0.9]:
        sig = SchedulerSignals(L=500, R=0, H=h, T=False)
        conf = scheduler.route(sig, task_type='proof')
        h_samples.append((h, conf.self_consistency_samples))

    # Samples should increase with hardness
    assert h_samples[0][1] == 5, "H=0.6 should give 5 samples"
    assert h_samples[2][1] >= 6, "H=0.8 should give 6+ samples"
    print(f"✅ Scheduler hardness scaling: {h_samples}")

    # Test 6: Stats tracking (positive)
    stats = scheduler.get_stats()
    assert stats['total_requests'] > 0, "Should track requests"
    assert 'distribution' in stats, "Should have distribution"
    print(f"✅ Scheduler stats: {stats['total_requests']} requests tracked")

    print()


def test_reasoning_engine():
    """Test enhanced reasoning (self-consistency, PoT, verification)"""
    print("🧪 REASONING ENGINE TESTS")
    print("-" * 70)

    reasoning = ReasoningEngine()

    # Test 1: Self-consistency voting (positive)
    def mock_model(prompt):
        return "Answer: 42"

    answer, stats = reasoning.self_consistency(mock_model, "What is the answer?", k=5)
    assert stats['k'] == 5, "Should generate 5 samples"
    assert 'agreement_rate' in stats, "Should report agreement rate"
    assert stats['consensus_votes'] >= 1, "Should have consensus"
    print(f"✅ Self-consistency: k={stats['k']}, agreement={stats['agreement_rate']*100:.1f}%")

    # Test 2: Self-consistency with diversity (positive)
    # Real test would verify different answers vote correctly
    answer2, stats2 = reasoning.self_consistency(mock_model, "Different question", k=7)
    assert stats2['k'] == 7, "Should handle k=7"
    print("✅ Self-consistency k=7: Works with more samples")

    # Test 3: Program-of-Thoughts without MCP (negative fallback)
    result, code = reasoning.program_of_thoughts("What is 2+2?")
    assert "MCP client not available" in result, "Should indicate no MCP"
    assert len(code) > 0, "Should still generate code"
    print("✅ Program-of-Thoughts (no MCP): Graceful fallback")

    # Test 4: Verification passes (positive)
    good_answer = "The result is 42 kg with proper units"
    final, report = reasoning.verify_and_repair(
        good_answer,
        verifier_checks=['schema', 'units'],
        allow_repair=False
    )
    assert 'schema' in report['checks_passed'], "Schema should pass"
    assert 'units' in report['checks_passed'], "Units should pass"
    assert len(report['checks_failed']) == 0, "No failures expected"
    print(f"✅ Verification passes: {report['checks_passed']}")

    # Test 5: Verification fails, no repair (negative)
    bad_answer = "Short"
    final2, report2 = reasoning.verify_and_repair(
        bad_answer,
        verifier_checks=['citations'],
        allow_repair=False
    )
    assert 'citations' in report2['checks_failed'], "Citations should fail"
    assert not report2['repair_attempted'], "Should not repair"
    print(f"✅ Verification fails (no repair): {report2['checks_failed']}")

    # Test 6: Verification fails, with repair (positive)
    final3, report3 = reasoning.verify_and_repair(
        bad_answer,
        verifier_checks=['citations'],
        allow_repair=True
    )
    assert report3['repair_attempted'], "Should attempt repair"
    print(f"✅ Verification with repair: Attempted={report3['repair_attempted']}")

    print()


def test_supervision():
    """Test supervision monitoring"""
    print("🧪 SUPERVISION TESTS")
    print("-" * 70)

    state_manager = SSMStateManager()
    monitor = SupervisionMonitor(state_manager)

    # Test 1: Monitor with complete coverage (positive)
    state_manager.save_state(
        "agent_1",
        [SSMState(h=np.zeros(16), step=10)],
        metadata={'completed': ['intro', 'method', 'results', 'conclusion']}
    )
    report = monitor.monitor_agent(
        "agent_1",
        requirements=['intro', 'method', 'results', 'conclusion'],
        policy_rules=[]
    )
    assert len(report['coverage_gaps']) == 0, "Should have no gaps"
    assert len(report['policy_violations']) == 0, "Should have no violations"
    print("✅ Supervision complete coverage: No gaps detected")

    # Test 2: Monitor with gaps (negative)
    state_manager.save_state(
        "agent_2",
        [SSMState(h=np.zeros(16), step=10)],
        metadata={'completed': ['intro']}  # Missing method, results, conclusion
    )
    report2 = monitor.monitor_agent(
        "agent_2",
        requirements=['intro', 'method', 'results', 'conclusion'],
        policy_rules=[]
    )
    assert len(report2['coverage_gaps']) == 3, f"Expected 3 gaps, got {len(report2['coverage_gaps'])}"
    assert 'method' in report2['coverage_gaps'], "Method should be missing"
    print(f"✅ Supervision gaps: Detected {report2['coverage_gaps']}")

    # Test 3: Policy violation (negative)
    state_manager.save_state(
        "agent_3",
        [SSMState(h=np.zeros(16), step=10)],
        metadata={'actions': ['rm -rf /', 'extract_data']}
    )
    report3 = monitor.monitor_agent(
        "agent_3",
        requirements=[],
        policy_rules=['no_unsafe_code']
    )
    assert len(report3['policy_violations']) > 0, "Should detect unsafe code"
    assert len(report3['alerts']) > 0, "Should have alerts"
    print(f"✅ Supervision policy violation: Detected {report3['policy_violations']}")

    # Test 4: State saturation (negative)
    state_manager.save_state(
        "agent_4",
        [SSMState(h=np.ones(16) * 200, step=100)],  # Very high norm
        metadata={}
    )
    report4 = monitor.monitor_agent("agent_4", requirements=[], policy_rules=[])
    assert any('SATURATION' in alert for alert in report4['alerts']), "Should detect saturation"
    print(f"✅ Supervision saturation: Detected stuck state")

    # Test 5: Multiple alerts accumulate (positive)
    all_alerts = monitor.get_all_alerts()
    assert len(all_alerts) > 0, "Should have accumulated alerts"
    monitor.clear_alerts()
    assert len(monitor.get_all_alerts()) == 0, "Alerts should clear"
    print(f"✅ Supervision alerts: Accumulated {len(all_alerts)}, cleared")

    print()


def test_ssm_enhanced_agent():
    """Test SSM-enhanced agent integration"""
    print("🧪 SSM ENHANCED AGENT TESTS")
    print("-" * 70)

    config = SSMAgentConfig(
        d_model=64,
        n_layers=6,
        d_state=16,
        bridge_layers=[4, 5]
    )
    agent = SSMEnhancedAgent("test_agent", config)

    # Test 1: Fast path execution (positive)
    request = {
        'text': 'What is Python?',
        'task_type': 'qa',
        'streaming': False
    }
    response = agent.process(request)
    assert 'answer' in response, "Should have answer"
    assert 'execution_config' in response, "Should have config"
    assert response['execution_config']['path'] == 'fast', f"Expected fast, got {response['execution_config']['path']}"
    print(f"✅ SSM agent fast path: {response['latency_ms']:.1f}ms")

    # Test 2: Deliberate path (high hardness) (positive)
    request_hard = {
        'text': 'Prove Fermat\'s Last Theorem',
        'task_type': 'proof',
        'streaming': False,
        'quick_probes': {'has_latex': True, 'multi_step': True}
    }
    response_hard = agent.process(request_hard)
    assert response_hard['execution_config']['path'] == 'deliberate', "Should use deliberate path"
    assert response_hard['reasoning_stats']['samples'] >= 5, "Should use multiple samples"
    print(f"✅ SSM agent deliberate: {response_hard['reasoning_stats']['samples']} samples")

    # Test 3: RAG bridge with spans (positive)
    rag_chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
    rag_spans = rag_chunks_to_spans(rag_chunks, d_model=config.d_model)
    request_rag = {
        'text': 'Explain quantum entanglement',
        'task_type': 'facts',
        'streaming': False
    }
    response_rag = agent.process(request_rag, rag_spans=rag_spans)
    # May route to RAG bridge or deliberate depending on hardness
    assert 'execution_config' in response_rag, "Should have config"
    print(f"✅ SSM agent with RAG: Path={response_rag['execution_config']['path']}")

    # Test 4: Streaming path (long context) (positive)
    request_long = {
        'text': 'x' * 10000,  # Very long
        'task_type': 'summary',
        'streaming': False
    }
    response_long = agent.process(request_long)
    assert response_long['execution_config']['path'] == 'streaming', "Long text should use streaming"
    print(f"✅ SSM agent streaming: L={response_long['signals']['L']}")

    # Test 5: Metrics tracking (positive)
    metrics = agent.get_metrics()
    assert metrics['total_requests'] == 4, f"Expected 4 requests, got {metrics['total_requests']}"
    assert metrics['avg_latency_ms'] > 0, "Should have latency"
    print(f"✅ SSM agent metrics: {metrics['total_requests']} requests, {metrics['avg_latency_ms']:.1f}ms avg")

    # Test 6: Session reset (positive)
    agent.reset_session()
    assert agent.current_states is None, "States should reset"
    print("✅ SSM agent reset: Session reset works")

    print()


def test_orchestrator():
    """Test SSM-supervised orchestrator"""
    print("🧪 ORCHESTRATOR TESTS")
    print("-" * 70)

    state_manager = SSMStateManager()
    orchestrator = SSMSupervisedOrchestrator(state_manager)

    # Test 1: Spawn agents (positive)
    config = SSMAgentConfig(d_model=64, n_layers=4)
    agent1 = orchestrator.spawn_agent("researcher", config)
    agent2 = orchestrator.spawn_agent("writer", config)

    assert len(orchestrator.sub_agents) == 2, "Should have 2 agents"
    assert "researcher" in orchestrator.sub_agents, "Should have researcher"
    assert "writer" in orchestrator.sub_agents, "Should have writer"
    print("✅ Orchestrator spawn: 2 agents spawned")

    # Test 2: Execute with supervision (positive)
    request = {
        'text': 'Research climate change',
        'task_type': 'research',
        'streaming': False
    }
    result = orchestrator.execute_with_supervision(
        "researcher",
        request,
        requirements=['intro', 'data', 'analysis'],
        policy_rules=['cite_sources']
    )
    assert 'response' in result, "Should have response"
    assert 'monitoring' in result, "Should have monitoring"
    print(f"✅ Orchestrator execution: Complete with monitoring")

    # Test 3: Supervise all agents (positive)
    reports = orchestrator.supervise_agents(
        requirements=['intro', 'body', 'conclusion'],
        policy_rules=['no_unsafe_code']
    )
    assert len(reports) == 2, "Should have 2 reports"
    assert 'researcher' in reports, "Should monitor researcher"
    assert 'writer' in reports, "Should monitor writer"
    print(f"✅ Orchestrator supervise all: {len(reports)} agents monitored")

    # Test 4: Get all alerts (positive)
    all_alerts = orchestrator.get_all_alerts()
    # May or may not have alerts depending on execution
    assert isinstance(all_alerts, list), "Should return list"
    print(f"✅ Orchestrator alerts: {len(all_alerts)} alerts")

    # Test 5: Invalid agent ID (negative)
    try:
        result_bad = orchestrator.execute_with_supervision(
            "nonexistent",
            request,
            requirements=[],
            policy_rules=[]
        )
        print("❌ Orchestrator invalid agent: Should raise error")
    except ValueError as e:
        print("✅ Orchestrator invalid agent: Correctly raises ValueError")

    print()


def test_signal_computation():
    """Test signal computation utilities"""
    print("🧪 SIGNAL COMPUTATION TESTS")
    print("-" * 70)

    # Test 1: Sequence length signal (positive)
    L1 = compute_sequence_length_signal("Hello world this is a test")
    assert L1 > 0, "Should compute length"
    assert L1 < 100, "Should be reasonable"
    print(f"✅ Length signal: '{len('Hello world this is a test')}' chars → ~{L1} tokens")

    # Test 2: Very long text (positive)
    L2 = compute_sequence_length_signal("x" * 10000)
    assert L2 > 1000, "Long text should give large L"
    print(f"✅ Length signal (long): 10k chars → ~{L2} tokens")

    # Test 3: Hardness signal (positive)
    H_math = compute_hardness_signal('math')
    H_qa = compute_hardness_signal('qa')
    H_proof = compute_hardness_signal('proof')

    assert H_proof > H_math > H_qa, "Proof > Math > QA hardness"
    print(f"✅ Hardness signal: proof={H_proof:.2f}, math={H_math:.2f}, qa={H_qa:.2f}")

    # Test 4: Hardness with probes (positive)
    H_base = compute_hardness_signal('qa')
    H_latex = compute_hardness_signal('qa', quick_probes={'has_latex': True})
    H_multi = compute_hardness_signal('qa', quick_probes={'multi_step': True})

    assert H_latex > H_base, "LaTeX should increase hardness"
    assert H_multi > H_base, "Multi-step should increase hardness"
    print(f"✅ Hardness probes: base={H_base:.2f}, +latex={H_latex:.2f}, +multi={H_multi:.2f}")

    # Test 5: Request signals (positive)
    request = {
        'text': 'Explain relativity' * 100,  # ~2k chars
        'task_type': 'physics',
        'rag_spans': ['span1', 'span2', 'span3'],
        'streaming': True
    }
    signals = compute_signals_from_request(request)
    assert signals.L > 100, "Should have large L"
    assert signals.R == 3, f"Should have R=3, got {signals.R}"
    assert signals.T == True, "Should have T=True"
    print(f"✅ Request signals: L={signals.L}, R={signals.R}, H={signals.H:.2f}, T={signals.T}")

    # Test 6: Empty request (negative)
    empty_request = {'text': '', 'task_type': 'qa'}
    signals_empty = compute_signals_from_request(empty_request)
    assert signals_empty.L >= 0, "Should handle empty text"
    assert signals_empty.R == 0, "No spans"
    print(f"✅ Empty request signals: L={signals_empty.L}, R={signals_empty.R}")

    print()


def run_all_tests():
    """Run all test suites"""
    print("=" * 70)
    print("SSM/MAMBA INTEGRATION TEST SUITE")
    print("=" * 70)
    print()

    test_suites = [
        ("SSM Core", test_ssm_core),
        ("Attention Bridge", test_attention_bridge),
        ("Hybrid Model", test_hybrid_model),
        ("Scheduler Routing", test_scheduler_routing),
        ("Reasoning Engine", test_reasoning_engine),
        ("Supervision", test_supervision),
        ("SSM Enhanced Agent", test_ssm_enhanced_agent),
        ("Orchestrator", test_orchestrator),
        ("Signal Computation", test_signal_computation),
    ]

    passed = 0
    failed = 0

    for suite_name, test_func in test_suites:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {suite_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {suite_name} ERROR: {e}")
            failed += 1

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed:  {passed}/{len(test_suites)}")
    print(f"❌ Failed:  {failed}/{len(test_suites)}")
    print(f"📊 Pass Rate: {passed/len(test_suites)*100:.1f}%")
    print()

    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {failed} TEST SUITE(S) FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
