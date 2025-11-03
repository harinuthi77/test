"""
SSM/Mamba Integration Layer for Tutoring Orchestrator

Connects the SSM/Mamba architecture with existing multi-agent tutor:
- Wraps HybridSSMTransformer for use with agents
- Integrates AdaptiveScheduler into orchestration
- Adds streaming supervision via SSMStateManager
- Enables enhanced reasoning (self-consistency, PoT)

This layer provides the "adaptive" capabilities:
- Adaptive compute (fast vs deliberate based on signals)
- Adaptive context (handles long sessions efficiently)
- Adaptive supervision (real-time monitoring of sub-agents)
- Adaptive reasoning (more samples for hard problems)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

from ssm_mamba_core import (
    HybridSSMTransformer,
    SSMStateManager,
    SSMState,
    AttentionBridge
)
from adaptive_scheduler import (
    AdaptiveScheduler,
    SchedulerSignals,
    ExecutionConfig,
    ExecutionPath,
    ReasoningEngine,
    SupervisionMonitor,
    compute_signals_from_request,
    create_default_scheduler
)


@dataclass
class SSMAgentConfig:
    """
    Configuration for SSM-enhanced agent

    Combines model architecture with execution strategy
    """
    # Model config
    d_model: int = 768
    n_layers: int = 12
    d_state: int = 16
    bridge_layers: List[int] = None  # Defaults to top 1/3 layers

    # Scheduler config
    long_context_threshold: int = 8000
    high_hardness_threshold: float = 0.6
    precision_tasks: List[str] = None

    # Execution config
    enable_streaming: bool = True
    enable_self_consistency: bool = True
    enable_pot: bool = True  # Program-of-Thoughts
    enable_supervision: bool = True


class SSMEnhancedAgent:
    """
    Agent wrapper that uses SSM/Mamba for inference

    Benefits:
    - O(L) inference time vs O(L²) for pure attention
    - Constant memory per token
    - Streaming state for supervision
    - Hybrid mode for RAG precision
    """

    def __init__(self,
                 agent_id: str,
                 config: SSMAgentConfig,
                 mcp_client=None):
        self.agent_id = agent_id
        self.config = config

        # Create SSM model
        self.model = HybridSSMTransformer(
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            bridge_layers=config.bridge_layers
        )

        # Create scheduler
        self.scheduler = AdaptiveScheduler(
            long_context_threshold=config.long_context_threshold,
            high_hardness_threshold=config.high_hardness_threshold,
            precision_tasks=config.precision_tasks or ['facts', 'legal', 'medical', 'citation']
        )

        # Create reasoning engine
        self.reasoning = ReasoningEngine(mcp_client=mcp_client)

        # State management
        self.state_manager = SSMStateManager()
        self.current_states = None  # Per-layer SSM states

        # Metrics
        self.metrics = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0,
            'path_counts': {'fast': 0, 'rag_bridge': 0, 'deliberate': 0, 'streaming': 0}
        }

    def process(self,
                request: Dict[str, Any],
                rag_spans: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Process request with adaptive execution

        Args:
            request: Request dict with:
                - text: Input text
                - task_type: Task type
                - streaming: Whether tools stream
                - quick_probes: Optional hardness hints
            rag_spans: Retrieved RAG span embeddings (for attention bridge)

        Returns:
            response: Dict with:
                - answer: Generated answer
                - execution_config: Routing decision
                - reasoning_stats: Self-consistency/PoT stats
                - latency_ms: Execution time
        """
        start_time = time.time()

        # Step 1: Compute signals and route
        signals = compute_signals_from_request(request)
        exec_config = self.scheduler.route(signals, task_type=request.get('task_type'))

        # Update metrics
        self.metrics['total_requests'] += 1
        self.metrics['path_counts'][exec_config.path.value] += 1

        # Step 2: Execute based on path
        if exec_config.path == ExecutionPath.FAST:
            answer, stats = self._execute_fast(request, exec_config)

        elif exec_config.path == ExecutionPath.RAG_BRIDGE:
            answer, stats = self._execute_rag_bridge(request, exec_config, rag_spans)

        elif exec_config.path == ExecutionPath.DELIBERATE:
            answer, stats = self._execute_deliberate(request, exec_config, rag_spans)

        elif exec_config.path == ExecutionPath.STREAMING:
            answer, stats = self._execute_streaming(request, exec_config)

        else:
            raise ValueError(f"Unknown execution path: {exec_config.path}")

        # Step 3: Measure latency
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_latency_ms'] += latency_ms

        return {
            'answer': answer,
            'execution_config': exec_config.to_dict(),
            'reasoning_stats': stats,
            'latency_ms': latency_ms,
            'signals': signals.to_dict()
        }

    def _execute_fast(self,
                      request: Dict,
                      config: ExecutionConfig) -> Tuple[str, Dict]:
        """Fast path: single pass, SSM only, low temp"""
        text = request.get('text', '')

        # Convert text to embeddings (mock)
        embeddings = self._text_to_embeddings(text)

        # Forward pass through SSM (no bridge)
        output, new_states = self.model.forward(
            embeddings,
            states=self.current_states,
            bridge_spans=None  # No bridge for fast path
        )

        # Update states
        self.current_states = new_states
        self.state_manager.save_state(
            self.agent_id,
            new_states,
            metadata={'path': 'fast', 'task': request.get('task_type')}
        )

        # Decode output (mock)
        answer = self._embeddings_to_text(output)

        stats = {
            'path': 'fast',
            'samples': 1,
            'bridge_used': False
        }

        return answer, stats

    def _execute_rag_bridge(self,
                           request: Dict,
                           config: ExecutionConfig,
                           rag_spans: Optional[List[np.ndarray]]) -> Tuple[str, Dict]:
        """RAG bridge: SSM + attention over retrieved spans"""
        text = request.get('text', '')
        embeddings = self._text_to_embeddings(text)

        # Forward pass with bridge
        output, new_states = self.model.forward(
            embeddings,
            states=self.current_states,
            bridge_spans=rag_spans  # Attend over RAG spans
        )

        self.current_states = new_states
        self.state_manager.save_state(
            self.agent_id,
            new_states,
            metadata={'path': 'rag_bridge', 'n_spans': len(rag_spans) if rag_spans else 0}
        )

        answer = self._embeddings_to_text(output)

        # Verify citations (since this is precision path)
        if config.run_verifier:
            answer, verify_report = self.reasoning.verify_and_repair(
                answer,
                verifier_checks=['schema', 'citations'],
                allow_repair=False
            )
            stats = {
                'path': 'rag_bridge',
                'samples': 1,
                'bridge_used': True,
                'n_spans': len(rag_spans) if rag_spans else 0,
                'verification': verify_report
            }
        else:
            stats = {
                'path': 'rag_bridge',
                'samples': 1,
                'bridge_used': True,
                'n_spans': len(rag_spans) if rag_spans else 0
            }

        return answer, stats

    def _execute_deliberate(self,
                           request: Dict,
                           config: ExecutionConfig,
                           rag_spans: Optional[List[np.ndarray]]) -> Tuple[str, Dict]:
        """Deliberate: self-consistency + PoT + verification + repair"""
        text = request.get('text', '')
        task_type = request.get('task_type', 'qa')

        # Step 1: Self-consistency (generate K samples)
        samples = []
        for i in range(config.self_consistency_samples):
            embeddings = self._text_to_embeddings(text)

            # Use bridge if spans available
            output, new_states = self.model.forward(
                embeddings,
                states=self.current_states,
                bridge_spans=rag_spans if config.use_bridge else None
            )

            sample = self._embeddings_to_text(output)
            samples.append(sample)

        # Vote on best sample
        # Real impl would use semantic similarity; here we use simple voting
        vote_counts = {}
        for sample in samples:
            vote_counts[sample] = vote_counts.get(sample, 0) + 1
        best_sample = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        agreement_rate = vote_counts[best_sample] / len(samples)

        # Step 2: Program-of-Thoughts (if math/code task)
        pot_result = None
        if config.enable_pot and task_type in ['math', 'code']:
            pot_result, pot_code = self.reasoning.program_of_thoughts(text)

        # Step 3: Verify and repair
        final_answer = best_sample
        verify_report = {}
        if config.run_verifier:
            final_answer, verify_report = self.reasoning.verify_and_repair(
                best_sample,
                verifier_checks=['schema', 'units', 'citations'],
                allow_repair=config.allow_repair
            )

        stats = {
            'path': 'deliberate',
            'samples': config.self_consistency_samples,
            'agreement_rate': agreement_rate,
            'bridge_used': config.use_bridge,
            'pot_enabled': config.enable_pot,
            'pot_result': pot_result,
            'verification': verify_report
        }

        return final_answer, stats

    def _execute_streaming(self,
                          request: Dict,
                          config: ExecutionConfig) -> Tuple[str, Dict]:
        """Streaming: incremental processing with state updates"""
        text = request.get('text', '')

        # Initialize streaming state
        if self.current_states is None:
            self.current_states = [
                SSMState(h=np.zeros(self.config.d_state))
                for _ in range(self.config.n_layers)
            ]

        # Process in chunks (simulate streaming)
        chunk_size = 512
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        outputs = []
        for i, chunk in enumerate(chunks):
            embeddings = self._text_to_embeddings(chunk)
            output, self.current_states = self.model.forward(
                embeddings,
                states=self.current_states,
                bridge_spans=None  # No bridge for streaming (speed priority)
            )
            outputs.append(output)

            # Save state snapshot (for supervision)
            self.state_manager.save_state(
                self.agent_id,
                self.current_states,
                metadata={
                    'path': 'streaming',
                    'chunk': i+1,
                    'total_chunks': len(chunks)
                }
            )

        # Combine outputs
        final_output = np.concatenate(outputs, axis=0) if outputs else np.zeros((1, self.config.d_model))
        answer = self._embeddings_to_text(final_output)

        stats = {
            'path': 'streaming',
            'chunks_processed': len(chunks),
            'bridge_used': False
        }

        return answer, stats

    def _text_to_embeddings(self, text: str) -> np.ndarray:
        """Convert text to embeddings (mock implementation)"""
        # Real implementation would use tokenizer + embedding layer
        # For demo: random embeddings
        length = min(len(text.split()), 512)  # Limit length
        return np.random.randn(length, self.config.d_model) * 0.1

    def _embeddings_to_text(self, embeddings: np.ndarray) -> str:
        """Convert embeddings to text (mock implementation)"""
        # Real implementation would use LM head + detokenizer
        # For demo: placeholder text
        return f"Generated answer (from {embeddings.shape[0]} tokens)"

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        avg_latency = (
            self.metrics['total_latency_ms'] / self.metrics['total_requests']
            if self.metrics['total_requests'] > 0
            else 0
        )

        return {
            'total_requests': self.metrics['total_requests'],
            'avg_latency_ms': avg_latency,
            'path_distribution': {
                path: count / self.metrics['total_requests']
                if self.metrics['total_requests'] > 0 else 0
                for path, count in self.metrics['path_counts'].items()
            },
            'scheduler_stats': self.scheduler.get_stats()
        }

    def reset_session(self):
        """Reset SSM states (e.g., new tutoring session)"""
        self.current_states = None
        self.state_manager.reset_state(self.agent_id)


class SSMSupervisedOrchestrator:
    """
    Orchestrator with SSM-based supervision

    Monitors sub-agents in real-time using streaming SSM state:
    - Detects coverage gaps early (mid-plan, not just at end)
    - Catches policy violations immediately
    - Identifies stuck agents (state saturation)
    - Enables faster corrective turns
    """

    def __init__(self, state_manager: SSMStateManager):
        self.state_manager = state_manager
        self.monitor = SupervisionMonitor(state_manager)
        self.sub_agents: Dict[str, SSMEnhancedAgent] = {}

    def spawn_agent(self,
                    agent_id: str,
                    config: SSMAgentConfig,
                    mcp_client=None) -> SSMEnhancedAgent:
        """Spawn new sub-agent with SSM backing"""
        agent = SSMEnhancedAgent(agent_id, config, mcp_client)
        self.sub_agents[agent_id] = agent
        return agent

    def supervise_agents(self,
                        requirements: List[str],
                        policy_rules: List[str]) -> Dict[str, Dict]:
        """
        Monitor all sub-agents in real-time

        Returns monitoring reports for each agent
        """
        reports = {}
        for agent_id, agent in self.sub_agents.items():
            report = self.monitor.monitor_agent(
                agent_id,
                requirements=requirements,
                policy_rules=policy_rules
            )
            reports[agent_id] = report

        return reports

    def get_all_alerts(self) -> List[str]:
        """Get all supervision alerts"""
        return self.monitor.get_all_alerts()

    def execute_with_supervision(self,
                                 agent_id: str,
                                 request: Dict,
                                 requirements: List[str],
                                 policy_rules: List[str],
                                 rag_spans: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Execute request with real-time supervision

        Returns response + monitoring report
        """
        agent = self.sub_agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Execute
        response = agent.process(request, rag_spans=rag_spans)

        # Supervise
        monitoring = self.monitor.monitor_agent(
            agent_id,
            requirements=requirements,
            policy_rules=policy_rules
        )

        return {
            'response': response,
            'monitoring': monitoring,
            'alerts': monitoring.get('alerts', [])
        }


# Utility: Convert RAG chunks to span embeddings
def rag_chunks_to_spans(chunks: List[str], d_model: int = 768) -> List[np.ndarray]:
    """
    Convert RAG retrieved chunks to span embeddings for attention bridge

    Args:
        chunks: Retrieved text chunks
        d_model: Model dimension

    Returns:
        span_embeddings: List of (S_i, D) arrays
    """
    spans = []
    for chunk in chunks:
        # Real implementation: tokenize + embed
        # For demo: random embeddings
        length = min(len(chunk.split()), 100)  # Limit span length
        span_emb = np.random.randn(length, d_model) * 0.1
        spans.append(span_emb)
    return spans


if __name__ == "__main__":
    print("SSM/Mamba Integration for Adaptive Tutor")
    print("=" * 70)

    # Create SSM-enhanced agent
    config = SSMAgentConfig(
        d_model=768,
        n_layers=12,
        d_state=16,
        bridge_layers=[8, 9, 10, 11],  # Top 4 layers
        enable_self_consistency=True,
        enable_pot=True
    )

    agent = SSMEnhancedAgent("tutor_main", config)
    print(f"✅ Created SSM-enhanced agent:")
    print(f"   - Agent ID: {agent.agent_id}")
    print(f"   - Model: {config.n_layers} layers, d={config.d_model}")
    print(f"   - Bridge layers: {config.bridge_layers}")

    # Test scenarios
    test_requests = [
        {
            'name': 'Simple Q&A (fast path)',
            'request': {
                'text': 'What is the capital of France?',
                'task_type': 'qa',
                'streaming': False
            }
        },
        {
            'name': 'Math problem (deliberate path)',
            'request': {
                'text': 'Prove that the sum of angles in a triangle is 180 degrees',
                'task_type': 'proof',
                'streaming': False,
                'quick_probes': {'has_latex': True, 'multi_step': True}
            }
        },
        {
            'name': 'History with citations (RAG bridge)',
            'request': {
                'text': 'Explain the causes of World War I',
                'task_type': 'history',
                'streaming': False
            },
            'rag_chunks': [
                "The assassination of Archduke Franz Ferdinand in 1914...",
                "Alliance systems between European powers created tensions...",
                "Economic competition and imperialism were underlying causes..."
            ]
        }
    ]

    print("\n" + "=" * 70)
    print("Processing Requests:\n")

    for test in test_requests:
        print(f"📋 {test['name']}")

        # Convert RAG chunks to spans if provided
        rag_spans = None
        if 'rag_chunks' in test:
            rag_spans = rag_chunks_to_spans(test['rag_chunks'], d_model=config.d_model)
            print(f"   RAG spans: {len(rag_spans)} spans")

        # Process
        response = agent.process(test['request'], rag_spans=rag_spans)

        print(f"   Signals: L={response['signals']['L']}, R={response['signals']['R']}, "
              f"H={response['signals']['H']:.2f}, T={response['signals']['T']}")
        print(f"   Path: {response['execution_config']['path'].upper()}")
        print(f"   Bridge: {'Yes' if response['execution_config']['use_bridge'] else 'No'}")
        print(f"   Samples: {response['reasoning_stats'].get('samples', 1)}")
        print(f"   Latency: {response['latency_ms']:.1f}ms")
        if 'agreement_rate' in response['reasoning_stats']:
            print(f"   Agreement: {response['reasoning_stats']['agreement_rate']*100:.1f}%")
        print()

    # Show metrics
    metrics = agent.get_metrics()
    print("=" * 70)
    print("Agent Metrics:")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
    print("\n  Path distribution:")
    for path, pct in metrics['path_distribution'].items():
        print(f"    {path:12s}: {pct*100:5.1f}%")

    # Test supervision
    print("\n" + "=" * 70)
    print("Testing Supervision:\n")

    state_manager = SSMStateManager()
    orchestrator = SSMSupervisedOrchestrator(state_manager)

    # Spawn sub-agents
    researcher = orchestrator.spawn_agent("researcher_1", config)
    writer = orchestrator.spawn_agent("writer_1", config)

    # Simulate work
    researcher.process({
        'text': 'Research French Revolution causes',
        'task_type': 'research',
        'streaming': False
    })

    # Supervise
    reports = orchestrator.supervise_agents(
        requirements=['intro', 'causes', 'timeline', 'impact'],
        policy_rules=['no_unsafe_code', 'cite_sources']
    )

    for agent_id, report in reports.items():
        print(f"✅ Agent: {agent_id}")
        print(f"   Coverage gaps: {report['coverage_gaps']}")
        print(f"   Violations: {report['policy_violations']}")
        print(f"   Alerts: {report['alerts'] if report['alerts'] else 'None'}")

    print("\n" + "=" * 70)
    print("SSM/Mamba integration is ready!")
    print("\nNext steps:")
    print("  1. Integrate with tutoring_orchestrator.py")
    print("  2. Add comprehensive tests")
    print("  3. Run benchmarks (latency, pass@1, faithfulness)")
    print("  4. Update documentation")
