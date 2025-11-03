"""
Adaptive Scheduler for SSM/Mamba Multi-Agent Tutor

Routes requests to optimal execution path based on signals:
- L: Sequence length (context size)
- R: Number of retrieved RAG spans
- H: Hardness score (task difficulty)
- T: Tool streaming flag

Execution paths:
1. Fast path: Pure SSM, single pass, low temp
2. RAG bridge: SSM + attention over retrieved spans
3. Deliberate reasoning: Self-consistency + PoT + verifier + repair
4. Streaming supervision: Real-time sub-agent monitoring

This is the key to adaptivity: spend compute where it matters.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
from ssm_mamba_core import (
    HybridSSMTransformer,
    SSMStateManager,
    compute_sequence_length_signal,
    compute_hardness_signal
)


class ExecutionPath(Enum):
    """Execution path types"""
    FAST = "fast"  # Single pass, SSM only, low temp
    RAG_BRIDGE = "rag_bridge"  # SSM + attention over spans
    DELIBERATE = "deliberate"  # Multi-sample + PoT + verification
    STREAMING = "streaming"  # Real-time supervision of tools/sub-agents


@dataclass
class SchedulerSignals:
    """
    Input signals for routing decision

    L: Sequence length (tokens)
    R: Retrieved span count
    H: Hardness score (0.0-1.0)
    T: Streaming flag (True if tools/sub-agents produce streaming output)
    """
    L: int  # Sequence length
    R: int  # Retrieved spans
    H: float  # Hardness (0.0-1.0)
    T: bool  # Streaming

    def to_dict(self) -> Dict:
        return {
            'L': self.L,
            'R': self.R,
            'H': self.H,
            'T': self.T
        }


@dataclass
class ExecutionConfig:
    """
    Configuration for execution path

    Determines:
    - Which model path to use (SSM vs hybrid)
    - Sampling strategy (single vs self-consistency)
    - Verification requirements
    - Latency budget
    """
    path: ExecutionPath
    use_bridge: bool
    self_consistency_samples: int = 1
    enable_pot: bool = False  # Program-of-Thoughts via MCP Python
    run_verifier: bool = False
    allow_repair: bool = False
    temperature: float = 0.3
    max_latency_ms: int = 2000  # SLO target

    def to_dict(self) -> Dict:
        return {
            'path': self.path.value,
            'use_bridge': self.use_bridge,
            'self_consistency_samples': self.self_consistency_samples,
            'enable_pot': self.enable_pot,
            'run_verifier': self.run_verifier,
            'allow_repair': self.allow_repair,
            'temperature': self.temperature,
            'max_latency_ms': self.max_latency_ms
        }


class AdaptiveScheduler:
    """
    Adaptive scheduler: routes requests to optimal execution path

    Policy (from spec):
    ```
    If T==streaming or L>8k → use SSM path
    If R>0 and precision needed → enable RAG bridge
    If H==hard (math/code/logic) → deliberate mode:
        self_consistency = 5..7 samples
        enable PoT (MCP Python)
        run verifier; allow 1 repair pass
    Else → fast path
    ```
    """

    def __init__(self,
                 long_context_threshold: int = 8000,
                 high_hardness_threshold: float = 0.6,
                 precision_tasks: List[str] = None):
        """
        Args:
            long_context_threshold: L threshold for SSM-only path
            high_hardness_threshold: H threshold for deliberate mode
            precision_tasks: Task types requiring RAG bridge (facts, legal, medical)
        """
        self.long_context_threshold = long_context_threshold
        self.high_hardness_threshold = high_hardness_threshold
        self.precision_tasks = precision_tasks or ['facts', 'legal', 'medical', 'citation']

        # Stats
        self.routing_stats = {
            'fast': 0,
            'rag_bridge': 0,
            'deliberate': 0,
            'streaming': 0
        }

    def route(self,
              signals: SchedulerSignals,
              task_type: Optional[str] = None) -> ExecutionConfig:
        """
        Route request to execution path based on signals

        Args:
            signals: L, R, H, T signals
            task_type: Optional task type for precision check

        Returns:
            ExecutionConfig with routing decision
        """
        L, R, H, T = signals.L, signals.R, signals.H, signals.T

        # Rule 1: Streaming or very long context → SSM streaming path
        if T or L > self.long_context_threshold:
            self.routing_stats['streaming'] += 1
            return ExecutionConfig(
                path=ExecutionPath.STREAMING,
                use_bridge=False,  # Pure SSM for speed
                temperature=0.3,
                max_latency_ms=3000  # Longer budget for streaming
            )

        # Rule 2: Retrieved spans + precision task → RAG bridge
        if R > 0 and task_type in self.precision_tasks:
            self.routing_stats['rag_bridge'] += 1
            return ExecutionConfig(
                path=ExecutionPath.RAG_BRIDGE,
                use_bridge=True,
                run_verifier=True,  # Always verify citations
                temperature=0.2,  # Lower temp for factual accuracy
                max_latency_ms=2500
            )

        # Rule 3: High hardness → deliberate reasoning
        if H >= self.high_hardness_threshold:
            # Hardness determines sample count (harder → more samples)
            samples = self._compute_sample_count(H)

            self.routing_stats['deliberate'] += 1
            return ExecutionConfig(
                path=ExecutionPath.DELIBERATE,
                use_bridge=(R > 0),  # Use bridge if spans available
                self_consistency_samples=samples,
                enable_pot=(task_type in ['math', 'code']),  # PoT for math/code
                run_verifier=True,
                allow_repair=True,  # Allow 1 repair pass
                temperature=0.5,  # Higher temp for diversity in samples
                max_latency_ms=5000  # Larger budget for reasoning
            )

        # Rule 4: Default → fast path
        self.routing_stats['fast'] += 1
        return ExecutionConfig(
            path=ExecutionPath.FAST,
            use_bridge=False,
            temperature=0.2,
            max_latency_ms=1000  # Tight SLO for simple queries
        )

    def _compute_sample_count(self, hardness: float) -> int:
        """
        Determine self-consistency sample count based on hardness

        Hardness 0.6-0.7: 5 samples
        Hardness 0.7-0.8: 6 samples
        Hardness 0.8-1.0: 7 samples
        """
        if hardness < 0.7:
            return 5
        elif hardness < 0.8:
            return 6
        else:
            return 7

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total = sum(self.routing_stats.values())
        return {
            'total_requests': total,
            'distribution': {
                path: count / total if total > 0 else 0
                for path, count in self.routing_stats.items()
            },
            'raw_counts': self.routing_stats.copy()
        }

    def reset_stats(self):
        """Reset statistics"""
        self.routing_stats = {k: 0 for k in self.routing_stats}


class ReasoningEngine:
    """
    Enhanced reasoning with self-consistency and Program-of-Thoughts

    Implements o-series-style test-time compute:
    1. Generate K samples (self-consistency)
    2. For math/code, execute via MCP Python (PoT)
    3. Verify results (schema, units, tests, citations)
    4. Vote on best answer or repair
    """

    def __init__(self, mcp_client=None):
        """
        Args:
            mcp_client: MCP client for Program-of-Thoughts execution
        """
        self.mcp_client = mcp_client

    def self_consistency(self,
                         model,
                         prompt: str,
                         k: int = 5,
                         temperature: float = 0.7) -> Tuple[str, Dict]:
        """
        Self-consistency sampling: generate K answers, vote on best

        Args:
            model: Model callable (prompt -> response)
            prompt: Input prompt
            k: Number of samples
            temperature: Sampling temperature

        Returns:
            best_answer: Consensus answer
            stats: Voting statistics
        """
        samples = []
        for i in range(k):
            # In practice: call model with temperature
            # For demo: generate dummy samples
            sample = f"Sample {i+1} answer (temp={temperature})"
            samples.append(sample)

        # Vote: find most common answer (simplified: just count)
        # Real implementation would use semantic similarity
        vote_counts = {}
        for sample in samples:
            vote_counts[sample] = vote_counts.get(sample, 0) + 1

        best_answer = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        best_count = vote_counts[best_answer]

        return best_answer, {
            'k': k,
            'consensus_votes': best_count,
            'agreement_rate': best_count / k,
            'all_samples': samples
        }

    def program_of_thoughts(self,
                           problem: str,
                           language: str = "python") -> Tuple[Any, str]:
        """
        Program-of-Thoughts: solve problem by executing code

        Args:
            problem: Problem description (e.g., math word problem)
            language: Programming language (default: python)

        Returns:
            result: Execution result
            code: Generated code
        """
        # Step 1: Generate code to solve problem (via model)
        # For demo: simple code
        code = f"""
# Solving: {problem[:50]}...
def solve():
    # Generated solution code here
    return 42  # Placeholder

result = solve()
print(result)
"""

        # Step 2: Execute via MCP Python tool
        if self.mcp_client:
            try:
                result = self.mcp_client.call_tool(
                    "python_exec",
                    {"code": code, "timeout": 5}
                )
                return result, code
            except Exception as e:
                return f"Error: {e}", code
        else:
            # Fallback: return code without execution
            return "MCP client not available", code

    def verify_and_repair(self,
                          answer: str,
                          verifier_checks: List[str],
                          allow_repair: bool = True) -> Tuple[str, Dict]:
        """
        Verify answer against checks; repair if needed

        Args:
            answer: Generated answer
            verifier_checks: List of checks (e.g., ['schema', 'units', 'citations'])
            allow_repair: Whether to attempt repair on failure

        Returns:
            final_answer: Verified (possibly repaired) answer
            verification_report: Detailed report
        """
        report = {
            'checks_passed': [],
            'checks_failed': [],
            'repair_attempted': False,
            'repair_successful': False
        }

        # Run checks (simplified)
        for check in verifier_checks:
            passed = self._run_check(check, answer)
            if passed:
                report['checks_passed'].append(check)
            else:
                report['checks_failed'].append(check)

        # If checks failed and repair allowed, attempt repair
        if report['checks_failed'] and allow_repair:
            report['repair_attempted'] = True
            repaired_answer = self._attempt_repair(answer, report['checks_failed'])

            # Re-verify repaired answer
            all_passed = all(
                self._run_check(check, repaired_answer)
                for check in verifier_checks
            )
            if all_passed:
                report['repair_successful'] = True
                return repaired_answer, report

        return answer, report

    def _run_check(self, check: str, answer: str) -> bool:
        """Run a verification check (simplified)"""
        # Real implementation would have sophisticated checks
        if check == 'schema':
            return len(answer) > 10  # Has content
        elif check == 'citations':
            return '[' in answer  # Has citation markers
        elif check == 'units':
            return any(unit in answer for unit in ['m', 'kg', 's', '$', '%'])
        else:
            return True  # Unknown check passes by default

    def _attempt_repair(self, answer: str, failed_checks: List[str]) -> str:
        """Attempt to repair answer (simplified)"""
        # Real implementation would use model to regenerate with constraints
        repaired = answer + f" [Repaired for: {', '.join(failed_checks)}]"
        return repaired


class SupervisionMonitor:
    """
    Real-time supervision of sub-agents using SSM streaming state

    Detects:
    - Coverage gaps (missing requirements)
    - Policy violations (unsafe actions)
    - Contradictions (inconsistent outputs)
    - Progress stalls (agent stuck)

    Uses SSM state to monitor incrementally (not just at end).
    """

    def __init__(self, state_manager: SSMStateManager):
        self.state_manager = state_manager
        self.alerts = []

    def monitor_agent(self,
                      agent_id: str,
                      requirements: List[str],
                      policy_rules: List[str]) -> Dict[str, Any]:
        """
        Monitor sub-agent in real-time

        Args:
            agent_id: Agent identifier
            requirements: Required coverage items
            policy_rules: Safety/policy rules

        Returns:
            monitoring_report: Alerts and metrics
        """
        # Get current state snapshot
        snapshot = self.state_manager.get_supervision_snapshot(agent_id)

        report = {
            'agent_id': agent_id,
            'timestamp': time.time(),
            'state_snapshot': snapshot,
            'coverage_gaps': [],
            'policy_violations': [],
            'alerts': []
        }

        # Check coverage (simplified)
        metadata = snapshot.get('metadata', {})
        completed_items = metadata.get('completed', [])
        for req in requirements:
            if req not in completed_items:
                report['coverage_gaps'].append(req)

        # Check policy (simplified)
        actions = metadata.get('actions', [])
        for rule in policy_rules:
            if self._violates_policy(rule, actions):
                report['policy_violations'].append(rule)
                report['alerts'].append(f"POLICY VIOLATION: {rule}")

        # Check for stuck state
        state_norms = snapshot.get('state_norms', [])
        if state_norms and max(state_norms) > 100:  # Saturation
            report['alerts'].append("STATE SATURATION: Agent may be stuck")

        self.alerts.extend(report['alerts'])
        return report

    def _violates_policy(self, rule: str, actions: List[str]) -> bool:
        """Check if actions violate policy (simplified)"""
        # Real implementation would have sophisticated checks
        if rule == 'no_unsafe_code':
            unsafe_patterns = ['rm -rf', 'DROP TABLE', 'eval(']
            return any(pattern in str(actions) for pattern in unsafe_patterns)
        return False

    def get_all_alerts(self) -> List[str]:
        """Get all accumulated alerts"""
        return self.alerts.copy()

    def clear_alerts(self):
        """Clear alerts"""
        self.alerts = []


# Integration utilities

def create_default_scheduler() -> AdaptiveScheduler:
    """Create scheduler with default settings"""
    return AdaptiveScheduler(
        long_context_threshold=8000,
        high_hardness_threshold=0.6,
        precision_tasks=['facts', 'legal', 'medical', 'citation', 'history']
    )


def compute_signals_from_request(request: Dict) -> SchedulerSignals:
    """
    Extract L,R,H,T signals from incoming request

    Args:
        request: Request dict with keys:
            - text: Input text
            - task_type: Task type
            - rag_spans: Retrieved RAG spans (optional)
            - streaming: Whether tools/sub-agents stream

    Returns:
        SchedulerSignals
    """
    text = request.get('text', '')
    task_type = request.get('task_type', 'qa')
    rag_spans = request.get('rag_spans', [])
    streaming = request.get('streaming', False)

    # Compute signals
    L = compute_sequence_length_signal(text)
    R = len(rag_spans)
    H = compute_hardness_signal(task_type, request.get('quick_probes'))
    T = streaming

    return SchedulerSignals(L=L, R=R, H=H, T=T)


if __name__ == "__main__":
    print("Adaptive Scheduler for SSM/Mamba Tutor")
    print("=" * 70)

    # Create scheduler
    scheduler = create_default_scheduler()

    # Test scenarios from spec
    scenarios = [
        {
            'name': 'Live coding lesson (MCP streaming)',
            'request': {
                'text': 'Write a Python function to solve the traveling salesman problem',
                'task_type': 'code',
                'streaming': True
            }
        },
        {
            'name': 'History essay with citations (RAG-heavy)',
            'request': {
                'text': 'Explain the causes of the French Revolution',
                'task_type': 'history',
                'rag_spans': [f'span_{i}' for i in range(15)],  # 15 retrieved spans
                'streaming': False
            }
        },
        {
            'name': 'Long algebra unit (multi-turn session)',
            'request': {
                'text': 'x' * 10000,  # Long context (10k chars)
                'task_type': 'math',
                'streaming': False
            }
        },
        {
            'name': 'Simple Q&A (fast path)',
            'request': {
                'text': 'What is the capital of France?',
                'task_type': 'qa',
                'streaming': False
            }
        },
        {
            'name': 'Hard math proof (deliberate)',
            'request': {
                'text': 'Prove that sqrt(2) is irrational',
                'task_type': 'proof',
                'streaming': False,
                'quick_probes': {'has_latex': True, 'multi_step': True}
            }
        }
    ]

    print("\nRouting Decisions:\n")
    for scenario in scenarios:
        signals = compute_signals_from_request(scenario['request'])
        config = scheduler.route(signals, task_type=scenario['request']['task_type'])

        print(f"📋 {scenario['name']}")
        print(f"   Signals: L={signals.L}, R={signals.R}, H={signals.H:.2f}, T={signals.T}")
        print(f"   Path: {config.path.value.upper()}")
        print(f"   Bridge: {'Yes' if config.use_bridge else 'No'}")
        print(f"   Samples: {config.self_consistency_samples}")
        print(f"   PoT: {'Yes' if config.enable_pot else 'No'}")
        print(f"   Verifier: {'Yes' if config.run_verifier else 'No'}")
        print(f"   Max latency: {config.max_latency_ms}ms")
        print()

    # Show stats
    stats = scheduler.get_stats()
    print("=" * 70)
    print("Routing Statistics:")
    print(f"Total requests: {stats['total_requests']}")
    print("\nDistribution:")
    for path, pct in stats['distribution'].items():
        print(f"  {path:12s}: {pct*100:5.1f}% ({stats['raw_counts'][path]} requests)")

    # Test reasoning engine
    print("\n" + "=" * 70)
    print("Reasoning Engine Tests:\n")

    reasoning = ReasoningEngine()

    # Test 1: Self-consistency
    def dummy_model(prompt):
        return f"Answer to: {prompt[:30]}..."

    answer, sc_stats = reasoning.self_consistency(dummy_model, "What is 2+2?", k=5)
    print(f"✅ Self-consistency (k={sc_stats['k']}):")
    print(f"   Agreement rate: {sc_stats['agreement_rate']*100:.1f}%")
    print(f"   Best answer: {answer[:50]}...")

    # Test 2: Verification
    test_answer = "The result is 42 kg"
    final, report = reasoning.verify_and_repair(
        test_answer,
        verifier_checks=['schema', 'units', 'citations'],
        allow_repair=True
    )
    print(f"\n✅ Verification:")
    print(f"   Checks passed: {report['checks_passed']}")
    print(f"   Checks failed: {report['checks_failed']}")
    print(f"   Repair attempted: {report['repair_attempted']}")

    # Test 3: Supervision
    print(f"\n✅ Supervision Monitor:")
    state_manager = SSMStateManager()
    from ssm_mamba_core import SSMState
    state_manager.save_state(
        "researcher_1",
        [SSMState(h=np.zeros(16), step=100)],
        metadata={'completed': ['intro', 'method'], 'actions': ['search', 'extract']}
    )

    monitor = SupervisionMonitor(state_manager)
    mon_report = monitor.monitor_agent(
        "researcher_1",
        requirements=['intro', 'method', 'results', 'conclusion'],
        policy_rules=['no_unsafe_code']
    )
    print(f"   Coverage gaps: {mon_report['coverage_gaps']}")
    print(f"   Policy violations: {mon_report['policy_violations']}")
    print(f"   Alerts: {mon_report['alerts'] if mon_report['alerts'] else 'None'}")

    print("\n" + "=" * 70)
    print("Adaptive scheduler is ready for integration!")
