"""
SSM/Mamba Adaptive System - Unified Module

Consolidates three modules into one:
1. SSM/Mamba Core Architecture (MambaLayer, AttentionBridge, HybridSSM)
2. Adaptive Scheduler (Routing, Reasoning, Supervision)
3. Integration Layer (SSMEnhancedAgent, Orchestrator)

This enables intelligent compute allocation across 5 adaptivity axes:
- Adaptive Compute: Fast vs deliberate reasoning
- Adaptive Context: O(1) memory for long sessions
- Adaptive Knowledge: RAG bridge for precision
- Adaptive Reasoning: Self-consistency + PoT
- Adaptive Governance: Real-time supervision
"""

# Re-export everything from the three modules
# This allows: from ssm_adaptive_system import MambaLayer, AdaptiveScheduler, etc.

# Import from core
from ssm_mamba_core import (
    MambaLayer,
    AttentionBridge,
    HybridSSMTransformer,
    SSMState,
    SSMStateManager,
    compute_sequence_length_signal,
    compute_hardness_signal
)

# Import from scheduler
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

# Import from integration
from ssm_integration import (
    SSMEnhancedAgent,
    SSMAgentConfig,
    SSMSupervisedOrchestrator,
    rag_chunks_to_spans
)

# Define __all__ for clean imports
__all__ = [
    # Core components
    'MambaLayer',
    'AttentionBridge',
    'HybridSSMTransformer',
    'SSMState',
    'SSMStateManager',
    # Scheduler
    'AdaptiveScheduler',
    'SchedulerSignals',
    'ExecutionConfig',
    'ExecutionPath',
    'ReasoningEngine',
    'SupervisionMonitor',
    # Integration
    'SSMEnhancedAgent',
    'SSMAgentConfig',
    'SSMSupervisedOrchestrator',
    # Utilities
    'compute_sequence_length_signal',
    'compute_hardness_signal',
    'compute_signals_from_request',
    'create_default_scheduler',
    'rag_chunks_to_spans',
]

# Quick usage example
if __name__ == "__main__":
    print("SSM/Mamba Adaptive System - Unified Module")
    print("=" * 70)
    print()
    print("Available components:")
    print()
    print("1. SSM/Mamba Core:")
    print("   - MambaLayer: O(L) selective state space layer")
    print("   - AttentionBridge: Thin cross-attention over RAG/tool spans")
    print("   - HybridSSMTransformer: SSM + attention hybrid model")
    print("   - SSMState & SSMStateManager: State management")
    print()
    print("2. Adaptive Scheduler:")
    print("   - AdaptiveScheduler: Routes by L,R,H,T signals")
    print("   - ReasoningEngine: Self-consistency, PoT, verification")
    print("   - SupervisionMonitor: Real-time sub-agent monitoring")
    print()
    print("3. Integration Layer:")
    print("   - SSMEnhancedAgent: Agent wrapper with SSM backing")
    print("   - SSMSupervisedOrchestrator: Multi-agent orchestration")
    print()
    print("Quick Start:")
    print("-" * 70)
    print("from ssm_adaptive_system import SSMEnhancedAgent, SSMAgentConfig")
    print()
    print("config = SSMAgentConfig(d_model=768, n_layers=12)")
    print("agent = SSMEnhancedAgent('tutor', config)")
    print("response = agent.process({'text': 'What is Python?', 'task_type': 'qa'})")
    print()
    print("See README.md for full documentation")
