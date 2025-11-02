"""
Multi-Agent Framework - Shared primitives for orchestration
Supports both web scraping (existing) and tutoring (new) modes
"""

import anthropic
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(Enum):
    """Agent types in the system"""
    ORCHESTRATOR = "orchestrator"
    DOMAIN_LEAD = "domain_lead"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    MEMORY_CURATOR = "memory_curator"
    SAFETY_SENTINEL = "safety_sentinel"
    WEB_SCRAPER = "web_scraper"  # Original functionality


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    role: AgentRole
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4000
    temperature: float = 0.2
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    memory_limit: int = 10  # Max conversation turns to keep


@dataclass
class TaskSpec:
    """Task specification for any agent operation"""
    task_id: str
    task_type: str  # "web_scraping" or "tutoring"
    input_data: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Standardized agent response"""
    agent_id: str
    role: AgentRole
    content: Any
    citations: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    self_report: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseAgent:
    """
    Base agent class with transformer optimizations built-in
    All agents (web scraper, tutoring agents) inherit from this
    """

    def __init__(self, config: AgentConfig, client: anthropic.Anthropic):
        self.config = config
        self.client = client
        self.agent_id = f"{config.role.value}_{int(time.time() * 1000)}"
        self.conversation_history = []
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "successful_calls": 0,
            "failed_calls": 0
        }

    def _manage_history(self, max_history: int = None) -> List[Dict]:
        """
        TRANSFORMER OPTIMIZATION: Keep only recent turns
        Reduces token count and leverages Claude's KV cache
        """
        max_hist = max_history or self.config.memory_limit
        if len(self.conversation_history) <= max_hist:
            return self.conversation_history

        # Keep last N turns (recency bias in transformers)
        return self.conversation_history[-max_hist:]

    def _optimize_prompt(self, prompt: str) -> str:
        """
        TRANSFORMER OPTIMIZATION: Structure for attention bias
        Critical info at start/end, compress middle
        """
        lines = prompt.split('\n')

        # If prompt is already short, return as-is
        if len(lines) < 20:
            return prompt

        # Extract key sections
        task_lines = [l for l in lines[:5] if l.strip()]  # First 5 lines usually task
        action_lines = [l for l in lines[-5:] if l.strip()]  # Last 5 usually action
        middle_lines = lines[5:-5]

        # Compress middle if too long
        if len(middle_lines) > 20:
            # Keep only lines with key markers
            middle_lines = [l for l in middle_lines if any(
                marker in l for marker in ['🎯', '✅', '📊', '⚡', '🔍', 'TASK:', 'ACTION:']
            )][:20]

        return '\n'.join(task_lines + [''] + middle_lines + [''] + action_lines)

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (4 chars ≈ 1 token)"""
        return len(text) // 4

    def call(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        use_optimization: bool = True
    ) -> AgentResponse:
        """
        Make API call with transformer optimizations

        Args:
            prompt: Text prompt
            images: Optional images (for multi-modal)
            use_optimization: Apply transformer optimizations

        Returns:
            AgentResponse with content and metadata
        """
        self.metrics["total_calls"] += 1

        try:
            # Apply optimizations
            if use_optimization:
                prompt = self._optimize_prompt(prompt)

            # Build message content
            content_blocks = []

            if images:
                for img in images:
                    content_blocks.append({"type": "image", "source": img})

            content_blocks.append({"type": "text", "text": prompt})

            # Manage conversation history
            messages = self._manage_history()
            messages.append({"role": "user", "content": content_blocks})

            # Estimate tokens
            estimated_tokens = self._count_tokens(prompt)
            self.metrics["total_tokens"] += estimated_tokens

            # API call
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=messages
            )

            # Extract response
            answer = response.content[0].text

            # Update history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": answer})

            self.metrics["successful_calls"] += 1

            return AgentResponse(
                agent_id=self.agent_id,
                role=self.config.role,
                content=answer,
                metrics={
                    "estimated_tokens": estimated_tokens,
                    "response_length": len(answer)
                }
            )

        except Exception as e:
            self.metrics["failed_calls"] += 1
            return AgentResponse(
                agent_id=self.agent_id,
                role=self.config.role,
                content=f"ERROR: {str(e)}",
                metrics={"error": str(e)}
            )

    def reset(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "agent_id": self.agent_id,
            "role": self.config.role.value,
            **self.metrics,
            "success_rate": self.metrics["successful_calls"] / max(1, self.metrics["total_calls"]),
            "avg_tokens_per_call": self.metrics["total_tokens"] / max(1, self.metrics["total_calls"])
        }


class MultiAgentOrchestrator:
    """
    Orchestrator that spawns and manages sub-agents
    Supports both web scraping and tutoring modes
    """

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, TaskSpec] = {}
        self.deviation_rules: List[Callable] = []

    def spawn_agent(self, config: AgentConfig) -> BaseAgent:
        """Create a new agent"""
        agent = BaseAgent(config, self.client)
        self.agents[agent.agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve an agent by ID"""
        return self.agents.get(agent_id)

    def register_task(self, task: TaskSpec):
        """Register a new task"""
        self.tasks[task.task_id] = task

    def add_deviation_rule(self, rule: Callable[[AgentResponse], bool]):
        """
        Add a deviation detection rule

        Args:
            rule: Function that takes AgentResponse and returns True if deviant
        """
        self.deviation_rules.append(rule)

    def check_deviations(self, response: AgentResponse) -> List[str]:
        """
        Check if response violates any deviation rules

        Returns:
            List of violation descriptions
        """
        violations = []
        for rule in self.deviation_rules:
            try:
                if rule(response):
                    violations.append(f"Rule violation by {rule.__name__}")
            except Exception as e:
                violations.append(f"Rule check error: {str(e)}")

        return violations

    def correct_deviation(
        self,
        agent: BaseAgent,
        original_prompt: str,
        violations: List[str]
    ) -> AgentResponse:
        """
        Send correction brief and re-run

        Args:
            agent: The agent that deviated
            original_prompt: Original prompt
            violations: List of violations detected

        Returns:
            Corrected response
        """
        correction_prompt = f"""CORRECTION REQUIRED

Your previous response had these issues:
{chr(10).join(f'- {v}' for v in violations)}

Please fix these issues and provide a corrected response.

Original task:
{original_prompt}
"""

        return agent.call(correction_prompt)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get metrics for all agents"""
        return {
            "total_agents": len(self.agents),
            "agents": {
                agent_id: agent.get_metrics()
                for agent_id, agent in self.agents.items()
            }
        }


# Default deviation rules
def check_no_citations(response: AgentResponse) -> bool:
    """Check if response claims facts but provides no citations"""
    content = str(response.content).lower()

    # Check for factual claims
    has_claims = any(word in content for word in [
        'research shows', 'studies indicate', 'according to',
        'data shows', 'evidence suggests', 'proven', 'fact'
    ])

    # Check for citations
    has_citations = len(response.citations) > 0 or any(
        marker in content for marker in ['[', 'source:', 'ref:', 'citation:']
    )

    return has_claims and not has_citations


def check_excessive_length(response: AgentResponse) -> bool:
    """Check if response is too long (token waste)"""
    content = str(response.content)
    estimated_tokens = len(content) // 4
    return estimated_tokens > 5000  # Configurable threshold


def check_low_confidence(response: AgentResponse) -> bool:
    """Check if self-reported confidence is too low"""
    self_report = response.self_report
    if not self_report:
        return False

    confidence = self_report.get('confidence', 1.0)
    return confidence < 0.5  # Configurable threshold
