"""
Multi-Agent Framework - Shared primitives for orchestration
Supports both web scraping (existing) and tutoring (new) modes

NOW WITH LEARNING:
- SQLite database for cross-session learning
- Reflection and stuck detection
- Success/failure pattern tracking
"""

import anthropic
import json
import time
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
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


# ============================================================================
# LEARNING DATABASE - Cross-session learning and memory
# ============================================================================

class LearningDatabase:
    """
    SQLite-based learning system for agents

    Features:
    - Stores successful patterns (what works)
    - Records failures (what doesn't work)
    - Learns website patterns
    - Maintains agent memory across sessions
    - Tracks results and outcomes
    """

    def __init__(self, db_path: str = 'agent_learning.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        """Initialize all learning tables"""
        cursor = self.conn.cursor()

        # Store successful action patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS success_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT,
                website_domain TEXT,
                action_sequence TEXT,
                success_rate REAL DEFAULT 1.0,
                times_used INTEGER DEFAULT 1,
                avg_steps INTEGER,
                last_used TEXT,
                notes TEXT
            )
        ''')

        # Store failed attempts to learn from
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT,
                website_domain TEXT,
                attempted_action TEXT,
                error_type TEXT,
                timestamp TEXT,
                context TEXT
            )
        ''')

        # Store extracted data/results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                task TEXT,
                result_type TEXT,
                result_data TEXT,
                confidence REAL,
                timestamp TEXT
            )
        ''')

        # Store website patterns learned
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS site_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                element_pattern TEXT,
                pattern_type TEXT,
                selector TEXT,
                reliability REAL,
                last_verified TEXT
            )
        ''')

        # Agent's memory/context across sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT,
                context TEXT,
                updated TEXT
            )
        ''')

        self.conn.commit()

    def learn_from_success(self, task_type: str, domain: str, actions: List[str], steps: int, notes: str = ""):
        """Record successful pattern for future use"""
        cursor = self.conn.cursor()
        action_seq = json.dumps(actions)

        # Check if pattern exists
        cursor.execute('''
            SELECT id, success_rate, times_used FROM success_patterns
            WHERE task_type = ? AND website_domain = ? AND action_sequence = ?
        ''', (task_type, domain, action_seq))

        existing = cursor.fetchone()

        if existing:
            # Update existing pattern
            pattern_id, success_rate, times_used = existing
            new_times = times_used + 1
            new_success_rate = (success_rate * times_used + 1.0) / new_times

            cursor.execute('''
                UPDATE success_patterns
                SET success_rate = ?, times_used = ?, avg_steps = ?, last_used = ?, notes = ?
                WHERE id = ?
            ''', (new_success_rate, new_times, steps, datetime.now().isoformat(), notes, pattern_id))
        else:
            # Insert new pattern
            cursor.execute('''
                INSERT INTO success_patterns
                (task_type, website_domain, action_sequence, success_rate, times_used, avg_steps, last_used, notes)
                VALUES (?, ?, ?, 1.0, 1, ?, ?, ?)
            ''', (task_type, domain, action_seq, steps, datetime.now().isoformat(), notes))

        self.conn.commit()

    def learn_from_failure(self, task_type: str, domain: str, action: str, error: str, context: str):
        """Record what didn't work"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO failures (task_type, website_domain, attempted_action, error_type, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_type, domain, action, error, datetime.now().isoformat(), context))
        self.conn.commit()

    def get_learned_strategies(self, task_type: str, domain: str, limit: int = 3) -> List[Dict]:
        """Retrieve proven successful strategies"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT action_sequence, success_rate, times_used, avg_steps, notes
            FROM success_patterns
            WHERE task_type = ? AND website_domain = ?
            ORDER BY success_rate DESC, times_used DESC
            LIMIT ?
        ''', (task_type, domain, limit))

        strategies = []
        for row in cursor.fetchall():
            strategies.append({
                'actions': json.loads(row[0]),
                'success_rate': row[1],
                'times_used': row[2],
                'avg_steps': row[3],
                'notes': row[4]
            })

        return strategies

    def save_result(self, session_id: str, task: str, result_data: Any, confidence: float):
        """Save final results with confidence score"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO results (session_id, task, result_type, result_data, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, task, 'completion', json.dumps(result_data), confidence, datetime.now().isoformat()))
        self.conn.commit()

    def get_memory(self, key: str) -> Optional[str]:
        """Retrieve agent memory by key"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT value FROM agent_memory WHERE key = ?', (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_memory(self, key: str, value: str, context: str = ""):
        """Store agent memory"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO agent_memory (key, value, context, updated)
            VALUES (?, ?, ?, ?)
        ''', (key, value, context, datetime.now().isoformat()))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()


# ============================================================================
# AGENT REFLECTION - Stuck detection and adaptive behavior
# ============================================================================

class AgentReflection:
    """
    Agent's ability to reflect on actions and adapt

    Features:
    - Detects stuck/loop states
    - Suggests alternatives when stuck
    - Tracks progress metrics
    - Records action history
    """

    def __init__(self, learning_db: Optional[LearningDatabase] = None):
        self.learning_db = learning_db
        self.action_history = []
        self.stuck_threshold = 5  # If same action 5 times, we're stuck
        self.progress_metrics = {
            'data_extracted': 0,
            'pages_visited': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'tokens_used': 0,
            'api_calls': 0
        }

    def record_action(self, action: str, success: bool, result: Any = None, context: Dict = None):
        """Record each action and outcome"""
        self.action_history.append({
            'action': action,
            'success': success,
            'result': result,
            'context': context or {},
            'timestamp': time.time()
        })

        if success:
            self.progress_metrics['successful_actions'] += 1
        else:
            self.progress_metrics['failed_actions'] += 1

    def is_stuck(self) -> Tuple[bool, str]:
        """
        Detect if agent is stuck in a loop

        Returns:
            (is_stuck, reason)
        """
        if len(self.action_history) < self.stuck_threshold:
            return False, ""

        recent = self.action_history[-self.stuck_threshold:]
        actions = [a['action'] for a in recent]

        # Check for repetitive actions
        if len(set(actions)) <= 2:
            return True, f"Repeating same actions: {set(actions)}"

        # Check for no progress
        recent_successes = [a['success'] for a in recent]
        if sum(recent_successes) == 0:
            return True, "No successful actions in recent steps"

        # Check for same URL visited repeatedly (web scraping specific)
        recent_urls = [a.get('context', {}).get('url') for a in recent if a.get('context')]
        if recent_urls and len(set(recent_urls)) == 1:
            return True, f"Stuck on same URL: {recent_urls[0]}"

        return False, ""

    def suggest_alternative(self, current_strategy: str) -> str:
        """Suggest different approach when stuck"""
        suggestions = {
            'clicking': 'Try extracting data directly without clicking individual items',
            'scrolling': 'Try using search or filters instead of scrolling',
            'typing': 'Try using buttons or navigation instead',
            'navigation': 'Try going back to homepage and using different path',
            'waiting': 'Try immediate action instead of waiting',
            'searching': 'Try browsing categories instead'
        }

        # Check action history for patterns
        if len(self.action_history) >= 3:
            recent_actions = [a['action'] for a in self.action_history[-3:]]
            for strategy, suggestion in suggestions.items():
                if all(strategy in action.lower() for action in recent_actions):
                    return suggestion

        return suggestions.get(current_strategy.lower(), 'Try a completely different approach')

    def get_progress_summary(self) -> str:
        """Summarize current progress"""
        total_actions = len(self.action_history)
        success_rate = (self.progress_metrics['successful_actions'] / max(1, total_actions)) * 100

        return f"""
📊 PROGRESS METRICS:
   ✓ Successful actions: {self.progress_metrics['successful_actions']}
   ✗ Failed actions: {self.progress_metrics['failed_actions']}
   📄 Pages visited: {self.progress_metrics['pages_visited']}
   📦 Data extracted: {self.progress_metrics['data_extracted']} items
   🎯 API calls: {self.progress_metrics['api_calls']}
   🪙 Tokens used: {self.progress_metrics['tokens_used']:,}

   Success rate: {success_rate:.1f}%
   Total actions: {total_actions}
"""

    def get_action_summary(self, last_n: int = 5) -> str:
        """Get summary of recent actions"""
        if not self.action_history:
            return "No actions yet"

        recent = self.action_history[-last_n:]
        summary = "Recent actions:\n"
        for i, action in enumerate(recent, 1):
            status = "✓" if action['success'] else "✗"
            summary += f"  {status} {action['action']}\n"

        return summary

    def reset(self):
        """Reset tracking for new task"""
        self.action_history = []
        self.progress_metrics = {
            'data_extracted': 0,
            'pages_visited': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'tokens_used': 0,
            'api_calls': 0
        }


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
