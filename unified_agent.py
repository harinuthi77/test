"""
UNIFIED ADAPTIVE AGENT
Combines web scraping (original) + tutoring system (new) in one interface
ALL ORIGINAL FUNCTIONALITY PRESERVED - Nothing lost!
"""

import os
import sys
from typing import Optional

# Import original web agent
from adaptive_agent import adaptive_agent as web_scraping_agent

# Import new tutoring system
from tutoring_orchestrator import TutoringOrchestrator, TutoringTask, LearnerProfile
from agent_transformer_optimizations import (
    smart_element_filtering,
    create_compact_element_description,
    optimize_prompt_for_transformer
)


class UnifiedAgent:
    """
    Unified agent supporting both modes:
    1. WEB_SCRAPING: Original functionality (adaptive_agent.py)
    2. TUTORING: New multi-agent tutoring system
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        # Initialize tutoring system (lazy loading)
        self._tutor: Optional[TutoringOrchestrator] = None

    @property
    def tutor(self) -> TutoringOrchestrator:
        """Lazy-load tutoring orchestrator"""
        if self._tutor is None:
            self._tutor = TutoringOrchestrator(self.api_key)
        return self._tutor

    def web_scrape(self, task: str):
        """
        ORIGINAL WEB SCRAPING MODE
        Uses the existing adaptive_agent.py functionality
        NO CHANGES - 100% original behavior
        """
        print("🌐 WEB SCRAPING MODE")
        print("="*70)
        web_scraping_agent(task)

    def teach(
        self,
        topic: str,
        goals: Optional[list] = None,
        learner_age: int = 16,
        learner_level: str = "high school",
        style: str = "step-by-step"
    ) -> dict:
        """
        NEW TUTORING MODE
        Multi-agent system with RAG, MCP, validate-twice

        Args:
            topic: What to teach
            goals: Learning objectives (auto-generated if None)
            learner_age: Student age
            learner_level: Education level
            style: Teaching style

        Returns:
            Dict with presented materials and stored refs
        """
        print("🎓 TUTORING MODE")
        print("="*70)

        # Auto-generate goals if not provided
        if goals is None:
            goals = [
                f"Understand {topic}",
                f"See practical examples of {topic}",
                f"Practice with {topic}"
            ]

        # Create task
        task = TutoringTask(
            topic=topic,
            goals=goals,
            learner=LearnerProfile(
                age=learner_age,
                level=learner_level,
                style_preference=style
            )
        )

        # Run tutoring
        result = self.tutor.tutor(task)

        return {
            "success": result.success,
            "presented": result.presented,
            "stored_refs": result.stored,
            "metrics": result.metrics
        }

    def add_knowledge(self, doc_id: str, text: str, source_name: str, metadata: dict = None):
        """
        Add knowledge source to tutoring system

        Args:
            doc_id: Unique document ID
            text: Document content
            source_name: Human-readable source name
            metadata: Optional metadata (author, year, etc.)
        """
        self.tutor.add_knowledge_source(doc_id, text, source_name, metadata)

    def approve_tool(self, tool_name: str):
        """Approve an MCP tool for tutoring mode"""
        self.tutor.approve_tool(tool_name)

    def get_capabilities(self) -> dict:
        """List agent capabilities"""
        return {
            "modes": {
                "web_scraping": {
                    "description": "Original adaptive web agent with learning",
                    "features": [
                        "Intelligent web navigation",
                        "Data extraction",
                        "Self-learning patterns",
                        "Reflection and adaptation",
                        "Screenshot analysis"
                    ],
                    "method": "web_scrape(task)"
                },
                "tutoring": {
                    "description": "Multi-agent tutoring system",
                    "features": [
                        "RAG-based research",
                        "Multi-agent collaboration",
                        "Citation tracking",
                        "Validate-twice methodology",
                        "MCP tool access",
                        "Deviation detection",
                        "Present vs store logic"
                    ],
                    "method": "teach(topic, ...)"
                }
            },
            "optimizations": {
                "transformer": {
                    "attention_optimization": "Critical info at start/end",
                    "context_management": "Recent turn window",
                    "token_reduction": "50% fewer tokens",
                    "smart_filtering": "Relevant elements only"
                }
            }
        }


def interactive_mode():
    """Interactive CLI for unified agent"""
    print("="*70)
    print("UNIFIED ADAPTIVE AGENT")
    print("Web Scraping + Tutoring in One System")
    print("="*70)

    # Initialize
    try:
        agent = UnifiedAgent()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Set ANTHROPIC_API_KEY environment variable")
        return

    # Show capabilities
    caps = agent.get_capabilities()
    print("\n📋 AVAILABLE MODES:")
    for mode_name, mode_info in caps["modes"].items():
        print(f"\n{mode_name.upper()}:")
        print(f"  {mode_info['description']}")
        print(f"  Method: {mode_info['method']}")

    # Mode selection
    print("\n" + "="*70)
    print("SELECT MODE:")
    print("  1. Web Scraping (original adaptive agent)")
    print("  2. Tutoring (new multi-agent system)")
    print("  3. Show capabilities")
    print("  4. Exit")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            # Web scraping mode
            task = input("\nWhat should the web agent do? ")
            if task:
                agent.web_scrape(task)
            else:
                print("Using default task...")
                agent.web_scrape(
                    "go to walmart.com and find me queen bed frames under $250 "
                    "with at least 1500 reviews and 4+ stars"
                )

        elif choice == "2":
            # Tutoring mode
            topic = input("\nWhat topic should I teach? ")
            if not topic:
                topic = "What are transformers in AI?"

            # Optional: add knowledge
            add_knowledge = input("Add knowledge sources? (y/n): ").lower()
            if add_knowledge == 'y':
                print("\nExample: Adding transformer knowledge...")
                agent.add_knowledge(
                    "transformers_intro",
                    """
                    Transformers are neural network architectures that revolutionized
                    natural language processing. They use self-attention mechanisms to
                    process sequences of data. The architecture was introduced in
                    'Attention Is All You Need' (Vaswani et al., 2017).

                    Key components:
                    1. Self-attention: allows tokens to attend to other tokens
                    2. Multi-head attention: parallel attention mechanisms
                    3. Position encodings: inject sequence order information
                    4. Feed-forward networks: process each position independently

                    Transformers are the basis for modern LLMs like GPT, BERT, and Claude.
                    """,
                    "Transformers Introduction",
                    {"type": "educational", "level": "intermediate"}
                )

            # Run tutoring
            result = agent.teach(
                topic=topic,
                learner_age=18,
                learner_level="undergraduate",
                style="step-by-step"
            )

            # Display results
            print("\n" + "="*70)
            print("LESSON PRESENTED:")
            print("="*70)
            print(f"\nTopic: {result['presented']['topic']}")
            print(f"\nExplanation:\n{result['presented']['explanation']}")

            if result['presented'].get('examples'):
                print(f"\nExamples:")
                for i, ex in enumerate(result['presented']['examples'], 1):
                    print(f"  {i}. {ex}")

            if result['presented'].get('key_citations'):
                print(f"\nSources:")
                for cite in result['presented']['key_citations'][:3]:
                    print(f"  - {cite}")

            print(f"\n📚 Extended materials stored (accessible on demand)")
            print(f"✅ Success: {result['success']}")

        elif choice == "3":
            # Show capabilities
            import json
            print("\n" + "="*70)
            print("AGENT CAPABILITIES:")
            print("="*70)
            print(json.dumps(caps, indent=2))

        elif choice == "4":
            print("\n👋 Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-4.")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "web":
            # Web scraping mode
            task = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
            agent = UnifiedAgent()
            agent.web_scrape(task or "default web scraping task")

        elif mode == "teach":
            # Tutoring mode
            topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "transformers"
            agent = UnifiedAgent()
            result = agent.teach(topic)
            print("\n✅ Tutoring complete!")

        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python unified_agent.py [web|teach|interactive] [args...]")

    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
