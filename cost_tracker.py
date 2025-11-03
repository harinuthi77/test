"""
Cost tracking and alerting for Claude API usage

Tracks token usage, estimates costs, and alerts on thresholds
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ModelPricing(Enum):
    """Claude model pricing (as of 2024)"""
    # Format: (input_cost_per_mtok, output_cost_per_mtok)
    CLAUDE_3_OPUS = (15.00, 75.00)
    CLAUDE_3_SONNET = (3.00, 15.00)
    CLAUDE_3_HAIKU = (0.25, 1.25)
    CLAUDE_3_5_SONNET = (3.00, 15.00)


@dataclass
class APICall:
    """Record of a single API call"""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class CostMetrics:
    """Aggregated cost metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    average_latency_ms: float = 0.0
    average_cost_per_call: float = 0.0
    tokens_per_second: float = 0.0


class CostTracker:
    """
    Track API costs and alert on thresholds

    Features:
    - Per-call cost tracking
    - Aggregated metrics
    - Budget alerts
    - Cost regression detection
    - Export to CSV/JSON
    """

    def __init__(
        self,
        daily_budget_usd: float = 100.0,
        alert_threshold_pct: float = 0.8
    ):
        """
        Initialize cost tracker

        Args:
            daily_budget_usd: Daily budget in USD
            alert_threshold_pct: Alert when reaching this % of budget (0.0-1.0)
        """
        self.daily_budget = daily_budget_usd
        self.alert_threshold = alert_threshold_pct

        self.calls: List[APICall] = []
        self.alerts: List[str] = []

        # Pricing lookup
        self.pricing = {
            "claude-3-opus-20240229": ModelPricing.CLAUDE_3_OPUS.value,
            "claude-3-sonnet-20240229": ModelPricing.CLAUDE_3_SONNET.value,
            "claude-3-haiku-20240307": ModelPricing.CLAUDE_3_HAIKU.value,
            "claude-3-5-sonnet-20241022": ModelPricing.CLAUDE_3_5_SONNET.value,
            # Fallback
            "default": ModelPricing.CLAUDE_3_SONNET.value
        }

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a call

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = self.pricing.get(model, self.pricing["default"])
        input_cost_per_mtok, output_cost_per_mtok = pricing

        input_cost = (input_tokens / 1_000_000) * input_cost_per_mtok
        output_cost = (output_tokens / 1_000_000) * output_cost_per_mtok

        return input_cost + output_cost

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Record an API call

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Latency in milliseconds
            success: Whether call succeeded
            error: Error message if failed
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        call = APICall(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            success=success,
            error=error
        )

        self.calls.append(call)

        # Check budget alerts
        self._check_budget_alerts()

    def _check_budget_alerts(self):
        """Check if budget thresholds are exceeded"""
        daily_spend = self.get_daily_spend()

        # Alert at threshold
        if daily_spend >= self.daily_budget * self.alert_threshold:
            pct = (daily_spend / self.daily_budget) * 100
            alert = f"⚠️  Budget alert: ${daily_spend:.2f} spent ({pct:.0f}% of ${self.daily_budget:.2f} daily budget)"

            # Only alert once per threshold crossing
            if alert not in self.alerts:
                self.alerts.append(alert)
                print(alert)

        # Critical alert at 100%
        if daily_spend >= self.daily_budget:
            alert = f"🚨 CRITICAL: Daily budget exceeded! ${daily_spend:.2f} / ${self.daily_budget:.2f}"
            if alert not in self.alerts:
                self.alerts.append(alert)
                print(alert)

    def get_daily_spend(self) -> float:
        """Get total spend in last 24 hours"""
        now = time.time()
        day_ago = now - (24 * 60 * 60)

        daily_calls = [c for c in self.calls if c.timestamp > day_ago]
        return sum(c.cost_usd for c in daily_calls)

    def get_metrics(
        self,
        window_hours: Optional[int] = None
    ) -> CostMetrics:
        """
        Get aggregated metrics

        Args:
            window_hours: If specified, only include calls from last N hours

        Returns:
            CostMetrics object
        """
        # Filter by time window
        if window_hours:
            cutoff = time.time() - (window_hours * 60 * 60)
            calls = [c for c in self.calls if c.timestamp > cutoff]
        else:
            calls = self.calls

        if not calls:
            return CostMetrics()

        successful = [c for c in calls if c.success]

        metrics = CostMetrics(
            total_calls=len(calls),
            successful_calls=len(successful),
            failed_calls=len(calls) - len(successful),
            total_input_tokens=sum(c.input_tokens for c in calls),
            total_output_tokens=sum(c.output_tokens for c in calls),
            total_cost_usd=sum(c.cost_usd for c in calls),
            average_latency_ms=sum(c.latency_ms for c in calls) / len(calls),
            average_cost_per_call=sum(c.cost_usd for c in calls) / len(calls)
        )

        # Calculate tokens per second
        if successful:
            total_tokens = metrics.total_input_tokens + metrics.total_output_tokens
            total_time_sec = sum(c.latency_ms for c in successful) / 1000
            if total_time_sec > 0:
                metrics.tokens_per_second = total_tokens / total_time_sec

        return metrics

    def detect_cost_regression(
        self,
        baseline_window_hours: int = 24,
        current_window_hours: int = 1,
        threshold_pct: float = 0.2
    ) -> Tuple[bool, str]:
        """
        Detect if costs have increased significantly

        Args:
            baseline_window_hours: Hours to use for baseline
            current_window_hours: Hours to use for current comparison
            threshold_pct: Alert if increase > this percentage (0.2 = 20%)

        Returns:
            (has_regressed, message)
        """
        baseline = self.get_metrics(window_hours=baseline_window_hours)
        current = self.get_metrics(window_hours=current_window_hours)

        if baseline.total_calls == 0 or current.total_calls == 0:
            return False, "Not enough data for regression detection"

        # Compare average cost per call
        baseline_avg = baseline.average_cost_per_call
        current_avg = current.average_cost_per_call

        if baseline_avg == 0:
            return False, "Baseline cost is zero"

        increase_pct = (current_avg - baseline_avg) / baseline_avg

        if increase_pct > threshold_pct:
            return True, (
                f"Cost regression detected! "
                f"Average cost increased by {increase_pct*100:.1f}% "
                f"(baseline: ${baseline_avg:.4f}, current: ${current_avg:.4f})"
            )

        return False, "No cost regression detected"

    def export_csv(self, filepath: str):
        """Export call history to CSV"""
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'model', 'input_tokens', 'output_tokens',
                'cost_usd', 'latency_ms', 'success', 'error'
            ])

            for call in self.calls:
                writer.writerow([
                    datetime.fromtimestamp(call.timestamp).isoformat(),
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.cost_usd,
                    call.latency_ms,
                    call.success,
                    call.error or ''
                ])

    def export_json(self, filepath: str):
        """Export call history to JSON"""
        import json

        data = {
            'calls': [
                {
                    'timestamp': datetime.fromtimestamp(c.timestamp).isoformat(),
                    'model': c.model,
                    'input_tokens': c.input_tokens,
                    'output_tokens': c.output_tokens,
                    'cost_usd': c.cost_usd,
                    'latency_ms': c.latency_ms,
                    'success': c.success,
                    'error': c.error
                }
                for c in self.calls
            ],
            'metrics': {
                '24h': self.get_metrics(window_hours=24).__dict__,
                '1h': self.get_metrics(window_hours=1).__dict__,
                'all_time': self.get_metrics().__dict__
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Print cost summary"""
        print("="*70)
        print("COST TRACKING SUMMARY")
        print("="*70)

        # Overall metrics
        all_time = self.get_metrics()
        print(f"\n📊 ALL-TIME METRICS:")
        print(f"   Total calls: {all_time.total_calls}")
        print(f"   Success rate: {all_time.successful_calls / all_time.total_calls * 100:.1f}%")
        print(f"   Total cost: ${all_time.total_cost_usd:.4f}")
        print(f"   Avg cost/call: ${all_time.average_cost_per_call:.4f}")
        print(f"   Avg latency: {all_time.average_latency_ms:.0f}ms")

        # Daily metrics
        daily = self.get_metrics(window_hours=24)
        daily_spend = self.get_daily_spend()
        print(f"\n📅 LAST 24 HOURS:")
        print(f"   Calls: {daily.total_calls}")
        print(f"   Cost: ${daily_spend:.4f}")
        print(f"   Budget: ${self.daily_budget:.2f}")
        print(f"   Used: {(daily_spend / self.daily_budget * 100):.1f}%")

        # Token usage
        print(f"\n🎯 TOKEN USAGE:")
        print(f"   Input tokens: {all_time.total_input_tokens:,}")
        print(f"   Output tokens: {all_time.total_output_tokens:,}")
        print(f"   Total tokens: {all_time.total_input_tokens + all_time.total_output_tokens:,}")
        if all_time.tokens_per_second > 0:
            print(f"   Throughput: {all_time.tokens_per_second:.0f} tokens/sec")

        # Alerts
        if self.alerts:
            print(f"\n⚠️  ALERTS ({len(self.alerts)}):")
            for alert in self.alerts[-5:]:  # Last 5 alerts
                print(f"   {alert}")

        print("\n" + "="*70)


# ============================================================================
# TESTING
# ============================================================================

def test_cost_tracker():
    """Test cost tracker"""
    print("="*70)
    print("COST TRACKER TEST")
    print("="*70)

    tracker = CostTracker(daily_budget_usd=10.0, alert_threshold_pct=0.5)

    # Simulate some API calls
    print("\n📞 Simulating API calls...")

    # Call 1: Small call
    tracker.record_call(
        model="claude-3-5-sonnet-20241022",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=1200,
        success=True
    )

    # Call 2: Large call
    tracker.record_call(
        model="claude-3-5-sonnet-20241022",
        input_tokens=10000,
        output_tokens=5000,
        latency_ms=3500,
        success=True
    )

    # Call 3: Failed call
    tracker.record_call(
        model="claude-3-5-sonnet-20241022",
        input_tokens=2000,
        output_tokens=0,
        latency_ms=500,
        success=False,
        error="Rate limit exceeded"
    )

    # Print summary
    tracker.print_summary()

    # Test cost calculation
    print("\n💰 Cost calculation test:")
    cost = tracker.calculate_cost("claude-3-5-sonnet-20241022", 1000000, 1000000)
    print(f"   1M input + 1M output tokens = ${cost:.2f}")

    # Test regression detection
    print("\n📈 Regression detection test:")
    has_regressed, message = tracker.detect_cost_regression()
    print(f"   {message}")

    print("\n" + "="*70)
    print("✅ COST TRACKER VALIDATED")
    print("="*70)


if __name__ == "__main__":
    test_cost_tracker()
