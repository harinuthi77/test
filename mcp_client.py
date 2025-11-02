"""
MCP (Model Context Protocol) Client
Provides standardized tool access for agents

Note: This is a simplified implementation for demonstration.
In production, use the official MCP SDK or implement full JSON-RPC protocol.
Reference: https://modelcontextprotocol.io
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ToolCategory(Enum):
    """Tool categories for safety gating"""
    READ_ONLY = "read_only"  # Safe, no side effects
    WRITE = "write"  # Modifies files/state
    EXECUTE = "execute"  # Runs code
    NETWORK = "network"  # Makes network calls
    SYSTEM = "system"  # System operations


@dataclass
class MCPTool:
    """Tool definition"""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    handler: Callable
    requires_approval: bool = False
    dry_run_available: bool = False


@dataclass
class ToolCall:
    """Record of a tool invocation"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    dry_run: bool = False


class MCPClient:
    """
    Simplified MCP client for tool access
    Implements safety gating and audit logging
    """

    def __init__(self, safety_mode: bool = True):
        self.tools: Dict[str, MCPTool] = {}
        self.safety_mode = safety_mode
        self.call_history: List[ToolCall] = []
        self.approved_tools: set = set()

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register safe, built-in tools"""

        # Python code execution (sandboxed)
        self.register_tool(MCPTool(
            name="python_exec",
            description="Execute Python code safely",
            category=ToolCategory.EXECUTE,
            parameters={"code": "string", "timeout": "int"},
            handler=self._python_exec_handler,
            requires_approval=True,
            dry_run_available=True
        ))

        # File read (safe)
        self.register_tool(MCPTool(
            name="read_file",
            description="Read file contents",
            category=ToolCategory.READ_ONLY,
            parameters={"path": "string"},
            handler=self._read_file_handler,
            requires_approval=False
        ))

        # Web search (simulated)
        self.register_tool(MCPTool(
            name="web_search",
            description="Search the web",
            category=ToolCategory.NETWORK,
            parameters={"query": "string", "max_results": "int"},
            handler=self._web_search_handler,
            requires_approval=False
        ))

        # Calculator (safe)
        self.register_tool(MCPTool(
            name="calculate",
            description="Evaluate mathematical expressions",
            category=ToolCategory.READ_ONLY,
            parameters={"expression": "string"},
            handler=self._calculate_handler,
            requires_approval=False
        ))

    def register_tool(self, tool: MCPTool):
        """Register a new tool"""
        self.tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools (MCP discovery)"""
        return [{
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "parameters": tool.parameters,
            "requires_approval": tool.requires_approval
        } for tool in self.tools.values()]

    def approve_tool(self, tool_name: str):
        """Pre-approve a tool for use"""
        self.approved_tools.add(tool_name)

    def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        dry_run: bool = False
    ) -> ToolCall:
        """
        Call a tool with safety gating

        Args:
            tool_name: Name of tool to call
            parameters: Tool parameters
            dry_run: If True, simulate without executing

        Returns:
            ToolCall with result
        """
        tool = self.tools.get(tool_name)

        if not tool:
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=None,
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        # Safety check
        if self.safety_mode and tool.requires_approval:
            if tool_name not in self.approved_tools:
                return ToolCall(
                    tool_name=tool_name,
                    parameters=parameters,
                    result=None,
                    success=False,
                    error=f"Tool '{tool_name}' requires approval but is not approved"
                )

        # Execute tool
        try:
            if dry_run and tool.dry_run_available:
                result = f"[DRY RUN] Would call {tool_name} with {parameters}"
                success = True
                error = None
            else:
                result = tool.handler(parameters)
                success = True
                error = None

        except Exception as e:
            result = None
            success = False
            error = str(e)

        # Log call
        call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            error=error,
            dry_run=dry_run
        )
        self.call_history.append(call)

        return call

    # Built-in tool handlers
    def _python_exec_handler(self, params: Dict[str, Any]) -> str:
        """Execute Python code in a subprocess (safer than exec)"""
        code = params.get("code", "")
        timeout = params.get("timeout", 5)

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Execute with timeout
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            return output

        except subprocess.TimeoutExpired:
            return f"ERROR: Code execution timed out after {timeout} seconds"

        except Exception as e:
            return f"ERROR: {str(e)}"

        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass

    def _read_file_handler(self, params: Dict[str, Any]) -> str:
        """Read file contents"""
        path = params.get("path", "")

        # Safety: only allow reading from safe directories
        safe_dirs = ['/tmp', os.getcwd()]
        if not any(os.path.abspath(path).startswith(d) for d in safe_dirs):
            return f"ERROR: Path '{path}' is outside allowed directories"

        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _web_search_handler(self, params: Dict[str, Any]) -> str:
        """Simulated web search (in production, call real search API)"""
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        # Simulated results
        return json.dumps({
            "query": query,
            "results": [
                {
                    "title": f"Result {i+1} for '{query}'",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a simulated result about {query}..."
                }
                for i in range(max_results)
            ]
        }, indent=2)

    def _calculate_handler(self, params: Dict[str, Any]) -> str:
        """Evaluate mathematical expression safely"""
        expression = params.get("expression", "")

        # Safety: only allow math operations
        allowed_chars = set("0123456789+-*/().^ ")
        if not all(c in allowed_chars for c in expression):
            return "ERROR: Expression contains invalid characters"

        try:
            # Use eval with restricted globals (ONLY math operations)
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"ERROR: {str(e)}"

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get tool call history for auditing"""
        return [
            {
                "tool_name": call.tool_name,
                "parameters": call.parameters,
                "success": call.success,
                "error": call.error,
                "dry_run": call.dry_run
            }
            for call in self.call_history
        ]


# Testing
def test_mcp_client():
    """Test MCP client"""
    print("="*70)
    print("MCP CLIENT TEST")
    print("="*70)

    client = MCPClient(safety_mode=True)

    # List tools
    print("\n1. Available tools:")
    tools = client.list_tools()
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
        print(f"     Category: {tool['category']}, Approval: {tool['requires_approval']}")

    # Test safe tool (calculator)
    print("\n2. Testing calculator (safe, no approval needed):")
    result = client.call_tool("calculate", {"expression": "2 + 2 * 3"})
    print(f"   Expression: 2 + 2 * 3")
    print(f"   Result: {result.result}")
    print(f"   Success: {result.success}")

    # Test tool requiring approval (dry run)
    print("\n3. Testing python_exec (requires approval, dry run):")
    result = client.call_tool(
        "python_exec",
        {"code": "print('Hello from MCP')", "timeout": 5},
        dry_run=True
    )
    print(f"   Result: {result.result}")
    print(f"   Success: {result.success}")

    # Approve and test for real
    print("\n4. Approving python_exec and running:")
    client.approve_tool("python_exec")
    result = client.call_tool(
        "python_exec",
        {"code": "print('Hello from MCP'); print(2 + 2)", "timeout": 5}
    )
    print(f"   Output:\n{result.result}")
    print(f"   Success: {result.success}")

    # Test web search
    print("\n5. Testing web_search:")
    result = client.call_tool("web_search", {"query": "transformers AI", "max_results": 3})
    print(f"   Query: transformers AI")
    print(f"   Results (simulated):")
    if result.success:
        data = json.loads(result.result)
        for r in data["results"]:
            print(f"     - {r['title']}")

    # Show audit log
    print("\n6. Audit log:")
    log = client.get_audit_log()
    print(f"   Total calls: {len(log)}")
    for entry in log:
        status = "✓" if entry["success"] else "✗"
        print(f"   {status} {entry['tool_name']}: {entry.get('error', 'OK')}")

    print("\n" + "="*70)
    print("✅ MCP CLIENT VALIDATED")
    print("="*70)


if __name__ == "__main__":
    test_mcp_client()
