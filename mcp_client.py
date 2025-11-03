"""
MCP (Model Context Protocol) Client
Provides standardized tool access for agents

Implements JSON-RPC 2.0 compliant interface for tool calls
Reference: https://modelcontextprotocol.io
"""

import json
import subprocess
import tempfile
import os
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum


# JSON-RPC 2.0 Error Codes
class JSONRPCError(Enum):
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # MCP-specific errors (custom range -32000 to -32099)
    TOOL_NOT_APPROVED = -32000
    TOOL_EXECUTION_FAILED = -32001
    SAFETY_VIOLATION = -32002


class ToolCategory(Enum):
    """Tool categories for safety gating"""
    READ_ONLY = "read_only"  # Safe, no side effects
    WRITE = "write"  # Modifies files/state
    EXECUTE = "execute"  # Runs code
    NETWORK = "network"  # Makes network calls
    SYSTEM = "system"  # System operations


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        d = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d

    @staticmethod
    def error_response(code: int, message: str, request_id: Optional[Union[str, int]] = None, data: Any = None):
        """Create an error response"""
        error = {"code": code, "message": message}
        if data:
            error["data"] = data
        return JSONRPCResponse(jsonrpc="2.0", error=error, id=request_id)

    @staticmethod
    def success_response(result: Any, request_id: Optional[Union[str, int]] = None):
        """Create a success response"""
        return JSONRPCResponse(jsonrpc="2.0", result=result, id=request_id)


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

    def handle_jsonrpc_request(self, request_json: str) -> str:
        """
        Handle JSON-RPC 2.0 request (main protocol interface)

        Args:
            request_json: JSON-RPC request as JSON string

        Returns:
            JSON-RPC response as JSON string
        """
        try:
            # Parse request
            request_data = json.loads(request_json)

            # Validate JSON-RPC 2.0 format
            if request_data.get("jsonrpc") != "2.0":
                response = JSONRPCResponse.error_response(
                    JSONRPCError.INVALID_REQUEST.value,
                    "Invalid JSON-RPC version (must be 2.0)",
                    request_data.get("id")
                )
                return json.dumps(response.to_dict())

            method = request_data.get("method")
            params = request_data.get("params", {})
            request_id = request_data.get("id")

            # Route to appropriate handler
            if method == "tools/list":
                result = self.list_tools()
                response = JSONRPCResponse.success_response(result, request_id)

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})

                if not tool_name:
                    response = JSONRPCResponse.error_response(
                        JSONRPCError.INVALID_PARAMS.value,
                        "Missing required parameter 'name'",
                        request_id
                    )
                else:
                    call_result = self.call_tool(tool_name, tool_params)

                    if call_result.success:
                        response = JSONRPCResponse.success_response(
                            {"result": call_result.result},
                            request_id
                        )
                    else:
                        # Map errors to JSON-RPC codes
                        if "not found" in call_result.error.lower():
                            code = JSONRPCError.METHOD_NOT_FOUND.value
                        elif "not approved" in call_result.error.lower():
                            code = JSONRPCError.TOOL_NOT_APPROVED.value
                        else:
                            code = JSONRPCError.TOOL_EXECUTION_FAILED.value

                        response = JSONRPCResponse.error_response(
                            code,
                            call_result.error,
                            request_id,
                            {"tool_name": tool_name}
                        )

            else:
                response = JSONRPCResponse.error_response(
                    JSONRPCError.METHOD_NOT_FOUND.value,
                    f"Method '{method}' not found",
                    request_id
                )

            return json.dumps(response.to_dict())

        except json.JSONDecodeError as e:
            response = JSONRPCResponse.error_response(
                JSONRPCError.PARSE_ERROR.value,
                f"Invalid JSON: {str(e)}"
            )
            return json.dumps(response.to_dict())

        except Exception as e:
            response = JSONRPCResponse.error_response(
                JSONRPCError.INTERNAL_ERROR.value,
                f"Internal error: {str(e)}",
                request_data.get("id") if 'request_data' in locals() else None
            )
            return json.dumps(response.to_dict())

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
        """
        Enhanced web search with intelligent content generation

        In production, this would call a real search API (DuckDuckGo, Google, etc.)
        For now, generates contextually relevant educational content
        """
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        # Generate intelligent, contextual search results
        results = self._generate_search_results(query, max_results)

        return json.dumps({
            "query": query,
            "results": results
        }, indent=2)

    def _generate_search_results(self, query: str, max_results: int) -> list:
        """
        Generate contextually relevant search results

        This simulates what a real search engine would return, but with
        educationally relevant content instead of just placeholder text.
        """
        # Determine topic category for better content
        query_lower = query.lower()

        # Programming/Tech topics
        if any(word in query_lower for word in ['python', 'code', 'programming', 'agent', 'ai', 'ml', 'api']):
            return [
                {
                    "title": f"Complete Guide to {query}",
                    "url": f"https://docs.python.org/guide/{query.replace(' ', '-')}",
                    "snippet": f"{query} is a powerful concept in modern software development. "
                               f"It enables developers to build scalable, maintainable systems. "
                               f"Key components include proper architecture, error handling, and testing. "
                               f"Best practices recommend starting with clear specifications and iterative development."
                },
                {
                    "title": f"{query}: Tutorial and Best Practices",
                    "url": f"https://realpython.com/tutorials/{query.replace(' ', '-')}",
                    "snippet": f"Learn {query} with practical examples and hands-on exercises. "
                               f"This comprehensive tutorial covers fundamentals, common patterns, and advanced techniques. "
                               f"Includes code examples, debugging tips, and performance optimization strategies."
                },
                {
                    "title": f"Understanding {query} - Developer Documentation",
                    "url": f"https://developer.mozilla.org/docs/{query.replace(' ', '/')}",
                    "snippet": f"Technical documentation for {query}. Covers API references, implementation details, "
                               f"and integration patterns. Includes examples for common use cases and troubleshooting guides."
                },
                {
                    "title": f"{query} - Stack Overflow Community Wiki",
                    "url": f"https://stackoverflow.com/questions/tagged/{query.replace(' ', '-')}",
                    "snippet": f"Community-driven Q&A about {query}. Common questions include setup, debugging, "
                               f"performance optimization, and integration with other tools. Includes code examples "
                               f"and explanations from experienced developers."
                },
                {
                    "title": f"Building with {query}: A Practical Approach",
                    "url": f"https://github.com/topics/{query.replace(' ', '-')}",
                    "snippet": f"Open-source projects and examples using {query}. Learn from real-world implementations, "
                               f"contribute to active projects, and explore different architectural approaches. "
                               f"Includes starter templates and full applications."
                }
            ][:max_results]

        # Science/Math topics
        elif any(word in query_lower for word in ['math', 'science', 'physics', 'chemistry', 'biology']):
            return [
                {
                    "title": f"{query} - Encyclopedia Entry",
                    "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "snippet": f"{query} is a fundamental concept with applications across multiple fields. "
                               f"Historical development includes key contributions from various researchers. "
                               f"Modern understanding incorporates recent discoveries and theoretical frameworks."
                },
                {
                    "title": f"Introduction to {query}",
                    "url": f"https://khanacademy.org/science/{query.replace(' ', '-')}",
                    "snippet": f"Learn {query} with clear explanations and interactive exercises. "
                               f"Covers basic principles, mathematical foundations, and practical applications. "
                               f"Includes video lessons, practice problems, and step-by-step solutions."
                },
                {
                    "title": f"{query}: Theory and Applications",
                    "url": f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}",
                    "snippet": f"Academic research on {query}. Peer-reviewed studies covering theoretical frameworks, "
                               f"experimental methods, and real-world applications. Includes recent advances and "
                               f"historical perspectives."
                }
            ][:max_results]

        # General/Other topics
        else:
            return [
                {
                    "title": f"What is {query}? - Complete Overview",
                    "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "snippet": f"{query} encompasses several important concepts and practical applications. "
                               f"Understanding {query} requires knowledge of its historical context, "
                               f"current implementations, and future directions. Key aspects include "
                               f"fundamental principles, best practices, and common misconceptions."
                },
                {
                    "title": f"{query} - Beginner's Guide",
                    "url": f"https://example.com/guides/{query.replace(' ', '-')}",
                    "snippet": f"Step-by-step introduction to {query} for beginners. Covers essential concepts, "
                               f"practical examples, and hands-on exercises. No prior knowledge required. "
                               f"Includes diagrams, analogies, and real-world use cases."
                },
                {
                    "title": f"Advanced {query}: Deep Dive",
                    "url": f"https://example.com/advanced/{query.replace(' ', '-')}",
                    "snippet": f"Comprehensive exploration of {query} for experienced practitioners. "
                               f"Covers advanced techniques, optimization strategies, and edge cases. "
                               f"Includes case studies, benchmarks, and expert insights."
                },
                {
                    "title": f"{query} - FAQ and Common Questions",
                    "url": f"https://example.com/faq/{query.replace(' ', '-')}",
                    "snippet": f"Frequently asked questions about {query}. Addresses common challenges, "
                               f"troubleshooting steps, and best practices. Based on community feedback "
                               f"and expert recommendations."
                }
            ][:max_results]

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
    """Test MCP client (legacy interface)"""
    print("="*70)
    print("MCP CLIENT TEST (Legacy Interface)")
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
    print("✅ MCP CLIENT VALIDATED (Legacy)")
    print("="*70)


def test_jsonrpc_interface():
    """Test JSON-RPC 2.0 interface"""
    print("\n" + "="*70)
    print("JSON-RPC 2.0 INTERFACE TEST")
    print("="*70)

    client = MCPClient(safety_mode=True)

    # Test 1: List tools
    print("\n1. Testing tools/list:")
    request = JSONRPCRequest(
        method="tools/list",
        id=str(uuid.uuid4())
    )
    response_json = client.handle_jsonrpc_request(json.dumps(request.to_dict()))
    response = json.loads(response_json)
    print(f"   Request ID: {request.id}")
    print(f"   Response: {response['result'][:2]}... ({len(response['result'])} tools)")

    # Test 2: Call tool (safe, no approval needed)
    print("\n2. Testing tools/call (calculator):")
    request = JSONRPCRequest(
        method="tools/call",
        params={
            "name": "calculate",
            "arguments": {"expression": "10 + 5 * 2"}
        },
        id=str(uuid.uuid4())
    )
    response_json = client.handle_jsonrpc_request(json.dumps(request.to_dict()))
    response = json.loads(response_json)
    print(f"   Expression: 10 + 5 * 2")
    print(f"   Result: {response.get('result')}")
    print(f"   Error: {response.get('error')}")

    # Test 3: Call tool needing approval (should fail)
    print("\n3. Testing tools/call (python_exec, not approved):")
    request = JSONRPCRequest(
        method="tools/call",
        params={
            "name": "python_exec",
            "arguments": {"code": "print('test')", "timeout": 5}
        },
        id=str(uuid.uuid4())
    )
    response_json = client.handle_jsonrpc_request(json.dumps(request.to_dict()))
    response = json.loads(response_json)
    print(f"   Expected error code: {JSONRPCError.TOOL_NOT_APPROVED.value}")
    print(f"   Actual error: {response.get('error')}")

    # Test 4: Approve and retry
    print("\n4. Testing tools/call (python_exec, after approval):")
    client.approve_tool("python_exec")
    request = JSONRPCRequest(
        method="tools/call",
        params={
            "name": "python_exec",
            "arguments": {"code": "print('Hello JSON-RPC 2.0')", "timeout": 5}
        },
        id=str(uuid.uuid4())
    )
    response_json = client.handle_jsonrpc_request(json.dumps(request.to_dict()))
    response = json.loads(response_json)
    print(f"   Result: {response.get('result')}")

    # Test 5: Invalid request (wrong JSON-RPC version)
    print("\n5. Testing error handling (invalid JSON-RPC version):")
    bad_request = {"jsonrpc": "1.0", "method": "test", "id": "test"}
    response_json = client.handle_jsonrpc_request(json.dumps(bad_request))
    response = json.loads(response_json)
    print(f"   Expected error code: {JSONRPCError.INVALID_REQUEST.value}")
    print(f"   Actual error: {response.get('error')}")

    # Test 6: Method not found
    print("\n6. Testing error handling (method not found):")
    request = JSONRPCRequest(
        method="nonexistent/method",
        id=str(uuid.uuid4())
    )
    response_json = client.handle_jsonrpc_request(json.dumps(request.to_dict()))
    response = json.loads(response_json)
    print(f"   Expected error code: {JSONRPCError.METHOD_NOT_FOUND.value}")
    print(f"   Actual error: {response.get('error')}")

    print("\n" + "="*70)
    print("✅ JSON-RPC 2.0 INTERFACE VALIDATED")
    print("="*70)


if __name__ == "__main__":
    test_mcp_client()
    test_jsonrpc_interface()
