# Building Agents with Databricks Apps: A Deep Technical Guide

This document provides an in-depth technical analysis of how to build, test, deploy, and invoke AI agents using Databricks Apps, based on the templates in this repository. Where the underlying technology is not explicitly shown in the code, I offer hypotheses clearly marked as **[Hypothesis]**.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Writing Agent Code: Frameworks & Patterns](#2-writing-agent-code-frameworks--patterns)
   - [LangGraph Agents](#21-langgraph-agents)
   - [OpenAI Agents SDK](#22-openai-agents-sdk)
   - [Non-Conversational Agents](#23-non-conversational-agents)
   - [Migration from Model Serving](#24-migration-from-model-serving)
3. [Agent Powerfulness: From Simple to Advanced](#3-agent-powerfulness-from-simple-to-advanced)
   - [Simple Chat](#31-simple-chat)
   - [Tool-Using Agents](#32-tool-using-agents)
   - [Memory-Equipped Agents](#33-memory-equipped-agents)
   - [Multi-Agent Orchestration](#34-multi-agent-orchestration)
4. [Deep Dive: Tools & MCP](#4-deep-dive-tools--mcp)
   - [What is MCP?](#41-what-is-mcp)
   - [SDK & Libraries for Tools](#42-sdk--libraries-for-tools)
   - [Transport Mechanism](#43-transport-mechanism)
   - [Authentication Between Agent and Tools](#44-authentication-between-agent-and-tools)
   - [Building Custom MCP Servers](#45-building-custom-mcp-servers)
   - [Tool Discovery](#46-tool-discovery)
5. [Deep Dive: Memory](#5-deep-dive-memory)
   - [Memory Types & Lifecycle](#51-memory-types--lifecycle)
   - [Short-Term Memory (Conversation History)](#52-short-term-memory-conversation-history)
   - [Long-Term Memory (User-Scoped Persistent)](#53-long-term-memory-user-scoped-persistent)
   - [Holistic Agent Example: All Memory Types Combined](#54-holistic-agent-example-all-memory-types-combined)
   - [Underlying Technology: Lakebase](#55-underlying-technology-lakebase)
6. [Testing Agents Locally](#6-testing-agents-locally)
7. [Deploying Agents](#7-deploying-agents)
8. [Invoking Agents](#8-invoking-agents)
9. [Observability: MLflow Tracing](#9-observability-mlflow-tracing)
10. [Other Important Dimensions](#10-other-important-dimensions)

---

## 1. Architecture Overview

Every agent in this repository follows a consistent three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                   Frontend (Chat UI)                        │
│  e2e-chatbot-app-next (Next.js) or streamlit/gradio/dash   │
│                                                             │
│  POST /invocations  ←→  Agent Server (port 8000)           │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP (FastAPI + Uvicorn)
┌────────────────────────────▼────────────────────────────────┐
│               MLflow GenAI Agent Server                     │
│  @invoke() → synchronous one-shot response                  │
│  @stream() → async generator of streaming events            │
│                                                             │
│  Framework: LangGraph │ OpenAI Agents SDK │ Raw OpenAI      │
└────────────────────────────┬────────────────────────────────┘
                             │ MCP Protocol (HTTP + SSE)
┌────────────────────────────▼────────────────────────────────┐
│                   Tool Servers (MCP)                        │
│  system.ai (code exec) │ Genie (data) │ Custom MCP servers  │
│                                                             │
│  + Memory Store (Lakebase) for short/long-term memory       │
└─────────────────────────────────────────────────────────────┘
```

**Key technology choices:**
- **Server framework**: MLflow `AgentServer` wrapping FastAPI + Uvicorn
- **Agent protocol**: MLflow Responses API (`ResponsesAgentRequest` / `ResponsesAgentResponse`)
- **Tool protocol**: MCP (Model Context Protocol) over HTTP
- **LLM access**: Databricks-hosted models (e.g., `databricks-gpt-5-2`, `databricks-claude-sonnet-4-5`)
- **Tracing**: MLflow autologging
- **Deployment**: Databricks Asset Bundles (DAB) → Databricks Apps

---

## 2. Writing Agent Code: Frameworks & Patterns

### 2.1 LangGraph Agents

**Template**: [`agent-langgraph/`](../agent-langgraph/)

LangGraph builds agents as directed graphs where nodes are processing steps and edges define control flow. The `create_agent()` helper from LangChain handles the graph construction.

**Core agent setup** ([`agent-langgraph/agent_server/agent.py`](../agent-langgraph/agent_server/agent.py)):

```python
from langchain.agents import create_agent
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
import mlflow

mlflow.langchain.autolog()  # Auto-trace all LangChain operations

# Module-level service principal client (ambient credentials)
sp_workspace_client = WorkspaceClient()

def init_mcp_client(workspace_client: WorkspaceClient) -> DatabricksMultiServerMCPClient:
    host_name = get_databricks_host_from_env()
    return DatabricksMultiServerMCPClient([
        DatabricksMCPServer(
            name="system-ai",
            url=f"{host_name}/api/2.0/mcp/functions/system/ai",
            workspace_client=workspace_client,
        ),
    ])

async def init_agent(workspace_client=None):
    mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    tools = await mcp_client.get_tools()
    return create_agent(tools=tools, model=ChatDatabricks(endpoint="databricks-gpt-5-2"))
```

**How the agent graph works internally** — **[Hypothesis]**: `create_agent()` constructs a `StateGraph` with at least two nodes:
1. **LLM node**: Calls `ChatDatabricks` to generate a response or tool call
2. **Tool node**: Executes tool calls and returns results

The edges form a loop: LLM → (if tool call) → Tool → LLM → (if done) → END. The state carries the full message history between nodes.

**Streaming endpoint** — The `@stream()` decorator registers an async generator with MLflow's `AgentServer`:

```python
@stream()
async def streaming(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    agent = await init_agent()
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}
    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
```

The dual `stream_mode=["updates", "messages"]` gives:
- **"updates"**: State changes at each node (tool results, state transitions)
- **"messages"**: Token-by-token LLM output chunks

**Synchronous endpoint** — `@invoke()` collects all streaming events:

```python
@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in streaming(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)
```

**Key dependencies** (`pyproject.toml`):
| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | ≥1.0.1 | Graph-based agent orchestration |
| `databricks-langchain` | ≥0.14.0 | LLM client + MCP integration |
| `langchain-mcp-adapters` | ≥0.1.11 | Converts MCP tools to LangChain tools |
| `mlflow` | ≥3.9.0 | Agent server, tracing, experiment tracking |

---

### 2.2 OpenAI Agents SDK

**Template**: [`agent-openai-agents-sdk/`](../agent-openai-agents-sdk/)

The OpenAI Agents SDK provides a simpler, more opinionated API. Instead of building graphs, you declare agents with instructions and tools, then run them.

**Core agent setup** ([`agent-openai-agents-sdk/agent_server/agent.py`](../agent-openai-agents-sdk/agent_server/agent.py)):

```python
from agents import Agent, Runner
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from agents import set_default_openai_client, set_default_openai_api, set_trace_processors

# Configure Databricks as the LLM backend
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])  # MLflow handles tracing
mlflow.openai.autolog()

async def init_mcp_server(workspace_client=None):
    return McpServer(
        url=build_mcp_url("/api/2.0/mcp/functions/system/ai", workspace_client=workspace_client),
        name="system.ai UC function MCP server",
        workspace_client=workspace_client,
    )

def create_coding_agent(mcp_server: McpServer) -> Agent:
    return Agent(
        name="Code execution agent",
        instructions="You are a code execution agent. You can execute code and return the results.",
        model="databricks-gpt-5-2",
        mcp_servers=[mcp_server],
    )
```

**Streaming with the OpenAI SDK** uses `Runner.run_streamed()`:

```python
@stream()
async def stream_handler(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    workspace_client = WorkspaceClient()
    async with await init_mcp_server(workspace_client) as mcp_server:
        agent = create_coding_agent(mcp_server)
        messages = [i.model_dump() for i in request.input]
        result = Runner.run_streamed(agent, input=messages)
        async for event in process_agent_stream_events(result.stream_events()):
            yield event
```

**Key difference from LangGraph**: The MCP server is used as an async context manager (`async with`), which handles the MCP session lifecycle (connect, list tools, disconnect).

**How `AsyncDatabricksOpenAI` works** — **[Hypothesis]**: This is a wrapper around the standard OpenAI async client that redirects API calls to Databricks Foundation Model endpoints. It likely intercepts `chat.completions.create()` calls and routes them to `https://{workspace}/serving-endpoints/{model}/invocations`, injecting Databricks authentication headers.

---

### 2.3 Non-Conversational Agents

**Template**: [`agent-non-conversational/`](../agent-non-conversational/)

Not all agents are chatbots. This template shows a document analysis agent that takes structured input and returns structured output.

```python
# agent-non-conversational/agent_server/agent.py
from openai import OpenAI

openai_client = OpenAI()  # Uses Databricks-hosted model

@invoke()
async def invoke(data: dict) -> dict:
    input_data = AgentInput(**data)  # document_text + list of yes/no questions
    analysis_results = []
    for question in input_data.questions:
        prompt = construct_analysis_prompt(question, input_data.document_text)
        llm_response = openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "databricks-gpt-5-2"),
            messages=[{"role": "user", "content": prompt}]
        )
        result = parse_response(llm_response.choices[0].message.content)
        analysis_results.append(result)
    return AgentOutput(results=analysis_results).model_dump()
```

**Key takeaway**: The `@invoke()` decorator works with any input/output shape, not just `ResponsesAgentRequest`/`ResponsesAgentResponse`. This makes it flexible for batch processing, document analysis, or any non-chat use case.

---

### 2.4 Migration from Model Serving

**Template**: [`agent-migration-from-model-serving/`](../agent-migration-from-model-serving/)

This template provides a scaffold for migrating agents from Databricks Model Serving to Databricks Apps. The key change is moving from `predict()`/`predict_stream()` methods to `@invoke()`/`@stream()` decorators:

```python
# Scaffold - replace with your migrated agent logic
@stream()
async def streaming(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    raise NotImplementedError(
        "Replace this with your migrated agent's streaming implementation."
    )
```

---

## 3. Agent Powerfulness: From Simple to Advanced

### 3.1 Simple Chat

The simplest agent is just an LLM with no tools — a direct pass-through to the model:

```python
# Conceptual simple agent (no template exists for this, but easy to build)
from agents import Agent, Runner

agent = Agent(
    name="Simple chatbot",
    instructions="You are a helpful assistant.",
    model="databricks-gpt-5-2",
    # No tools, no MCP servers
)

@invoke()
async def handle(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    result = await Runner.run(agent, [i.model_dump() for i in request.input])
    return ResponsesAgentResponse(output=result.new_items)
```

### 3.2 Tool-Using Agents

Adding tools gives agents the ability to execute code, query data, and interact with external systems. See [Section 4](#4-deep-dive-tools--mcp) for full technical details.

### 3.3 Memory-Equipped Agents

Memory gives agents context across messages (short-term) and across conversations (long-term). See [Section 5](#5-deep-dive-memory) for full technical details.

### 3.4 Multi-Agent Orchestration

**Template**: [`agent-openai-agents-sdk-multiagent/`](../agent-openai-agents-sdk-multiagent/)

The multi-agent pattern uses an **orchestrator agent** that routes requests to specialized sub-agents:

```python
# agent-openai-agents-sdk-multiagent/agent_server/agent.py

SUBAGENTS = [
    {
        "name": "genie",
        "type": "genie",                    # Databricks Genie (data analysis)
        "space_id": "<SPACE-UUID>",
        "description": "Query structured data..."
    },
    {
        "name": "app_agent",
        "type": "app",                       # Another Databricks App
        "endpoint": "<APP-NAME>",
        "description": "Query a specialist agent..."
    },
    {
        "name": "serving_endpoint",
        "type": "serving_endpoint",          # Model Serving endpoint
        "endpoint": "<ENDPOINT>",
        "description": "Query a hosted model..."
    }
]
```

**Three integration mechanisms for sub-agents:**

| Type | Mechanism | URL Pattern |
|------|-----------|-------------|
| `genie` | MCP Server | `/api/2.0/mcp/genie/{space_id}` |
| `app` | Responses API | `apps/{endpoint}` via `DatabricksOpenAI` |
| `serving_endpoint` | Responses API | `{endpoint}` via `DatabricksOpenAI` |

**Sub-agent tool creation** — Non-Genie sub-agents are wrapped as `function_tool`s:

```python
def _make_subagent_tool(subagent: dict):
    endpoint = subagent["endpoint"]
    model = f"apps/{endpoint}" if subagent["type"] == "app" else endpoint

    async def _call(question: str) -> str:
        response = await _tool_client.responses.create(
            model=model,
            input=[{"role": "user", "content": question}],
        )
        return response.output_text

    _call.__name__ = f"query_{subagent['name']}"
    _call.__doc__ = subagent["description"]
    return function_tool(_call)
```

**Orchestrator agent** combines MCP tools (for Genie) and function tools (for other sub-agents):

```python
def create_orchestrator_agent(mcp_server: McpServer) -> Agent:
    return Agent(
        name="Orchestrator",
        instructions="Route the user's request to the most appropriate tool or data source...",
        model="databricks-claude-sonnet-4-5",
        mcp_servers=[mcp_server] if mcp_server else [],
        tools=subagent_tools,  # function_tool wrappers for app/serving agents
    )
```

---

## 4. Deep Dive: Tools & MCP

### 4.1 What is MCP?

**Model Context Protocol (MCP)** is an open standard (created by Anthropic) that defines how AI models interact with external tools and data sources. Think of it as "USB-C for AI" — a universal interface between agents and capabilities.

MCP defines a client-server architecture:
- **MCP Client** (inside the agent): Discovers tools, calls them, processes results
- **MCP Server** (external service): Exposes tools with schemas, executes requests

### 4.2 SDK & Libraries for Tools

Different SDKs are used depending on the agent framework:

**For LangGraph agents** — `databricks-langchain`:

```python
from databricks_langchain import DatabricksMCPServer, DatabricksMultiServerMCPClient

# Create client that connects to multiple MCP servers
client = DatabricksMultiServerMCPClient([
    DatabricksMCPServer(
        name="system-ai",
        url=f"{host}/api/2.0/mcp/functions/system/ai",
        workspace_client=workspace_client,
    ),
])

# Fetch tools — returns LangChain-compatible Tool objects
tools = await client.get_tools()
# These tools are passed directly to create_agent(tools=tools, ...)
```

**[Hypothesis]**: `DatabricksMultiServerMCPClient` likely wraps `langchain-mcp-adapters`, which converts MCP tool definitions into LangChain `BaseTool` objects. Each tool's `_arun()` method sends an MCP `tools/call` request to the server.

**For OpenAI Agents SDK** — `databricks-openai`:

```python
from databricks_openai.agents import McpServer

# Create MCP server reference
mcp_server = McpServer(
    url=build_mcp_url("/api/2.0/mcp/functions/system/ai"),
    name="system.ai UC function MCP server",
    workspace_client=workspace_client,
)

# Used as async context manager (handles session lifecycle)
async with await init_mcp_server() as mcp_server:
    agent = Agent(mcp_servers=[mcp_server], ...)
    result = await Runner.run(agent, messages)
```

**For building MCP servers** — `fastmcp`:

```python
from fastmcp import FastMCP

mcp_server = FastMCP(name="custom-mcp-server")

@mcp_server.tool
def my_tool(param: str) -> dict:
    """Tool description used by the AI"""
    return {"result": do_something(param)}
```

### 4.3 Transport Mechanism

MCP in this repository uses **HTTP with Streamable HTTP transport** (previously SSE — Server-Sent Events):

```
Agent (MCP Client)                    MCP Server
       │                                   │
       │─── POST /mcp ──────────────────►  │  (initialize)
       │◄── 200 OK + session_id ────────   │
       │                                   │
       │─── POST /mcp ──────────────────►  │  (tools/list)
       │◄── 200 OK + tool schemas ──────   │
       │                                   │
       │─── POST /mcp ──────────────────►  │  (tools/call)
       │◄── 200 OK + result ────────────   │  (or SSE stream for long ops)
       │                                   │
```

**[Hypothesis]**: The FastMCP `http_app()` method creates a FastAPI/Starlette app that serves the MCP protocol at `/mcp`. The transport likely uses JSON-RPC 2.0 messages over HTTP POST, with optional SSE for streaming responses. The `DatabricksMCPServer` and `McpServer` classes act as HTTP clients to this endpoint.

**Endpoint URL patterns for different tool types:**

```
System AI tools:     https://{workspace}/api/2.0/mcp/functions/system/ai
UC Functions:        https://{workspace}/api/2.0/mcp/functions/{catalog}/{schema}
Genie Spaces:        https://{workspace}/api/2.0/mcp/genie/{space_id}
Vector Search:       https://{workspace}/api/2.0/mcp/vector-search/{catalog}/{schema}
External MCP:        https://{workspace}/api/2.0/mcp/external/{connection_name}
Custom MCP Apps:     https://{app_url}/mcp
```

### 4.4 Authentication Between Agent and Tools

Authentication flows through two mechanisms:

#### Service Principal Authentication (Default)

When the agent server starts, it creates a module-level `WorkspaceClient()` using ambient credentials:

```python
# agent_server/agent.py
sp_workspace_client = WorkspaceClient()  # Uses DATABRICKS_HOST + DATABRICKS_TOKEN or OAuth
```

This client is passed to the MCP server constructor, which uses it to authenticate requests:

```python
DatabricksMCPServer(
    workspace_client=workspace_client,  # Auth is embedded in the client
    url=f"{host}/api/2.0/mcp/functions/system/ai",
)
```

**[Hypothesis]**: The `DatabricksMCPServer` likely injects `Authorization: Bearer <token>` headers into every MCP HTTP request. The token comes from `workspace_client.config.token` or is refreshed via OAuth.

#### On-Behalf-Of User Authentication

For user-scoped operations, the agent extracts the user's token from request headers:

```python
# agent_server/utils.py
def get_user_workspace_client() -> WorkspaceClient:
    token = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")
```

**The full auth flow:**

```
User (Browser)                    Databricks Apps Proxy              Agent Server
      │                                   │                              │
      │── Request + OAuth cookie ───────► │                              │
      │                                   │─── Request + headers ──────► │
      │                                   │    x-forwarded-access-token  │
      │                                   │    x-forwarded-email         │
      │                                   │    x-forwarded-user          │
      │                                   │                              │
      │                                   │                              │── MCP call with
      │                                   │                              │   user's token
      │                                   │                              │──────► MCP Server
```

**[Hypothesis]**: The Databricks Apps proxy performs OAuth token exchange — it takes the user's session cookie, mints a short-lived access token, and injects it as `x-forwarded-access-token`. This token has the same permissions as the user, enabling on-behalf-of tool execution.

#### MCP Server Auth (Custom Servers)

Custom MCP servers deployed as Databricks Apps receive the same forwarded headers:

```python
# mcp-server-hello-world/server/utils.py
import contextvars

header_store = contextvars.ContextVar("header_store")  # Thread-safe header storage

def get_user_authenticated_workspace_client():
    headers = header_store.get({})
    token = headers.get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")
```

```python
# mcp-server-hello-world/server/app.py — middleware captures headers
@combined_app.middleware("http")
async def capture_headers(request: Request, call_next):
    header_store.set(dict(request.headers))
    return await call_next(request)
```

**`contextvars.ContextVar`** ensures thread safety — each async request gets its own context, so concurrent requests don't mix up auth tokens.

### 4.5 Building Custom MCP Servers

**Template**: [`mcp-server-hello-world/`](../mcp-server-hello-world/)

**Step 1: Define tools** ([`server/tools.py`](../mcp-server-hello-world/server/tools.py)):

```python
def load_tools(mcp_server):
    @mcp_server.tool
    def health() -> dict:
        """Check the health of the MCP server"""
        return {"status": "healthy", "message": "Custom MCP Server is healthy and connected."}

    @mcp_server.tool
    def get_current_user() -> dict:
        """Get information about the current authenticated user"""
        w = utils.get_user_authenticated_workspace_client()
        user = w.current_user.me()
        return {"display_name": user.display_name, "user_name": user.user_name}
```

**Step 2: Create the server app** ([`server/app.py`](../mcp-server-hello-world/server/app.py)):

```python
from fastmcp import FastMCP

mcp_server = FastMCP(name="custom-mcp-server")
load_tools(mcp_server)
mcp_app = mcp_server.http_app()  # MCP protocol at /mcp

# Combine with custom FastAPI routes
combined_app = FastAPI(
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)
```

**Step 3: Deploy** via `databricks.yml` and `app.yaml`:

```yaml
# app.yaml
command: ["uv", "run", "custom-mcp-server"]
```

**Step 4: Connect from an agent** by adding the MCP server to the agent's configuration:

```python
DatabricksMCPServer(
    name="my-custom-tools",
    url="https://<app-url>/mcp",
    workspace_client=workspace_client,
)
```

**OpenAPI-based MCP server** ([`mcp-server-open-api-spec/`](../mcp-server-open-api-spec/)): This template generates MCP tools from an OpenAPI spec, allowing any REST API to be exposed as MCP tools:

```python
@mcp_server.tool()
def list_api_endpoints(search_query: Optional[str] = None) -> Dict[str, Any]:
    """Discovers available API endpoints with optional filtering"""
    ...

@mcp_server.tool()
def invoke_api_endpoint(endpoint_path: str, http_method: str, parameters=None) -> Dict[str, Any]:
    """Executes an API endpoint with appropriate parameters"""
    ...
```

### 4.6 Tool Discovery

The repository includes a discovery script ([`agent-langgraph/scripts/discover_tools.py`](../agent-langgraph/scripts/discover_tools.py)) that enumerates all available tools in a workspace:

```python
results = {
    "uc_functions": discover_uc_functions(w),           # SQL UDFs in Unity Catalog
    "uc_tables": discover_uc_tables(w),                 # Tables as data sources
    "vector_search_indexes": discover_vector_search_indexes(w),  # RAG indexes
    "genie_spaces": discover_genie_spaces(w),           # NL-to-SQL interfaces
    "custom_mcp_servers": discover_custom_mcp_servers(w),  # Apps with mcp-* prefix
    "external_mcp_servers": discover_external_mcp_servers(w),  # UC connections
}
```

Custom MCP servers are discovered by convention — apps with names starting with `mcp-`:

```python
def discover_custom_mcp_servers(w: WorkspaceClient):
    apps = w.apps.list()
    return [{"name": app.name, "url": app.url} for app in apps if app.name.startswith("mcp-")]
```

---

## 5. Deep Dive: Memory

### 5.1 Memory Types & Lifecycle

The repository implements three types of memory, each with different scope and persistence:

| Memory Type | Scope | Persistence | SDK Class | Backend |
|-------------|-------|-------------|-----------|---------|
| **Short-term (LangGraph)** | Per conversation thread | Within thread | `AsyncCheckpointSaver` | Lakebase |
| **Short-term (OpenAI SDK)** | Per session | Within session | `AsyncDatabricksSession` | Lakebase |
| **Long-term** | Per user (cross-conversation) | Indefinite | `AsyncDatabricksStore` | Lakebase + Embeddings |

**Memory lifecycle diagram:**

```
                          SHORT-TERM MEMORY
                     (per thread / per session)
                     
  Request 1          Request 2          Request 3
  ┌──────┐          ┌──────┐          ┌──────┐
  │User: │          │User: │          │User: │
  │ Hi   │──save──► │What's│──save──► │ Bye  │──save──► ...
  │AI:   │          │ 2+2? │          │AI:   │
  │Hello!│          │AI: 4 │          │ Bye! │
  └──────┘          └──────┘          └──────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                thread_id="abc-123"
                (all messages visible to agent)

                          LONG-TERM MEMORY
                     (per user, across conversations)
                     
  Conv A (thread_id=1)          Conv B (thread_id=2)
  ┌─────────────────┐          ┌─────────────────┐
  │ User: I prefer  │          │ User: What do I │
  │  dark mode      │          │  like?          │
  │ Agent:          │          │ Agent:          │
  │  *saves memory* │──────────│  *searches*     │
  │  save_user_     │  same    │  get_user_      │
  │  memory(        │  user_id │  memory(        │
  │  "ui_pref",     │ ───────► │  "preferences") │
  │  {"dark_mode"}) │          │  → "dark mode"  │
  └─────────────────┘          └─────────────────┘
       user_id="alice@company.com"
```

### 5.2 Short-Term Memory (Conversation History)

#### LangGraph Pattern

**Template**: [`agent-langgraph-short-term-memory/`](../agent-langgraph-short-term-memory/)

**Thread ID resolution** — priority order ([`agent-langgraph-short-term-memory/agent_server/agent.py`](../agent-langgraph-short-term-memory/agent_server/agent.py)):

```python
def _get_or_create_thread_id(request: ResponsesAgentRequest) -> str:
    # 1. Explicit thread_id from custom_inputs
    ci = dict(request.custom_inputs or {})
    if "thread_id" in ci and ci["thread_id"]:
        return str(ci["thread_id"])
    # 2. conversation_id from MLflow ChatContext
    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)
    # 3. Generate a new UUID
    return str(uuid_utils.uuid7())
```

**Checkpoint-based memory** — The agent's full state (all messages) is saved after each invocation:

```python
from databricks_langchain import AsyncCheckpointSaver

LAKEBASE_INSTANCE_NAME = os.environ.get("LAKEBASE_INSTANCE_NAME", "")

@stream()
async def streaming(request: ResponsesAgentRequest):
    thread_id = _get_or_create_thread_id(request)
    
    async with AsyncCheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as checkpointer:
        agent = await init_agent(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        async for event in process_agent_astream_events(
            agent.astream(input_state, config, stream_mode=["updates", "messages"])
        ):
            yield event
```

**How `AsyncCheckpointSaver` works** — **[Hypothesis]**: When `agent.astream()` completes a node execution, LangGraph's checkpoint system calls `checkpointer.aput()` to persist the graph state (including all messages) to a Lakebase table. On the next invocation with the same `thread_id`, `checkpointer.aget()` restores the state, so the agent "remembers" the conversation. The checkpoint key is `(thread_id, checkpoint_ns, checkpoint_id)` where `checkpoint_ns` is typically empty and `checkpoint_id` is auto-incremented.

#### OpenAI SDK Pattern

**Template**: [`agent-openai-agents-sdk-short-term-memory/`](../agent-openai-agents-sdk-short-term-memory/)

```python
from databricks_openai.agents import AsyncDatabricksSession

session = AsyncDatabricksSession(
    session_id=get_session_id(request),  # equivalent to thread_id
    instance_name=LAKEBASE_INSTANCE_NAME,
)

async with await init_mcp_server() as mcp_server:
    agent = create_coding_agent(mcp_server)
    messages = await deduplicate_input(request, session)  # Remove already-seen messages
    result = await Runner.run(agent, messages, session=session)
    return ResponsesAgentResponse(
        output=sanitize_output_items(result.new_items),
        custom_outputs={"session_id": session.session_id},  # Return for client to reuse
    )
```

**[Hypothesis]**: `AsyncDatabricksSession` stores conversation turns in a Lakebase table keyed by `session_id`. The `deduplicate_input()` method compares incoming messages against stored history and only passes new messages to `Runner.run()`. The session ID is returned to the client so it can be sent back in subsequent requests.

### 5.3 Long-Term Memory (User-Scoped Persistent)

**Template**: [`agent-langgraph-long-term-memory/`](../agent-langgraph-long-term-memory/)

Long-term memory is fundamentally different from short-term — it's a **semantic key-value store** that the agent actively manages through tool calls.

**Memory store initialization** ([`agent-langgraph-long-term-memory/agent_server/agent.py`](../agent-langgraph-long-term-memory/agent_server/agent.py)):

```python
from databricks_langchain import AsyncDatabricksStore

EMBEDDING_ENDPOINT = "databricks-gte-large-en"
EMBEDDING_DIMS = 1024

async with AsyncDatabricksStore(
    instance_name=LAKEBASE_INSTANCE_NAME,
    embedding_endpoint=EMBEDDING_ENDPOINT,  # For semantic search
    embedding_dims=EMBEDDING_DIMS,          # 1024-dimensional vectors
) as store:
    await store.setup()  # Create tables if needed
    config = {"configurable": {"store": store, "user_id": user_id}}
```

**Memory tools** ([`agent-langgraph-long-term-memory/agent_server/utils_memory.py`](../agent-langgraph-long-term-memory/agent_server/utils_memory.py)):

```python
@tool
async def get_user_memory(query: str, config: RunnableConfig) -> str:
    """Search for relevant information about the user from long-term memory."""
    user_id = config.get("configurable", {}).get("user_id")
    store = config.get("configurable", {}).get("store")
    namespace = ("user_memories", user_id.replace(".", "-"))
    
    results = await store.asearch(namespace, query=query, limit=5)  # Semantic search!
    
    if not results:
        return "No memories found for this user."
    memory_items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
    return f"Found {len(results)} relevant memories:\n" + "\n".join(memory_items)

@tool
async def save_user_memory(memory_key: str, memory_data_json: str, config: RunnableConfig) -> str:
    """Save information about the user to long-term memory."""
    user_id = config.get("configurable", {}).get("user_id")
    store = config.get("configurable", {}).get("store")
    namespace = ("user_memories", user_id.replace(".", "-"))
    
    memory_data = json.loads(memory_data_json)
    await store.aput(namespace, memory_key, memory_data)
    return f"Successfully saved memory '{memory_key}' for user."

@tool
async def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
    """Delete a specific memory from the user's long-term memory."""
    user_id = config.get("configurable", {}).get("user_id")
    store = config.get("configurable", {}).get("store")
    namespace = ("user_memories", user_id.replace(".", "-"))
    
    await store.adelete(namespace, memory_key)
    return f"Successfully deleted memory '{memory_key}' for user."
```

**System prompt instructs the agent to use memory tools:**

```python
SYSTEM_PROMPT = """You are a helpful assistant. Use the available tools to answer questions.

You have access to memory tools that allow you to remember information about users:
- Use get_user_memory to search for previously saved information about the user
- Use save_user_memory to remember important facts, preferences, or details the user shares
- Use delete_user_memory to forget specific information when asked

Always check for relevant memories at the start of a conversation to provide personalized responses."""
```

**User ID extraction** — similar to thread ID, with priority order:

```python
def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    custom_inputs = dict(request.custom_inputs or {})
    if "user_id" in custom_inputs:
        return custom_inputs["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None
```

**How semantic search works** — **[Hypothesis]**: When `save_user_memory` is called, `AsyncDatabricksStore.aput()` stores the key-value pair AND computes an embedding of the value using `databricks-gte-large-en` (a Databricks-hosted embedding model). The embedding is stored alongside the data. When `get_user_memory` calls `store.asearch(namespace, query=query)`, the query is also embedded, and a vector similarity search (likely cosine similarity) finds the most relevant memories. This enables queries like "what does the user prefer?" to match memories stored as "ui_pref: dark mode" even though the words don't exactly match.

### 5.4 Holistic Agent Example: All Memory Types Combined

Below is a **hypothetical** comprehensive agent that combines all three memory types. This is not directly from the templates but synthesizes patterns from `agent-langgraph-short-term-memory` and `agent-langgraph-long-term-memory`:

```python
"""
Holistic memory agent combining:
1. Short-term memory (conversation history via checkpointer)
2. Long-term memory (user preferences via store + semantic search)
3. Working memory (ephemeral scratchpad via graph state) [Hypothesis]
"""
import mlflow
from langchain.agents import create_agent
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
    AsyncCheckpointSaver,
    AsyncDatabricksStore,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
import json

mlflow.langchain.autolog()

LAKEBASE_INSTANCE_NAME = "my-lakebase-instance"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
EMBEDDING_DIMS = 1024

# ─── LONG-TERM MEMORY TOOLS ───────────────────────────────────────────────

@tool
async def get_user_memory(query: str, config) -> str:
    """Search for relevant information about the user from long-term memory.
    
    Use this to recall user preferences, past decisions, or any previously
    saved facts about the user. The search is semantic — you don't need
    exact keywords.
    """
    user_id = config.get("configurable", {}).get("user_id")
    store = config.get("configurable", {}).get("store")
    if not user_id or not store:
        return "Long-term memory unavailable (no user_id or store)."
    
    namespace = ("user_memories", user_id.replace(".", "-"))
    results = await store.asearch(namespace, query=query, limit=5)
    
    if not results:
        return "No memories found."
    return "\n".join([f"- [{r.key}]: {json.dumps(r.value)}" for r in results])


@tool
async def save_user_memory(memory_key: str, memory_data_json: str, config) -> str:
    """Save a fact or preference about the user to long-term memory.
    
    Args:
        memory_key: Short identifier (e.g., "preferred_language", "team_name")
        memory_data_json: JSON object with the data to store
    """
    user_id = config.get("configurable", {}).get("user_id")
    store = config.get("configurable", {}).get("store")
    if not user_id or not store:
        return "Cannot save: long-term memory unavailable."
    
    namespace = ("user_memories", user_id.replace(".", "-"))
    memory_data = json.loads(memory_data_json)
    await store.aput(namespace, memory_key, memory_data)
    return f"Saved '{memory_key}' to long-term memory."


@tool
async def delete_user_memory(memory_key: str, config) -> str:
    """Delete a specific memory about the user."""
    user_id = config.get("configurable", {}).get("user_id")
    store = config.get("configurable", {}).get("store")
    if not user_id or not store:
        return "Cannot delete: long-term memory unavailable."
    
    namespace = ("user_memories", user_id.replace(".", "-"))
    await store.adelete(namespace, memory_key)
    return f"Deleted '{memory_key}' from long-term memory."


# ─── AGENT INITIALIZATION ─────────────────────────────────────────────────

async def init_agent(checkpointer, mcp_tools):
    """Create agent with MCP tools + memory tools."""
    all_tools = mcp_tools + [get_user_memory, save_user_memory, delete_user_memory]
    return create_agent(
        tools=all_tools,
        model=ChatDatabricks(endpoint="databricks-claude-sonnet-4-5"),
        checkpointer=checkpointer,  # Short-term memory: auto-saves conversation state
    )


# ─── STREAMING ENDPOINT ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a personalized assistant with memory capabilities.

MEMORY PROTOCOL:
1. At the START of every conversation, call get_user_memory("user context") to load
   relevant memories about the user.
2. When the user shares preferences, facts, or important decisions, proactively call
   save_user_memory() to remember them for future conversations.
3. When the user asks you to forget something, call delete_user_memory().

SHORT-TERM MEMORY: You automatically remember everything said in this conversation.
The thread_id keeps your conversation state. If the user returns to the same thread,
you'll have full context.

LONG-TERM MEMORY: Use the memory tools to persist facts across conversations.
These memories are tied to the user (not the thread), so they're available in
any conversation with the same user.
"""

@mlflow.trace
async def streaming(request):
    """
    Memory lifecycle per request:
    
    1. Extract thread_id → used for short-term checkpoint memory
    2. Extract user_id → used for long-term semantic memory
    3. Create checkpointer (AsyncCheckpointSaver) → manages conversation state
    4. Create store (AsyncDatabricksStore) → manages user memories
    5. Run agent with config containing both IDs and store reference
    6. Checkpointer auto-saves state after each graph node execution
    7. Memory tools explicitly save/search/delete in store when called by LLM
    """
    thread_id = _get_or_create_thread_id(request)
    user_id = _get_user_id(request)
    
    # Initialize MCP tools for external capabilities
    mcp_client = DatabricksMultiServerMCPClient([...])
    mcp_tools = await mcp_client.get_tools()
    
    # SHORT-TERM: Conversation state persisted per thread
    async with AsyncCheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME) as checkpointer:
        
        # LONG-TERM: User memories with semantic search
        async with AsyncDatabricksStore(
            instance_name=LAKEBASE_INSTANCE_NAME,
            embedding_endpoint=EMBEDDING_ENDPOINT,
            embedding_dims=EMBEDDING_DIMS,
        ) as store:
            await store.setup()
            
            agent = await init_agent(checkpointer, mcp_tools)
            
            config = {
                "configurable": {
                    "thread_id": thread_id,   # Short-term memory scope
                    "user_id": user_id,       # Long-term memory scope
                    "store": store,           # Store reference for memory tools
                }
            }
            
            messages = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages}
            
            async for event in agent.astream(messages, config, stream_mode=["updates", "messages"]):
                yield transform_event(event)
```

**Memory scope and ID summary:**

| Memory Type | ID | Scope | Created By | Example |
|-------------|-----|-------|-----------|---------|
| **Short-term** | `thread_id` | Single conversation | Auto-generated (uuid7) or client-provided | `"abc-123-def"` |
| **Long-term** | `user_id` | All conversations for one user | Extracted from request context | `"alice@company.com"` |
| **Working** | N/A | Single request | Graph state (in-memory) | Intermediate tool results |
| **Namespace** | `("user_memories", user_id)` | Partitions within store | Hardcoded in memory tools | `("user_memories", "alice-company-com")` |

### 5.5 Underlying Technology: Lakebase

All memory types are backed by **Lakebase**, Databricks' serverless SQL database:

**[Hypothesis]**: Lakebase is likely a managed PostgreSQL-compatible or proprietary serverless database optimized for Databricks workloads. Key evidence:
- The `e2e-chatbot-app-next` README mentions "Postgres" and "Lakebase" together as database options
- `AsyncCheckpointSaver` and `AsyncDatabricksStore` both take an `instance_name` parameter
- The `resolve_lakebase_instance_name()` utility resolves hostnames to instance names via the workspace API

```python
# utils_memory.py — Lakebase instance resolution
def resolve_lakebase_instance_name(instance_name: str) -> str:
    if not _is_lakebase_hostname(instance_name):
        return instance_name  # Already a name
    
    client = WorkspaceClient()
    instances = list(client.database.list_database_instances())
    for instance in instances:
        if hostname in (instance.read_write_dns, instance.read_only_dns):
            return instance.name
    raise ValueError(f"No Lakebase instance found for hostname: {instance_name}")
```

**[Hypothesis]**: Under the hood, Lakebase likely stores:

| Memory Type | Hypothesized Table Structure |
|-------------|------------------------------|
| **Checkpoints** | `checkpoints(thread_id, checkpoint_ns, checkpoint_id, data JSONB, created_at)` |
| **Sessions** | `sessions(session_id, messages JSONB[], updated_at)` |
| **User Store** | `store(namespace TEXT[], key TEXT, value JSONB, embedding VECTOR(1024), updated_at)` |

The embedding column enables vector similarity search via `pgvector`-style `<->` operators (cosine distance).

---

## 6. Testing Agents Locally

### Quickstart Setup

Every agent template includes a quickstart script:

```bash
uv run quickstart  # Interactive setup
```

This automates ([`scripts/quickstart.py`](../agent-langgraph/scripts/quickstart.py)):
1. **Prerequisites check**: Verifies `uv`, `node`, `npm`, `databricks` CLI
2. **Authentication**: Sets up Databricks OAuth profile
3. **MLflow experiment**: Creates per-user experiment at `/Users/{username}/agents-on-apps`
4. **Lakebase** (optional): Configures memory database
5. **`.env` file**: Writes configuration

### Running the Agent Locally

```bash
# Start agent server only (port 8000)
uv run start-server

# Start agent server + chat UI (port 3000 + 8000)
uv run start-app
```

The startup sequence ([`scripts/start_app.py`](../agent-langgraph/scripts/start_app.py)):
1. Loads `.env` for local configuration
2. Configures MLflow tracking
3. Starts the agent server (FastAPI + Uvicorn on port 8000)
4. Starts the chat frontend (Next.js on port 3000)
5. Waits for both to be ready

### Testing MCP Servers

```bash
cd mcp-server-hello-world
uv run pytest tests/
```

Integration tests ([`tests/test_integration_server.py`](../mcp-server-hello-world/tests/test_integration_server.py)):

```python
@pytest.fixture(scope="session")
def run_mcp_server():
    port = _find_free_port()
    proc = subprocess.Popen(shlex.split(f"uv run custom-mcp-server --port {port}"))
    _wait_for_server_startup(f"http://127.0.0.1:{port}")
    yield f"http://127.0.0.1:{port}"
    os.killpg(proc.pid, signal.SIGTERM)

def test_list_tools(run_mcp_server):
    client = DatabricksMCPClient(server_url=f"{run_mcp_server}/mcp")
    tools = client.list_tools()
    assert len(tools) > 0
```

### Testing Agent Endpoints

```python
# agent-non-conversational/test_agent.py
def test_agent(base_url="http://localhost:8000"):
    data = {
        "document_text": "Total assets: $2,300,000...",
        "questions": ["Do documents contain a balance sheet?"]
    }
    response = requests.post(f"{base_url}/invocations", json=data)
    assert response.status_code == 200
    assert len(response.json()["results"]) > 0
```

---

## 7. Deploying Agents

### Databricks Asset Bundles (DAB)

Every agent is deployed using Databricks Asset Bundles, configured in `databricks.yml`:

```yaml
# databricks.yml
bundle:
  name: agent_langgraph

resources:
  experiments:
    agent_langgraph_experiment:
      name: /Users/${workspace.current_user.userName}/${bundle.name}-${bundle.target}

  apps:
    agent_langgraph:
      name: "${bundle.target}-agent-langgraph"
      description: "LangGraph agent application"
      source_code_path: ./
      resources:
        - name: 'experiment'
          experiment:
            experiment_id: "${resources.experiments.agent_langgraph_experiment.id}"
            permission: 'CAN_MANAGE'

targets:
  dev:
    mode: development
    default: true
  prod:
    mode: production
    resources:
      apps:
        agent_langgraph:
          name: agent-langgraph  # Shorter name for production
```

### Deployment Commands

```bash
# Validate the bundle configuration
databricks bundle validate

# Deploy to development (default target)
databricks bundle deploy

# Deploy to production
databricks bundle deploy --target prod

# Deploy and run the app
databricks bundle deploy && databricks bundle run agent_langgraph
```

### Runtime Configuration (`app.yaml`)

```yaml
command: ["uv", "run", "start-app"]
env:
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: MLFLOW_REGISTRY_URI
    value: "databricks-uc"
  - name: API_PROXY
    value: "http://localhost:8000/invocations"
  - name: MLFLOW_EXPERIMENT_ID
    valueFrom: experiment
```

**How deployment works** — **[Hypothesis]**: When you run `databricks bundle deploy`:
1. DAB packages the source code (`source_code_path: ./`)
2. Creates an MLflow experiment in the workspace
3. Registers a Databricks App with the specified name
4. Uploads the code to the app's storage
5. The app starts with the specified `command` and `env` variables
6. Databricks Apps proxy handles SSL, authentication, and routing

---

## 8. Invoking Agents

### HTTP API (Direct)

```bash
# Invoke (synchronous)
curl -X POST https://<app-url>/invocations \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"role": "user", "content": "Hello, what can you do?"}]
  }'

# Stream (Server-Sent Events)
curl -X POST https://<app-url>/invocations/stream \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"role": "user", "content": "Write a Python hello world"}]
  }'
```

### Databricks OpenAI SDK (Programmatic)

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI()

# Invoke via Responses API (apps/ prefix)
response = client.responses.create(
    model="apps/my-agent-app",
    input=[{"role": "user", "content": "What's 2+2?"}]
)
print(response.output_text)

# Invoke via Model Serving endpoint
response = client.responses.create(
    model="my-serving-endpoint",
    input=[{"role": "user", "content": "Analyze this data..."}]
)
```

### With Memory (Custom Inputs)

```python
# Short-term memory — resume a conversation thread
response = client.responses.create(
    model="apps/my-agent-app",
    input=[{"role": "user", "content": "What did I just ask you?"}],
    extra_body={
        "custom_inputs": {"thread_id": "abc-123"}
    }
)

# Long-term memory — identify the user
response = client.responses.create(
    model="apps/my-agent-app",
    input=[{"role": "user", "content": "What are my preferences?"}],
    extra_body={
        "custom_inputs": {"user_id": "alice@company.com", "thread_id": "new-thread"}
    }
)
```

---

## 9. Observability: MLflow Tracing

Every agent template includes automatic tracing:

```python
# LangGraph agents
mlflow.langchain.autolog()

# OpenAI Agents SDK
mlflow.openai.autolog()
```

**What gets traced:**
- Every LLM call (model, prompt, response, latency, tokens)
- Every tool invocation (tool name, input, output, latency)
- Agent-level spans (full request → response)
- Streaming events (chunked output)

**Experiment tracking** via `databricks.yml`:

```yaml
resources:
  experiments:
    agent_langgraph_experiment:
      name: /Users/${workspace.current_user.userName}/${bundle.name}-${bundle.target}
```

Traces appear in the MLflow Experiment UI in the Databricks workspace, giving you:
- Request/response logs for every invocation
- Tool call chains and their results
- Latency breakdown per node
- Error tracking and debugging

---

## 10. Other Important Dimensions

### 10.1 Frontend Options

The repository provides templates for 8+ UI frameworks:

| Framework | Template | Best For |
|-----------|----------|----------|
| **Next.js** | `e2e-chatbot-app-next/` | Production chat apps with persistent history |
| **Streamlit** | `streamlit-chatbot-app/` | Quick prototypes, data apps |
| **Gradio** | `gradio-chatbot-app/` | ML demos, interactive interfaces |
| **Dash** | `dash-chatbot-app/` | Data dashboards with chat |
| **Shiny** | `shiny-chatbot-app/` | R-ecosystem integration |
| **Flask** | `flask-hello-world-app/` | Custom web apps |

### 10.2 On-Behalf-Of (OBO) User Pattern

Templates with `-obo-user` suffix (e.g., `streamlit-data-app-obo-user`) execute operations with the calling user's permissions instead of the service principal:

```python
# streamlit-data-app-obo-user pattern
headers = st.context.headers
token = headers.get("X-Forwarded-Access-Token")
workspace_client = WorkspaceClient(token=token, auth_type="pat")
# All subsequent API calls use the user's permissions
```

### 10.3 Security Considerations

- **Token forwarding**: `x-forwarded-access-token` is a short-lived token injected by the Databricks Apps proxy. **[Hypothesis]**: It's likely a PAT-equivalent token scoped to the authenticated user.
- **Namespace sanitization**: User IDs are sanitized (`user_id.replace(".", "-")`) to prevent namespace injection.
- **Context isolation**: `contextvars.ContextVar` ensures concurrent requests don't share auth state.
- **Service principal vs. user**: Agents default to service principal credentials but can upgrade to user credentials for sensitive operations.

### 10.4 Package Management

All Python templates use `uv` for fast, reproducible dependency management:

```bash
uv run start-server    # Runs with locked dependencies
uv run quickstart      # Setup automation
uv run pytest          # Testing
```

**`pyproject.toml`** defines scripts, dependencies, and Python version constraints (≥3.11).

### 10.5 Summary: Choosing Your Architecture

| If you need... | Use this template | Framework |
|----------------|-------------------|-----------|
| Simple tool-using agent | `agent-openai-agents-sdk/` | OpenAI Agents SDK |
| Graph-based control flow | `agent-langgraph/` | LangGraph |
| Conversation memory | `agent-*-short-term-memory/` | Either |
| Personalized user memory | `agent-langgraph-long-term-memory/` | LangGraph |
| Multi-agent routing | `agent-openai-agents-sdk-multiagent/` | OpenAI Agents SDK |
| Document analysis (non-chat) | `agent-non-conversational/` | Raw OpenAI |
| Custom tools | `mcp-server-hello-world/` | FastMCP |
| API-to-tool bridge | `mcp-server-open-api-spec/` | FastMCP + OpenAPI |
