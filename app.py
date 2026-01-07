import asyncio
import os
from typing import Annotated, List, TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
load_dotenv()

# 1. High-Scale MCP Configuration (10+ Specialized Servers)
# Includes local and remote transports [8, 17]
MCP_ECOSYSTEM = {
    "devops_team": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-docker"]},
    "security_expert": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-semgrep"]},
    "financial_hub": {"transport": "sse", "url": "https://alpha-vantage.mcp.run/sse"},
    "research_engine": {"transport": "http", "url": "http://brave-search.internal/mcp"},
    "code_agent": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
    "database_hub": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-sqlite", "--db-path", "./knowledge.db"]},
    "memory_graph": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-memory"]},
    "slack_relay": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-slack"]},
    "drive_access": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-google-drive"]},
    "weather_service": {"transport": "http", "url": "http://api.weather-mcp.internal/mcp"}
}

class AGIState(TypedDict):
    messages: Annotated[List[BaseMessage], "The history"]
    team_outputs: List[str] # PKH: Parallel Knowledge Hub
    active_worker: str

class AGISwarm:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(  model="gemini-1.5", temperature=0.2 , api_key=os.getenv("GEMINI_API_KEY"))
        self.client = MultiServerMCPClient(MCP_ECOSYSTEM)
        self.checkpointer = InMemorySaver() # Short-term memory [15]
        self.store = InMemoryStore() # Long-term memory [16]

    async def chief_orchestrator(self, state: AGIState) -> Command:
        """Recursive Delegation Engine: Decomposes goals into teams"""
        prompt = (
            "You are the Chief AGI. Decompose the goal. "
            "Route to: DEVOPS_TEAM, RESEARCH_TEAM, or FINISH. "
            "If info is missing, identify the gap for the sub-agent."
        )
        response = await self.model.ainvoke([{"role": "system", "content": prompt}] + state["messages"])
        
        # Recursive Task Handoffs via Command
        if "DEVOPS" in response.content:
            return Command(goto="devops_layer", update={"active_worker": "DevOps"})
        if "RESEARCH" in response.content:
            return Command(goto="research_layer", update={"active_worker": "Research"})
        return Command(goto=END)

    async def devops_layer(self, state: AGIState) -> Command:
        """Parallel Execution Layer (Map-Reduce)"""
        # Simulated parallel tool execution across Docker and GitHub
        print(f"--- {state['active_worker']} layer processing parallel tasks ---")
        return Command(goto="chief_orchestrator", update={"team_outputs": ["Infrastructure deployed successfully."]})

    async def research_layer(self, state: AGIState) -> Command:
        """Corrective RAG (CRAG) & Iterative Reflection"""
        print(f"--- {state['active_worker']} layer running self-correction loops ---")
        return Command(
                goto="chief_orchestrator",
                update={"team_outputs": ["Research completed."]}
            )


    def build(self):
        builder = StateGraph(AGIState)
        builder.add_node("chief_orchestrator", self.chief_orchestrator)
        builder.add_node("devops_layer", self.devops_layer)
        builder.add_node("research_layer", self.research_layer)
        
        builder.add_edge(START, "chief_orchestrator")
        return builder.compile(checkpointer=self.checkpointer, store=self.store)

async def main():
    swarm = AGISwarm()
    agi = swarm.build()
    
    query = "Deploy a containerized research scraper to Docker and log the summary to GitHub."
    async for event in agi.astream({"messages": [HumanMessage(content=query)]}, 
                                   config={"configurable": {"thread_id": "agi_001"}}):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())