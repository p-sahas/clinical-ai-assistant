"""
Web Search Tool using Tavily API.

This is the ONLY external tool registered for agent use.
It represents actions on the external world (unlike RAG which is internal memory).

References:
- Neural Maze: Agent Tools - The Bridge to the Outside World
  https://theneuralmaze.substack.com/p/agent-tools-the-bridge-to-the-outside
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
from tavily import TavilyClient

from utils.config import get_config, get_api_key


class WebSearchTool:
    """
    Tavily-powered web search tool for real-world healthcare queries.
    
    Focuses on operational information like hospital hours, contacts,
    addresses, and current announcements.
    """
    
    def __init__(
        self,
        max_results: Optional[int] = None,
        prefer_domains: Optional[List[str]] = None,
        freshness_days: Optional[int] = None,
        search_depth: str = "advanced"
    ):
        """
        Initialize web search tool.
        
        Args:
            max_results: Maximum search results to return
            prefer_domains: Preferred domain extensions (.lk, .gov, .org)
            freshness_days: Prefer results within N days
            search_depth: "basic" or "advanced"
        """
        config = get_config()
        
        self.api_key = get_api_key("tavily")
        self.client = TavilyClient(api_key=self.api_key)
        
        self.max_results = max_results or config.get("web.max_results", 5)
        self.prefer_domains = prefer_domains or config.get("web.prefer_domains", [".lk", ".gov", ".org"])
        self.freshness_days = freshness_days or config.get("web.freshness_days", 120)
        self.search_depth = search_depth or config.get("web.search_depth", "advanced")
        self.timezone = ZoneInfo(config.get("output.timezone", "Asia/Colombo"))
        
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform web search for query.
        
        Args:
            query: Search query
            max_results: Override default max results
            include_domains: Specific domains to include
            
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        
        # Prepare search parameters
        search_params = {
            "query": query,
            "max_results": max_results or self.max_results,
            "search_depth": self.search_depth,
            "include_answer": True,
            "include_raw_content": False
        }
        
        # Add domain filtering if specified
        if include_domains:
            search_params["include_domains"] = include_domains
        
        try:
            # Perform search
            response = self.client.search(**search_params)
            
            # Process results
            results = self._process_results(response)
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "answer": response.get("answer", ""),
                "result_count": len(results),
                "latency_ms": latency_ms,
                "checked_at": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
            }
            
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "results": [],
                "latency_ms": latency_ms
            }
    
    def _process_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process and rank search results.
        
        Args:
            response: Raw Tavily response
            
        Returns:
            Processed and ranked results
        """
        results = response.get("results", [])
        
        if not results:
            return []
        
        # Score results based on preferences
        scored_results = []
        
        for result in results:
            url = result.get("url", "")
            score = result.get("score", 0.5)
            
            # Boost preferred domains
            for domain in self.prefer_domains:
                if domain in url:
                    score += 0.2
                    break
            
            # Check if government or official site
            if ".gov" in url or "official" in url.lower():
                score += 0.15
            
            scored_results.append({
                "title": result.get("title", ""),
                "url": url,
                "content": result.get("content", ""),
                "score": min(score, 1.0),  # Cap at 1.0
                "published_date": result.get("published_date", "")
            })
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results
    
    def format_results(
        self,
        search_response: Dict[str, Any],
        include_urls: bool = True
    ) -> str:
        """
        Format search results as text.
        
        Args:
            search_response: Response from search()
            include_urls: Include URLs in output
            
        Returns:
            Formatted results string
        """
        if search_response["status"] == "error":
            return f"Web search failed: {search_response.get('error', 'Unknown error')}"
        
        results = search_response["results"]
        
        if not results:
            return (
                "No reliable web results found. "
                "Please verify information by calling the official hospital hotline."
            )
        
        output_parts = []
        
        # Add Tavily's answer summary if available
        if search_response.get("answer"):
            output_parts.append(f"Summary: {search_response['answer']}\n")
        
        # Add individual results
        output_parts.append("External Web Sources:\n")
        
        for idx, result in enumerate(results[:self.max_results], 1):
            title = result["title"]
            content = result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"]
            
            result_text = f"{idx}. {title}\n{content}"
            
            if include_urls:
                result_text += f"\nURL: {result['url']}"
            
            output_parts.append(result_text)
        
        # Add metadata
        checked_at = search_response.get("checked_at", "")
        if checked_at:
            output_parts.append(f"\nInformation checked at: {checked_at}")
        
        return "\n\n".join(output_parts)
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get tool definition for agent registration (OpenAI format).
        
        Returns:
            Tool definition dictionary
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for current, real-world information about healthcare facilities, "
                    "hospital hours, contact information, addresses, and operational announcements. "
                    "Use this when internal knowledge doesn't contain operational/current information. "
                    "Prefers official domains (.lk, .gov, .org)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query about hospital/healthcare operational information"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }


def register_web_search_tool() -> WebSearchTool:
    """
    Create and register web search tool with global tool registry.
    
    Returns:
        WebSearchTool instance
    """
    from utils.llm_services import get_tool_registry
    
    tool = WebSearchTool()
    registry = get_tool_registry()
    
    # Register the tool
    def web_search_func(query: str, max_results: int = 5) -> str:
        """Execute web search and return formatted results."""
        response = tool.search(query, max_results=max_results)
        return tool.format_results(response)
    
    registry.register_tool(
        name="web_search",
        func=web_search_func,
        description=tool.get_tool_definition()["function"]["description"],
        parameters=tool.get_tool_definition()["function"]["parameters"]
    )
    
    return tool


# Global tool instance
_web_search_instance: Optional[WebSearchTool] = None


def get_web_search_tool() -> WebSearchTool:
    """
    Get global web search tool instance (singleton pattern).
    
    Returns:
        WebSearchTool instance
    """
    global _web_search_instance
    
    if _web_search_instance is None:
        _web_search_instance = WebSearchTool()
        
    return _web_search_instance

