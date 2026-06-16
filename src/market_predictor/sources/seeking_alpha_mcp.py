from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeekingAlphaMcpEndpointSpec:
    name: str
    endpoint: str
    params: str
    feed_type: str


class SeekingAlphaMcpDiscovery:
    """Boundary for future RapidAPI MCP-backed endpoint discovery.

    The current Codex session does not expose the RapidAPI MCP server as a
    callable tool. Keep this boundary separate from the runtime REST collector
    so endpoint discovery can be added without touching model or feature code.
    """

    def available(self) -> bool:
        return False

    def discover_endpoint_specs(self) -> list[SeekingAlphaMcpEndpointSpec]:
        raise RuntimeError(
            "RapidAPI Seeking Alpha MCP is not registered in this Codex session. "
            "Register mcp/rapidapi-seeking-alpha.mcp.json.example in the host config, "
            "then rerun endpoint discovery."
        )
