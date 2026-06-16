# RapidAPI MCP: Seeking Alpha

This project can use the RapidAPI Seeking Alpha API through two paths:

- Runtime collector path: direct REST calls using `RAPIDAPI_KEY` from `.env`.
- Documentation/discovery path: RapidAPI MCP, when the Codex host has this MCP server registered.

Do not store Seeking Alpha web credentials in this project. The RapidAPI host uses the RapidAPI key, not the Seeking Alpha account password.

## MCP Config

Use `mcp/rapidapi-seeking-alpha.mcp.json.example` as the template. Replace `${RAPIDAPI_KEY}` with the local key only in your user-level Codex MCP config, not in committed project files.

```json
{
  "mcpServers": {
    "RapidAPI Hub - Seeking Alpha": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.rapidapi.com",
        "--header",
        "x-api-host: seeking-alpha.p.rapidapi.com",
        "--header",
        "x-api-key: ${RAPIDAPI_KEY}"
      ]
    }
  }
}
```

## Current Session Note

The MCP server must be registered by the Codex host before it appears as a callable tool. In this session, it is not exposed as a tool yet, so the project continues to use the REST adapter and config-driven endpoints.

## What To Pull From MCP

When the MCP server is available, use it to confirm:

- endpoint path for Seeking Alpha news
- endpoint path for analysis/articles
- endpoint path for quant ratings
- endpoint path for earnings calendar and earnings history
- exact required query parameter names
- response JSON shape for each endpoint
- RapidAPI rate-limit headers

Then update only `configs/default.toml` unless the response shape requires a new parser.
