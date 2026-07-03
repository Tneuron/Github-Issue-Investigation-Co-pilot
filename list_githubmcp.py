import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from github_mcp_client import GitHubMCPClient


async def main():
    client = GitHubMCPClient()

    async with stdio_client(client.server_params) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()

            print("\n" + "=" * 80)
            print("AVAILABLE GITHUB MCP TOOLS")
            print("=" * 80)

            for tool in tools_result.tools:
                print(f"\nTool: {tool.name}")
                print(f"Description: {tool.description}")
                print("Input schema:")
                print(tool.inputSchema)


if __name__ == "__main__":
    asyncio.run(main())
