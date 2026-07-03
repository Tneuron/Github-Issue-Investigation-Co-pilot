import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@dataclass
class GitHubIssue:
    owner: str
    repo: str
    number: int
    title: str
    body: str
    url: str
    state: str
    comments: list[str] = field(default_factory=list)
    @property
    def full_repo_name(self) -> str:
        return f"{self.owner}/{self.repo}"

class GitHubMCPClient:
    def __init__(self):
        load_dotenv()
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN is missing. Add it to your .env file.")
        self.server_params = StdioServerParameters(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
            ],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": token,
            },
        )

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        async with AsyncExitStack() as stack:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(self.server_params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            response = await session.call_tool(
                "issue_read",
                {
                    "method": "get",
                    "owner": owner,
                    "repo": repo,
                    "issue_number": issue_number,
                },
            )
            issue_data = self._extract_json(response)
            return GitHubIssue(
                owner=owner,
                repo=repo,
                number=issue_number,
                title=issue_data["title"],
                body=issue_data.get("body") or "",
                url=issue_data.get(
                    "html_url",
                    f"https://github.com/{owner}/{repo}/issues/{issue_number}",
                ),
                state=issue_data.get("state", "unknown"),
            )

    async def get_repository_commit(self, owner: str, repo: str) -> str:
        async with AsyncExitStack() as stack:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(self.server_params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            response = await session.call_tool(
                "get_commit",
                {
                    "owner": owner,
                    "repo": repo,
                    "sha": "main",
                    "detail": "none",
                },
            )
            commit_data = self._extract_json(response)
            return commit_data["sha"]

    @staticmethod
    def _extract_json(response) -> dict:
        for item in response.content:
            if hasattr(item, "text") and item.text:
                return json.loads(item.text)
        raise ValueError(
            "GitHub MCP tool returned no JSON content."
        )