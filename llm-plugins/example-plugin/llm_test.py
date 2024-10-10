import httpx
import ijson
import llm
import urllib.parse
from rich.console import Console
import click
from typing_extensions import Annotated

console = Console(style="green on black")


@llm.hookimpl
def register_commands(cli):
    @cli.command(name="hello-world")
    @click.option("--message", help="Message")
    def hello_world(
        message: str = "Hello world!",
    ):
        "Print Message"
        console.print(message)
    
