import os
import csv
import json
import math

import fitz
import typer
import chevron
import logging
import logging.config
import polars as pl

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from rich.table import Table
from rich.console import Console
from importlib import resources
from typing import Optional
from typing_extensions import Annotated

from openaihelper import functions as F
from openaihelper import data
from openaihelper import __app_name__, __version__

# -----------------------------------------------------------------------------
# setup
# -----------------------------------------------------------------------------

# load environment variables
load_dotenv()

# load in the configuration file
with resources.path(data, "config.yml") as path:
    CONFIG = F.config(path)

# setup logging
logging.config.dictConfig(CONFIG["logging"])
logger = logging.getLogger(__name__)

# setup the rich console
console = Console(style="green on black")

# setup the typer app
app = typer.Typer()


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
def version_callback(value: bool) -> None:
    if value:
        console.print(f"[blue]version: {__app_name__} v{__version__}")
        raise typer.Exit()


# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------
@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show the application's version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None
) -> Optional[bool]:
    """
    OpenAI Helper CLI
    """
    return version


# -----------------------------------------------------------------------------
@app.command()
def config() -> None:
    console.print(f"{CONFIG}")


# -----------------------------------------------------------------------------
@app.command()
def pdf2text(
    in_dir: Annotated[Path, typer.Argument(help="Path to input PDF files")] = None,
    out: Annotated[Path, typer.Option(help="Path to output text files")] = Path("."),
    start: Annotated[int, typer.Option(help="Start page")] = 0,
    end: Annotated[int, typer.Option(help="End page")] = math.inf,
):
    """
    Extract text from a collection of PDF files and write each output to a text file.
    """
    assert end >= start
    check_args(in_dir, CONFIG_FILE_PATH, out)
    if not out.exists():
        out.mkdir(parents=True)

    for pdf in in_dir.glob("*.pdf"):
        console.print(f"processing pdf file: {pdf.name}")
        logging.info(f"extracting text from: {pdf.name}")
        try:
            doc = fitz.open(pdf)
            textfile = out / f"{pdf.stem}.txt"
            pages = [page for page in doc if start <= page.number <= end]
            textfile.write_text(chr(12).join([page.get_text(sort=True) for page in pages]))
        except Exception as e:
            logger.error(f"exception: {type(e)}: {e}")
            continue


# -----------------------------------------------------------------------------
@app.command()
def chat_complete(
    config_file: Annotated[Path, typer.Argument(help="Config file")] = None,
    data_file: Annotated[Path, typer.Argument(help="Data file")] = None,
    id_col: Annotated[str, typer.Option(help="Column name for the id")] = "id",
    out: Annotated[Path, typer.Option(help="Path to output files")] = Path("."),
    count: Annotated[bool, typer.Option(help="Count input tokens")] = False,
):
    """
    Complete a list of chat prompts using OpenAI's API.
    """

    # Read the config file
    if not config_file.exists():
        logger.error(f"Config file {config_file.name} not found")
        console.print(f"[yellow on red]Config file {config_file.name} not found")
        raise typer.Abort()

    config = F.config(config_file)
    system_prompt_template = config["system_prompt"]
    user_prompt_template = config["user_prompt"]
    del config["system_prompt"]
    del config["user_prompt"]

    # Read the data file
    if not data_file.exists():
        msg = f"Data file {data_file.name} not found"
        logger.error(msg)
        console.print(f"[yellow on red]{msg}")
        raise typer.Abort()

    # Make sure the data file has an id column
    df = pl.read_csv(data_file)
    if not id_col in df.columns:
        df = df.with_row_index(name=id_col)

    # Loop through the data
    data: List[Dict] = df.to_dicts()
    for i in tqdm(range(len(data))):
        system_prompt = chevron.render(system_prompt_template, data[i])
        user_prompt = chevron.render(user_prompt_template, data[i])

        if count:
            count_tokens = F.count_tokens(f"{system_prompt}\n{user_prompt}", config["model"])
            logger.info(f"{data[i]['id']} input tokens: {count_tokens}")
            continue

        messages = F.make_message_body(system_prompt, user_prompt)
        config["messages"] = "<messages>"
        logger.info(f"request body for row [{i}]: {config}")
        config["messages"] = messages
        response = F.completion_with_backoff(**config)
        logger.info(f"{response}")
        out_file = out / f"{data[i]['id']}.json"
        response_content = json.loads(response.choices[0].message.json())
        response_content["row_id"] = data[i]["id"]
        out_file.write_text(json.dumps(response_content))


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app()
