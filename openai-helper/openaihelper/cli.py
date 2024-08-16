import os
import csv
import json
import math

import fitz
import typer
import openai
import chevron
import logging
import logging.config
import polars as pl

from pathlib import Path
from dotenv import load_dotenv
from rich.table import Table
from rich.progress import track
from rich.console import Console
from datetime import datetime
from importlib import resources
from typing import Optional
from typing_extensions import Annotated

from openaihelper import utils as F
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
        console.print(f"[blue]{__version__}")
        raise typer.Exit()


# -----------------------------------------------------------------------------
def path_callback(path: Path):
    if not path.exists():
        raise typer.BadParameter(f"Path {path} does not exist")
    return path


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
    batch_file: Annotated[Path, typer.Argument(help="Batch file", callback=path_callback)] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Path to output files", rich_help_panel="Options")] = Path("."),
    format: Annotated[str, typer.Option("--format", "-f", help="Output format", rich_help_panel="Options")] = "json",
):
    """
    Run a batch file line by line in synchronous non-batch mode.
    """

    if not out.exists():
        out.mkdir(parents=True)

    # Read the batch file
    with open(batch_file, "r") as f:
        client = openai.OpenAI(
            organization=os.environ["OPENAI_ORG_ID"],
            project=os.environ["OPENAI_PROJ_ID"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        requests = f.readlines()
        for i, request in enumerate(requests):
            request = json.loads(request)
            logger.info(f"processing request: {request['custom_id']}")
            console.print(f"processing request: {request['custom_id']}")
            response = F.completion_with_backoff(client, request['body'])
            if format == "text":
                out_file = out / f"{request['custom_id']}-response.txt"
                out_file.write_text(response.choices[0].message.content)
            elif format == "json":   
                out_file = out / f"{request['custom_id']}-response.json"
                out_file.write_text(response.to_json())


# -----------------------------------------------------------------------------
@app.command()
def make_batch_file(
    config_file: Annotated[Path, typer.Argument(help="Config file", callback=path_callback)] = None,
    data_file: Annotated[Path, typer.Argument(help="Data file", callback=path_callback)] = None,
    id_col: Annotated[str, typer.Option(help="Column name for the id", rich_help_panel="Options")] = "id",
    out: Annotated[Path, typer.Option("--out", "-o", help="Path to output file", rich_help_panel="Options")] = Path("."),
    batch_name: Annotated[str, typer.Option("--batch", help="Batch name", rich_help_panel="Options")] = "batch",
) -> None:
    """
    Make a batch file for OpenAI
    """

    # Read the config file
    config = F.config(config_file)
    system_prompt_template = config["system_prompt"]
    user_prompt_template = config["user_prompt"]
    del config["system_prompt"]
    del config["user_prompt"]

    # Read the data file
    df = pl.read_csv(data_file)
    if not id_col in df.columns:
        df = df.with_row_index(name=id_col)

    # Create the output file
    if not out.exists():
        out.mkdir(parents=True)
    out_file = out / f"{batch_name}-requests.jsonl"
    out_file.write_text("")

    # Loop through the data to create a jsonl batch file
    requests = []
    data: List[Dict] = df.to_dicts()
    for index in track(range(len(data)), description="Processing..."):
        system_prompt = chevron.render(system_prompt_template, data[index])
        user_prompt = chevron.render(user_prompt_template, data[index])
        messages = F.make_message_body(system_prompt, user_prompt)
        config = config.copy()
        config["messages"] = messages
        request = {
            "custom_id": f"id_{data[index]['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": config,
        }
        requests.append(request)

    out_file.write_text("\n".join([json.dumps(r) for r in requests]))
    console.print(f"Batch file created: {out_file}")


# -----------------------------------------------------------------------------
@app.command()
def upload_batch_file(
    batch_file: Annotated[Path, typer.Argument(help="Batch file", callback=path_callback)] = None,
):
    """
    Upload a batch file to OpenAI
    """
    client = openai.OpenAI(
        organization=os.environ["OPENAI_ORG_ID"],
        project=os.environ["OPENAI_PROJ_ID"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    batch_input_file = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    console.print(f"Uploaded batch file: {batch_file}")
    console.print(f"[orange1]{batch_input_file}")
    logger.info(f"Uploaded batch file: {batch_file}")
    logger.info(f"{batch_input_file}")


# -----------------------------------------------------------------------------
@app.command()
def start_batch(
    batch_file_id: Annotated[str, typer.Argument(help="Batch file ID")] = None,
    description: Annotated[str, typer.Option("--desc", help="Description of the batch job")] = "batch job",
):
    """
    Start a batch job
    """
    client = openai.OpenAI(
        organization=os.environ["OPENAI_ORG_ID"],
        project=os.environ["OPENAI_PROJ_ID"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # create batch job
    batch_create_response = client.batches.create(
        input_file_id=batch_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    logger.info(batch_create_response)
    console.print(batch_create_response)


# -----------------------------------------------------------------------------
@app.command()
def get_batch_results(
    batch_id: Annotated[str, typer.Argument(help="Batch ID")] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Path to output file")] = Path("."),
    batch_name: Annotated[str, typer.Option("--batch", help="Batch name", rich_help_panel="Options")] = "batch",
):
    """
    Download batch results to a file if the batch job is completed.
    If not completed, the status is displayed.
    """
    client = openai.OpenAI(
        organization=os.environ["OPENAI_ORG_ID"],
        project=os.environ["OPENAI_PROJ_ID"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    batch_retrieve_response = client.batches.retrieve(batch_id)
    logger.info(batch_retrieve_response)
    console.print(batch_retrieve_response)
    if batch_retrieve_response.status == "completed":
        file_response = client.files.content(batch_retrieve_response.output_file_id)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f"{batch_name}-responses.jsonl"
        out_file.write_text(file_response.text)
        logger.info(f"writing json output to {out_file}")
        console.print(f"[orange1]writing json output to {out_file}")



# -----------------------------------------------------------------------------
@app.command()
def list_batches(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit the number of batches to list")] = 100,
):
    client = openai.OpenAI(
        organization=os.environ["OPENAI_ORG_ID"],
        project=os.environ["OPENAI_PROJ_ID"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    batches = client.batches.list(limit=limit)
    batches = sorted(batches, key=lambda x: x.created_at)
    for b in batches:
        console.print(b.id, b.status, datetime.fromtimestamp(b.created_at))

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app()
