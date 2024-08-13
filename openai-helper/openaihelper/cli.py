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
    config_file: Annotated[Path, typer.Argument(help="Config file", callback=path_callback)] = None,
    data_file: Annotated[Path, typer.Argument(help="Data file", callback=path_callback)] = None,
    id_col: Annotated[str, typer.Option(help="Column name for the id", rich_help_panel="Options")] = "id",
    out: Annotated[Path, typer.Option("--out", "-o", help="Path to output files", rich_help_panel="Options")] = Path("."),
    count: Annotated[bool, typer.Option(help="Count input tokens", rich_help_panel="Options")] = False,
    batch: Annotated[bool, typer.Option(help="Create a batch file", rich_help_panel="Options")] = False,
):
    """
    Complete a list of chat prompts using OpenAI's API.
    """

    # Read the config file
    config = F.config(config_file)
    system_prompt_template = config["system_prompt"]
    user_prompt_template = config["user_prompt"]
    del config["system_prompt"]
    del config["user_prompt"]
    logger.info(f"request body parameters: {config}")

    # Read the data file
    df = pl.read_csv(data_file)
    if not id_col in df.columns:
        df = df.with_row_index(name=id_col)

    # Create the output file
    if not out.exists():
        out.mkdir(parents=True)
    out_file = out / "responses.jsonl"
    out_file.write_text("")

    # Loop through the data
    data: List[Dict] = df.to_dicts()
    for i in track(range(len(data)), description="Processing..."):
        system_prompt = chevron.render(system_prompt_template, data[i])
        user_prompt = chevron.render(user_prompt_template, data[i])

        if count:
            count_tokens = F.count_tokens(f"{system_prompt}\n{user_prompt}", config["model"])
            logger.info(f"{data[i]['id']} input tokens: {count_tokens}")
            continue

        messages = F.make_message_body(system_prompt, user_prompt)
        config["messages"] = messages
        response = F.completion_with_backoff(**config)
        logger.info(f"{response}")
        response_content = json.loads(response.choices[0].message.json())
        response_content["row_id"] = data[i]["id"]
        with open(out_file, "a") as f:
            f.write(f"{json.dumps(response_content)}\n")
            f.flush()


# -----------------------------------------------------------------------------
@app.command()
def make_batch(
    config_file: Annotated[Path, typer.Argument(help="Config file", callback=path_callback)] = None,
    data_file: Annotated[Path, typer.Argument(help="Data file", callback=path_callback)] = None,
    id_col: Annotated[str, typer.Option(help="Column name for the id", rich_help_panel="Options")] = "id",
    out: Annotated[Path, typer.Option("--out", "-o", help="Path to output file", rich_help_panel="Options")] = Path("."),
    batch_name: Annotated[str, typer.Option("--batch", help="Batch name", rich_help_panel="Options")] = "batch",
):
    """
    Make a batch file for OpenAI's API.
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
def upload_batch(
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
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app()
