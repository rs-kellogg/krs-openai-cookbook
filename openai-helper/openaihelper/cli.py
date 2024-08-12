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
    out: Annotated[Path, typer.Option(help="Path to output text files")] = Path("."),
    count: Annotated[bool, typer.Option(help="Count input tokens")] = False,
    data_file: Annotated[Path, typer.Option("--data", help="Data file")] = None,
    id_col: Annotated[str, typer.Option(help="ID column name")] = "id",
    text_col: Annotated[str, typer.Option(help="Text column name")] = "text",
):
    """
    Accept a data file with text and complete the prompt for each text and write output to a csv file.
    """

    # Read the config file and set the message body
    assert config_file.exists()
    config = F.config(config_file)
    system_prompt = config["system_prompt"]
    user_prompt = config["user_prompt"]
    messages = F.make_message_body(system_prompt, user_prompt)
    del config["system_prompt"]
    del config["user_prompt"]
    config["messages"] = messages

    # Read the data file and check its format
    if data_file:
        assert data_file.exists()    
    df = pl.read_csv(data_file)
    assert "id" in df.columns
    assert "text" in df.columns
    texts = list(df["text"])
    ids = list(df["id"])
    for id, text in zip(ids, texts):
        console.print(f"processing row: {id}")
        logger.info(f"request: {config}")
        response = F.completion_with_backoff(**config)
        logger.info(f"{response}")
        
    # if not out.exists():
    #     out.mkdir(parents=True)
    # # Complete prompt for each row and write results to csv file
    # out_csv_path = out / f"{data_file_path.stem}_responses.csv"
    # if not out_csv_path.exists():
    #     out_csv = open(out_csv_path, "w")
    #     writer = csv.writer(out_csv)
    #     writer.writerow(["id", "response"])
    #     out_csv.close()

    # with open(f"{out}/{data_file_path.stem}_responses.csv", "a") as out_csv:
    #     writer = csv.writer(out_csv)

    #     for i in tqdm(range(len(texts))):
    #         # check if the text is too long
    #         n_tokens = F.count_tokens(texts[i], encoding_name)
    #         if (n_tokens + n_prompt_tokens) > max_token_len:
    #             writer.writerow([ids[i], "TOO_LONG"])
    #             logging.warn(f"Data point {ids[i]} not completed")
    #         else:
    #             # complete the prompt
    #             response = F.chat_complete(
    #                 client=client,
    #                 model_name=model_name,
    #                 user_prompt=user_prompt,
    #                 system_prompt=system_prompt,
    #                 text=texts[i],
    #             )
    #             # validate the response
    #             valid = F.validate_result(response)
    #             if valid:
    #                 logging.info(f"Data point {ids[i]} completed")
    #                 writer.writerow([ids[i], json.dumps(response)])
    #             else:
    #                 logging.warn(f"Error for data point {ids[i]}: {response}")
    #         out_csv.flush()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app()
