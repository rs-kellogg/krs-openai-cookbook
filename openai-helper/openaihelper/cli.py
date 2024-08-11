import os
import csv
import json
import yaml

import fitz
import typer
import openai
import logging
import logging.config

import pandas as pd
import polars as pl

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from rich import console as cons
from tenacity import wait_random_exponential

from openaihelper import functions as F

# -----------------------------------------------------------------------------
# setup
# -----------------------------------------------------------------------------

# load environment variables
load_dotenv()

# setup the typer app
app = typer.Typer()

# setup the rich console
console = cons.Console(style="green on black")

# load in the configuration file
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
with open(dir_path / "config.yaml") as f:
    config = yaml.safe_load(f.read())

# setup logging
logging.config.dictConfig(config["logging"])
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
def check_args(
    data_file_path: Path,
    config_file_path: Path,
    outdir: Path,
):
    assert data_file_path.exists()
    assert config_file_path.exists()
    if not outdir.exists():
        outdir.mkdir(parents=True)


# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
@app.command()
def pdf2text(
    in_dir: Path = typer.Argument(..., help="Path to input PDF files"),
    out: Optional[Path] = typer.Option(
        Path("."),
        "--dir",
        help="The directory where the extracted output files will be created.",
    ),
    start: Optional[int] = typer.Option(0, help="Start page"),
    end: Optional[int] = typer.Option(10_000, help="End page"),
):
    """
    Extract text from a collection of PDF files and write each output to a text file.
    """
    assert end >= start
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
def count_tokens(
    data_file_path: Path = typer.Argument(..., help="Data file path name"),
    config_file_path: Path = typer.Argument(..., help="Config file path name"),
    out: Path = typer.Option(Path("."), help="Output directory"),
):
    """
    Count tokens in a data file and write output to a csv file.
    """
    check_args(data_file_path, config_file_path, out)

    # Read config file
    config = F.config(config_file_path)
    encoding_name = config["encoding_name"]
    max_token_len = config["max_token_len"]

    # Read data file
    df = pd.read_csv(data_file_path)
    assert "id" in df.columns
    assert "text" in df.columns
    texts = list(df["text"])
    ids = list(df["id"])

    # Count tokens and write results to csv file
    out_csv = open(f"{out}/{data_file_path.stem}_counts.csv", "a")
    writer = csv.writer(out_csv)
    writer.writerow(["id", "count"])
    for i in tqdm(range(len(texts))):
        n_tokens = F.count_tokens(texts[i], encoding_name)
        writer.writerow([ids[i], n_tokens])
        out_csv.flush()
        logging.info(f"Data point {ids[i]} has {n_tokens} tokens")
    out_csv.close()


# -----------------------------------------------------------------------------
@app.command()
def complete_prompt(
    data_file_path: Path = typer.Argument(..., help="Data file path name"),
    config_file_path: Path = typer.Argument(..., help="Config file path name"),
    out: Path = typer.Option(Path("."), help="Output directory"),
):
    """
    Accept a data file with text and complete the prompt for each text and write output to a csv file.
    """
    check_args(data_file_path, config_file_path, out)

    # Read config file and setup OpenAI
    config = F.config(config_file_path)
    encoding_name = config["encoding_name"]
    max_token_len = config["max_token_len"]
    model_name = config["model_name"]
    user_prompt = config["user_prompt"]
    system_prompt = config["system_prompt"]
    n_prompt_tokens = F.count_tokens(system_prompt + user_prompt, encoding_name)

    # Read the data file and check its format
    df = pd.read_csv(data_file_path)
    assert "id" in df.columns
    assert "text" in df.columns
    texts = list(df["text"])
    ids = list(df["id"])

    # Complete prompt for each row and write results to csv file
    out_csv_path = out / f"{data_file_path.stem}_responses.csv"
    if not out_csv_path.exists():
        out_csv = open(out_csv_path, "w")
        writer = csv.writer(out_csv)
        writer.writerow(["id", "response"])
        out_csv.close()

    with open(f"{out}/{data_file_path.stem}_responses.csv", "a") as out_csv:
        writer = csv.writer(out_csv)
        client = openai.OpenAI()

        for i in tqdm(range(len(texts))):
            # check if the text is too long
            n_tokens = F.count_tokens(texts[i], encoding_name)
            if (n_tokens + n_prompt_tokens) > max_token_len:
                writer.writerow([ids[i], "TOO_LONG"])
                logging.warn(f"Data point {ids[i]} not completed")
            else:
                # complete the prompt
                response = F.chat_complete(
                    client=client,
                    model_name=model_name,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    text=texts[i],
                )
                # validate the response
                valid = F.validate_result(response)
                if valid:
                    logging.info(f"Data point {ids[i]} completed")
                    writer.writerow([ids[i], json.dumps(response)])
                else:
                    logging.warn(f"Error for data point {ids[i]}: {response}")
            out_csv.flush()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app()
