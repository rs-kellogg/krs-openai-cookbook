from pathlib import Path
from tqdm import tqdm
import pandas as pd
import openai
import csv
import json
import logging
from typing import Optional
import fitz
from rich import console as cons
from tenacity import wait_random_exponential
from openaihelper import functions as F
import typer


# -----------------------------------------------------------------------------
console = cons.Console(style="green on black")

logging.basicConfig(filename="openai-helper.log", encoding="utf-8", level=logging.INFO)


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
app = typer.Typer()
if __name__ == "__main__":
    app()


# -----------------------------------------------------------------------------
@app.command()
def text2speech(
    in_dir: Path = typer.Argument(..., help="Path to input text files"),
    out_dir: Optional[Path] = typer.Option(
        Path("."),
        "--dir",
        help="The directory where the audio files will be created.",
    ),
):
    """
    Convert a collection of text files to audio files.
    """
    client = openai.OpenAI()
    for file in in_dir.glob("*.txt"):
        console.print(f"processing text file: {file.name}")
        speech_file_path = out_dir / f"{file.stem}.mp3"
        response = F.text2speech(client, file.read_text())
        response.stream_to_file(speech_file_path)


# -----------------------------------------------------------------------------
@app.command()
def speech2text(
    in_dir: Path = typer.Argument(..., help="Path to input audio files"),
    out_dir: Optional[Path] = typer.Option(
        Path("."),
        "--dir",
        help="The directory where the text files will be created.",
    ),
):
    """
    Convert a collection of audio files to text files.
    """
    client = openai.OpenAI()
    for file in in_dir.glob("*.mp3"):
        console.print(f"processing audio file: {file.name}")
        text_file_path = out_dir / f"{file.stem}.txt"
        response = F.speech2text(client, file)
        text_file_path.write_text(response.text)


# -----------------------------------------------------------------------------
@app.command()
def pdf2text(
    in_dir: Path = typer.Argument(..., help="Path to input PDF files"),
    out_dir: Optional[Path] = typer.Option(
        Path("."),
        "--dir",
        help="The directory where the extracted output files will be created.",
    ),
    start_page: Optional[int] = typer.Option(0, help="Start page"),
    end_page: Optional[int] = typer.Option(10_000, help="End page"),
):
    """
    Extract text from a collection of PDF files and write each output to a text file.
    """
    assert end_page >= start_page
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for pdf in in_dir.glob("*.pdf"):
        console.print(f"processing pdf file: {pdf.name}")
        logging.info(f"extracting text from: {pdf.name}")
        try:
            doc = fitz.open(pdf)
            textfile = out_dir / f"{pdf.stem}.txt"
            pages = [page for page in doc if start_page <= page.number <= end_page]
            textfile.write_text(chr(12).join([page.get_text(sort=True) for page in pages]))
        except Exception as e:
            logger.error(f"exception: {type(e)}: {e}")
            continue


# -----------------------------------------------------------------------------
@app.command()
def count_tokens(
    data_file_path: Path = typer.Argument(..., help="Data file path name"),
    config_file_path: Path = typer.Argument(..., help="Config file path name"),
    outdir: Path = typer.Option(Path("."), help="Output directory"),
):
    """
    Count tokens in a data file and write output to a csv file.
    """
    check_args(data_file_path, config_file_path, outdir)

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
    out_csv = open(f"{outdir}/{data_file_path.stem}_counts.csv", "a")
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
    outdir: Path = typer.Option(Path("."), help="Output directory"),
):
    """
    Accept a data file with text and complete the prompt for each text and write output to a csv file.
    """
    check_args(data_file_path, config_file_path, outdir)

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
    out_csv_path = outdir / f"{data_file_path.stem}_responses.csv"
    if not out_csv_path.exists():
        out_csv = open(out_csv_path, "w")
        writer = csv.writer(out_csv)
        writer.writerow(["id", "response"])
        out_csv.close()

    with open(f"{outdir}/{data_file_path.stem}_responses.csv", "a") as out_csv:
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
