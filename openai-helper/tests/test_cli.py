import os
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from openaihelper.cli import app
import pytest


runner = CliRunner()

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


def test_cli(tmpdir):
    result = runner.invoke(
        app,
        ["count-tokens", 
         "--help"
         ],
    )
    assert result.exit_code == 0
    assert "Usage: root count-tokens [OPTIONS] DATA_FILE_PATH CONFIG_FILE_PATH" in result.stdout
    assert "Count tokens in a data file and write output to a csv file." in result.stdout


    result = runner.invoke(
        app,
        ["count-tokens", 
         str(dir_path/"data/arxiv_metadata-1.csv"), 
         str(dir_path/"config.yml"), 
         "--outdir", 
         str(tmpdir),
         ],
    )
    assert result.exit_code == 0
    assert "arxiv_metadata-1_counts.csv" in os.listdir(tmpdir)
    assert "id,count" in (tmpdir/"arxiv_metadata-1_counts.csv").read_text("utf-8")
    assert "2310.00014,1346" in (tmpdir/"arxiv_metadata-1_counts.csv").read_text("utf-8")

