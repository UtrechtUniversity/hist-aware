import sys
import os

from typing import Optional
from tabulate import tabulate
import click-spinner
import typer
from notebook.utils import *
from notebook.text_selection import *

app = typer.Typer()

@app.command()
def prepare_files(name: Optional[str] = None):
    if name:
        typer.echo(f"Hello {name}")
    else:
        typer.echo("Hello World!")


if __name__ == "__main__":
    app()
