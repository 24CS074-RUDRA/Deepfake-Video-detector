from pathlib import Path

def create_folder(folder: Path):

    folder.mkdir(parents=True, exist_ok=True)