"""Where an image named in a row is found. The rows of this project are text only (`images: []`); the function stays
because the collator and the length planner take image paths relative to the dataset root."""
from pathlib import Path


def resolve_artifact_ref(path, root):
    return Path(root) / path
