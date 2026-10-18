from pathlib import Path


def get_data_dir():
  """Return the project data directory, creating it when needed."""
  data_dir = Path(__file__).resolve().parent.parent / "data"
  data_dir.mkdir(parents=True, exist_ok=True)
  return data_dir
