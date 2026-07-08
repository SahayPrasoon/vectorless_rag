import sys
from pathlib import Path
from dotenv import load_dotenv


def setup():

    current = Path.cwd()

    project_root = current.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    load_dotenv(project_root / ".env")

    return project_root