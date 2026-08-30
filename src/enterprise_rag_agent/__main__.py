from __future__ import annotations

import uvicorn

from .api import app
from .config import Settings


def main() -> None:
    settings = Settings.load()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()

