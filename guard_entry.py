"""Точка входа exe (PyInstaller)."""
import sys

from guard.app.main import main

if __name__ == "__main__":
    sys.exit(main())
