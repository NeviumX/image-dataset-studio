"""PyInstaller entry point; package-relative imports stay inside the package."""

from image_dataset_studio.app import main

if __name__ == "__main__":
    raise SystemExit(main())
