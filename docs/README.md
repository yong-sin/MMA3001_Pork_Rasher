# API Documentation

This folder contains the HTML documentation for the project's Python source code.

- `index.html`: API documentation index.
- `dataset_preparation.html`: Dataset preparation and validation functions in `src/dataset_preparation.py`.
- `inspection.html`: Packaging inspection function in `inspection.py`.

The HTML pages explain the functions, parameters, return values, and implementation.

## Regenerate the HTML with pdoc

Run the following commands from the repository root in an environment with Python and the required packages:

```
bash


python -m pip install pdoc pyyaml ultralytics
PYTHONPATH=./src python -m pdoc inspection dataset_preparation -o docs
```

Rebuild the pages after editing either source file so the documentation stays in sync with the code.
