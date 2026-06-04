# Privacy_Assignment_4

This repository include our main implementation of Pseudonymization using Local Ollama and a test file that lays the ground works for Differential Privacy using diffprivlib.

## Getting Started

### Dependencies

To run this project you need Ollama running locally with Gemma4 installed

To install the Python dependencies 
```
uv init --bare
uv add -r requirements.txt
```

### Run the Pseudonymization

```
uv run app.py presidio
```

OR

```
uv run app.py llm
```

You can also do `uv run app.py both` to run both sanitizers in parallel, however this seems to give poor results in comparison to them individually.