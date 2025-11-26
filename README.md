# Implementing AI Agent with LlamaIndex and LangChain

## Skills
- **Document Search**: Search in local documents using RAG (LlamaIndex).
- **Web Search**: Search the web using DuckDuckGo.
- **Wikipedia**: Query Wikipedia for general knowledge.
- **Calculator**: Perform accurate mathematical calculations.
- **Time**: Check the current date and time.
- **Python REPL**: Execute Python code for complex logic.
- **File System**: Read, write, and manage local files.
- **Joke**: Get a random joke from an API.

## How to Use
- Install the dependencies:
```bash
pip install -r requirements.txt
```

- Get an API key for [Google Gemini](https://ai.google.dev/gemini-api/docs/api-key) and put it into .env file.

- Run the code:
```bash
python -m src.main
```

## Development
- Run tests:
```bash
pytest
```
- Format code:
```bash
black .
isort .
```
- Lint code:
```bash
flake8 .
mypy .
```