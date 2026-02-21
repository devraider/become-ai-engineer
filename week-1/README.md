# Week 1 - Setup your AI environment

Start your AI journey by setting up your environment and tools. In this week, we will cover the following topics:

### Python - `uv`

> [https://docs.astral.sh/uv/guides/integration/fastapi/](https://docs.astral.sh/uv/getting-started/installation/)

### Install

Mac/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows

```PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### VS Code

> https://code.visualstudio.com/

#### Python Extension

> https://marketplace.visualstudio.com/items?itemName=ms-python.python
> https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
> https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy
> https://marketplace.visualstudio.com/items?itemName=ms-python.pylint
> https://marketplace.visualstudio.com/items?itemName=ms-toolsai.python-ds-extension-pack

### GIT

> https://git-scm.com/install/

### Docker

> https://www.docker.com/get-started/

### Node and NPM

> https://nodejs.org/

For simplicity `nvm` is a great alternative:

https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating

### Take away

- setup Python AI environment
- build AI API with endpoints

**Example**:

- POST /sentiment
- POST /summary

Create sentiment analysis endpoints:

```python

from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

sentiment = pipeline("sentiment-analysis")

@app.post("/sentiment")
def analyze(text: str):
    return sentiment(text)
```
