# Meteoroid Ablation Models

## Install

To install:
```bash
    pip install ablate
```

### msise00 atmospheric model

```bash
    mkdir /my/env/msise00_source
    cd /my/env/msise00_source
    git clone https://github.com/scivision/msise00
    cd msise00
    pip install -e .
    python -c "import msise00; msise00.build()"
```


## Development

### Poetry

To generate a new `requirements.txt` file to be used in the GitLab CI/CD chain run:

```bash
poetry export -f requirements.txt --output requirements.txt
```

