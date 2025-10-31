# `metablate`

DESCRIPTION HERE

## Installing

```bash
pip install metablate
```

### msise00

`msise00` currently has a bug where it needs to be manually built upon install, the following
commands will fix that issue.

```bash
    mkdir /my/env/msise00_source
    cd /my/env/msise00_source
    git clone https://github.com/scivision/msise00
    cd msise00
    pip install -e .
    python -c "import msise00; msise00.build()"
```

## Develop

### External code-contributions

Generally external code-contributions are made trough a "Fork-and-pull"
workflow towards the `main` branch.
