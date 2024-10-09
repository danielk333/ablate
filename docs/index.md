# `ablate`

DESCRIPTION HERE

## Installing

```bash
pip install ablate
```

### msise00

```bash
    mkdir /my/env/msise00_source
    cd /my/env/msise00_source
    git clone https://github.com/scivision/msise00
    cd msise00
    pip install -e .
    python -c "import msise00; msise00.build()"
```

## Develop

### Internal development

Please refer to the style and contribution guidelines documented in the
[IRF Software Contribution Guide](https://danielk.developer.irf.se/software_contribution_guide/).

### External code-contributions

Generally external code-contributions are made trough a "Fork-and-pull"
workflow towards the `main` branch.
