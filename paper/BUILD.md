# Building the paper

## Required tools

A standard TeX distribution with `pdflatex`, `bibtex`, and (preferably) `latexmk`.

On macOS, the simplest complete install is MacTeX:

```bash
brew install --cask mactex-no-gui
eval "$(/usr/libexec/path_helper)"
```

A smaller option is BasicTeX, then install `latexmk` and the usual NeurIPS packages with `tlmgr`.

## Compile

From this directory (`paper/`):

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled PDF is written to `paper/main.pdf`.
