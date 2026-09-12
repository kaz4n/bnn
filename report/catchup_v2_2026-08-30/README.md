# Catch-up report, 30 August 2026

`cw305_catchup.tex` — the report. Figures in `figures/`, tables in `tables/`.

## Building the PDF

`cw305_catchup.pdf` is checked in, 12 pages. `pdflatex` lives in the **WSL Ubuntu**
distribution (`/usr/bin/pdflatex`), not on the Windows `PATH`, so build it from there.
Run twice so `\ref{tab:headline}` resolves:

```bash
wsl.exe -d Ubuntu -e bash -lc 'cd "/mnt/c/Users/narut/OneDrive/Desktop/Project/bnn/report/catchup_v2_2026-08-30" && pdflatex -interaction=nonstopmode cw305_catchup.tex && pdflatex -interaction=nonstopmode cw305_catchup.tex'
```

Builds clean: no undefined references, no overfull boxes. The two wide tables are
wrapped in `\resizebox` because they otherwise run off an A4 page at 2.3 cm margins.

The preamble uses only common packages — `geometry`, `graphicx`, `xcolor`, `amsmath`,
`caption`, `parskip`, `hyperref` — so a base TeX Live install is enough.

## Regenerating figures and tables

```bash
"C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe" make_figures.py
```

```bash
"C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe" make_tables.py
```

`make_tables.py` reads the score JSONs under `attack/hardtests_20260830/` and writes
`tables/*.tex`. No number in the report is typed by hand, so re-running it after any
further experiment updates the report by itself. A run whose score file is absent simply
does not get a row, so the report can never quote a result that was not produced.

Every figure that shows a power signal is drawn from a real capture `.npz`. The
explanatory diagrams (signal chain, authenticity gate, algorithm flow, geometry,
comparison table) are drawn with matplotlib primitives.

## Note on disk space

The machine ran out of disk during these runs. Attack *bundles* are copies of capture
files with the ground truth stripped; each was deleted once its score was written and
can be rebuilt with `active_power_template_cw305.py make-bundle`. For the paper-scale
capture the bulk of the 500 raw traces was also deleted after the template, bundle and
truth were extracted; the capture manifest and three sample traces are kept so
provenance and figures still work. Recapturing takes about five minutes on the bench.
