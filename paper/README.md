# Paper

This folder contains the working LaTeX draft for the research paper.

## Build

From the repository root:

```powershell
.\paper\build.ps1
```

To build and open the PDF:

```powershell
.\paper\build.ps1 -Open
```

The PDF and compile logs are written to:

```text
paper/build/
```

The script tries these compilers in order:

1. `tectonic`
2. `pdflatex`
3. `latexmk`

If compilation fails, read the terminal output first. The detailed LaTeX log is usually available at `paper/build/main.log`.
