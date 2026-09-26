param(
    [switch]$Open
)

# "Continue", not "Stop": under Windows PowerShell 5.1 any stderr line from a
# native tool (e.g. MiKTeX's "you have not checked for updates" notice) becomes
# a terminating error under "Stop". Real failures are caught by the PDF check.
$ErrorActionPreference = "Continue"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $PaperDir "build"
$MainTex = Join-Path $PaperDir "main.tex"
$PdfPath = Join-Path $BuildDir "main.pdf"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

Push-Location $PaperDir
try {
    if (Test-CommandExists "tectonic") {
        tectonic --keep-logs --keep-intermediates --outdir $BuildDir $MainTex
    }
    elseif (Test-CommandExists "pdflatex") {
        pdflatex -interaction=nonstopmode -file-line-error -halt-on-error "-output-directory=$($BuildDir)" $MainTex
        if (Test-CommandExists "bibtex") {
            # bibtex resolves \bibdata relative to its working directory, so run it
            # inside the build dir with BIBINPUTS pointing back at the paper dir.
            $env:BIBINPUTS = "$PaperDir;"
            Push-Location $BuildDir
            try { bibtex main } finally { Pop-Location }
        }
        pdflatex -interaction=nonstopmode -file-line-error -halt-on-error "-output-directory=$($BuildDir)" $MainTex
        pdflatex -interaction=nonstopmode -file-line-error -halt-on-error "-output-directory=$($BuildDir)" $MainTex
    }
    elseif (Test-CommandExists "latexmk") {
        latexmk -pdf -interaction=nonstopmode -file-line-error -halt-on-error "-outdir=$($BuildDir)" $MainTex
    }
    else {
        throw "No LaTeX compiler found. Install Tectonic, MiKTeX, or TeX Live, then run this script again."
    }
}
finally {
    Pop-Location
}

if (Test-Path $PdfPath) {
    Write-Host "Built PDF: $PdfPath"
    # Page-budget guardrail (DAN-24): the main body must end on page 12 or earlier.
    $AuxPath = Join-Path $BuildDir "main.aux"
    $EndLine = Select-String -Path $AuxPath -Pattern '\\newlabel\{end-of-main\}\{\{[^}]*\}\{(\d+)\}' | Select-Object -First 1
    if ($EndLine) {
        $EndPage = [int]$EndLine.Matches[0].Groups[1].Value
        $Colour = if ($EndPage -le 12) { "Green" } else { "Yellow" }
        Write-Host "Main body ends on page $EndPage (budget: 12)" -ForegroundColor $Colour
    }
    if ($Open) {
        Start-Process $PdfPath
    }
}
else {
    throw "LaTeX command completed, but no PDF was found at $PdfPath."
}
