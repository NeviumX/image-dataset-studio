param([switch]$SkipChecks)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
$previousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uvをインストールしてください。' }
    $env:UV_PROJECT_ENVIRONMENT = '.venv-cuda'
    uv sync --extra dev --extra cuda --locked
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
    if (-not $SkipChecks) {
        uv run --no-sync ruff check src tests run_app.py
        if ($LASTEXITCODE -ne 0) { throw 'Lint failed' }
        $previousQtPlatform = $env:QT_QPA_PLATFORM
        $env:QT_QPA_PLATFORM = 'offscreen'
        try {
            uv run --no-sync pytest -q
            if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
        } finally {
            if ($null -eq $previousQtPlatform) {
                Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
            } else {
                $env:QT_QPA_PLATFORM = $previousQtPlatform
            }
        }
    }
    uv run --no-sync python -m PyInstaller --noconfirm ImageDatasetStudio.spec
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
    Write-Host '配布先: dist\ImageDatasetStudio（フォルダー全体を配布してください）'
} finally {
    if ($null -eq $previousProjectEnvironment) {
        Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    } else {
        $env:UV_PROJECT_ENVIRONMENT = $previousProjectEnvironment
    }
    Pop-Location
}
