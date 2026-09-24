param(
    [Parameter(Mandatory=$true)][string]$Pacote,
    [Parameter(Mandatory=$true)][string]$Saida,
    [Parameter(Mandatory=$true)][double]$Tarifa,
    [Parameter(Mandatory=$true)][ValidateSet(220,380)][int]$Tensao,
    [double]$Autonomia = 6,
    [ValidateSet(30,50,100)][int]$Disponibilidade = 50,
    [string]$Python = 'C:/Users/mathe/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
)
$ErrorActionPreference = 'Stop'
$pacePacote = (Resolve-Path -LiteralPath $Pacote).Path
$paceSaida = [System.IO.Path]::GetFullPath($Saida)
$pacePythonPath = $env:PYTHONPATH
Push-Location $PSScriptRoot
try {
    $env:PYTHONPATH = (Join-Path $PSScriptRoot '.runtime-pace')
    & $Python -m aurum.estudo_coleta $pacePacote --saida $paceSaida --tarifa $Tarifa --tensao $Tensao --autonomia $Autonomia --disponibilidade $Disponibilidade
    if ($LASTEXITCODE -ne 0) { throw 'O estudo falhou. Consulte execucao.json na pasta de saída, se criada.' }
} finally {
    $env:PYTHONPATH = $pacePythonPath
    Pop-Location
}
