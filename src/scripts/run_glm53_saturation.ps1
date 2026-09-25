$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$secretPath = Join-Path $env:USERPROFILE '.codex\secrets\sdu-glm53.dpapi'
$previousKey = $env:SDU_API_KEY
try {
    if (-not $env:SDU_API_KEY) {
        $secure = Get-Content -LiteralPath $secretPath | ConvertTo-SecureString
        $env:SDU_API_KEY = [System.Net.NetworkCredential]::new('', $secure).Password
    }
    & "$repo\.venv\Scripts\python.exe" -X utf8 "$PSScriptRoot\glm53_saturation.py" @args
    if ($LASTEXITCODE -ne 0) { throw "Evaluation exited with code $LASTEXITCODE" }
} finally {
    $env:SDU_API_KEY = $previousKey
}
