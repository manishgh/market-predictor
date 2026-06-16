param(
    [string]$TaskName = "MarketPredictorLiveMidnight",
    [string]$ProjectDir = "C:\project\market-predictor",
    [string]$At = "00:00",
    [string]$Tickers = "",
    [int]$LookbackDays = 3,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path
$RunScript = Join-Path $ProjectDir "scripts\run_live_midnight.ps1"
if (-not (Test-Path -LiteralPath $RunScript)) {
    throw "Live run script not found at $RunScript"
}

$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScript`" -ProjectDir `"$ProjectDir`" -LookbackDays $LookbackDays -Workers $Workers"
if ($Tickers.Trim().Length -gt 0) {
    $Argument += " -Tickers `"$Tickers`""
}

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Argument -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10)

$Description = "Runs market-predictor live-once daily: collect latest events, validate, score sentiment, refresh features, score models, and curate matured labels."

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description $Description `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
