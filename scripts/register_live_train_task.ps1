param(
    [string]$TaskName = "MarketPredictorLiveTrainEvent",
    [string]$ProjectDir = "C:\project\market-predictor",
    [string]$At = "00:45",
    [int]$MinLiveRows = 1000,
    [double]$MinAccuracy = 0.505
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path -LiteralPath $ProjectDir).Path
$RunScript = Join-Path $ProjectDir "scripts\run_live_train_event.ps1"
if (-not (Test-Path -LiteralPath $RunScript)) {
    throw "Live retrain script not found at $RunScript"
}

$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$RunScript`" -ProjectDir `"$ProjectDir`" -MinLiveRows $MinLiveRows -MinAccuracy $MinAccuracy"
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Argument -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10)

$Description = "Guarded market-predictor event-model retrain from historical plus matured live labels. Promotes only if validation gates pass."

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description $Description `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
