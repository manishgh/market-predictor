[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [ValidatePattern('^([01][0-9]|2[0-3]):[0-5][0-9]$')][string]$At = "00:00"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectDir).Path
$runner = Join-Path $root "scripts\run_swing_data_collection.ps1"
foreach ($required in @($runner, (Join-Path $root ".venv\Scripts\python.exe"),
    (Join-Path $root "configs\swing_incremental_collection.toml"),
    (Join-Path $root "configs\swing_incremental_sec.toml"))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Cannot install collection task; missing file: $required"
    }
}
$taskName = "MarketPredictorSwingCollectionMidnight"
$legacyNames = @("MarketPredictorLiveMidnight", "MarketPredictorLiveTrainEvent")
$existing = @(Get-ScheduledTask | Where-Object {
    $_.TaskName -in ($legacyNames + @($taskName)) -and $_.TaskPath -eq "\"
})
$legacy = @($existing | Where-Object { $_.TaskName -in $legacyNames })

function Read-ActionPath([string]$Arguments, [string]$Option) {
    $pattern = '(?i)(?:^|\s)-' + [regex]::Escape($Option) + '\s+(?:"([^"]+)"|''([^'']+)''|(\S+))(?=\s|$)'
    $matchesFound = [regex]::Matches($Arguments, $pattern)
    if ($matchesFound.Count -ne 1) { throw "Task must contain exactly one -$Option path." }
    foreach ($group in $matchesFound[0].Groups | Select-Object -Skip 1) {
        if ($group.Success) { return [IO.Path]::GetFullPath($group.Value) }
    }
    throw "Task -$Option path is missing."
}

foreach ($task in $existing) {
    if ($task.State -eq "Running") { throw "Stop the existing $($task.TaskName) job before migration." }
    if (@($task.Actions).Count -ne 1) { throw "Task action count differs: $($task.TaskName)" }
    $expectedScript = switch ($task.TaskName) {
        "MarketPredictorLiveMidnight" { Join-Path $root "scripts\run_live_midnight.ps1" }
        "MarketPredictorLiveTrainEvent" { Join-Path $root "scripts\run_live_train_event.ps1" }
        default { $runner }
    }
    $actionRoot = Read-ActionPath $task.Actions[0].Arguments "ProjectDir"
    $actionScript = Read-ActionPath $task.Actions[0].Arguments "File"
    if (-not $root.Equals($actionRoot, [StringComparison]::OrdinalIgnoreCase) -or
        -not $expectedScript.Equals($actionScript, [StringComparison]::OrdinalIgnoreCase) -or
        -not $root.Equals([IO.Path]::GetFullPath($task.Actions[0].WorkingDirectory), [StringComparison]::OrdinalIgnoreCase)) {
        throw "Existing $($task.TaskName) is not owned by $root; inspect it manually."
    }
}

if ($PSCmdlet.ShouldProcess($root, "Install source-only collection and retire the two broken legacy tasks")) {
    $backup = Join-Path $root ("data\runtime\scheduler_migrations\" + (Get-Date -Format "yyyyMMddTHHmmss"))
    New-Item -ItemType Directory -Path $backup -Force | Out-Null
    foreach ($task in $existing) {
        Export-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath |
            Out-File -LiteralPath (Join-Path $backup ($task.TaskName + ".xml")) -Encoding utf8
    }
    $action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`" -ProjectDir `"$root`"" `
        -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Daily -At $At
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 30) -ExecutionTimeLimit (New-TimeSpan -Hours 4)
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -TaskPath "\" -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Description "Source-only Alpaca swing news, SIP daily bars and SEC filings. No training." `
        -Force | Out-Null
    $installed = Get-ScheduledTask -TaskName $taskName -TaskPath "\"
    if ($installed.Actions[0].Arguments -ne $action.Arguments) { throw "New task verification failed." }
    foreach ($task in $legacy) {
        Unregister-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath -Confirm:$false
    }
    Write-Output "Installed $taskName at $At local time. Migration backup: $backup"
    Write-Output "Runs while this Windows user is signed in; missed runs catch up when available. Requires AC power."
}
