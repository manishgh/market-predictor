[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string]$DataRoot,
    [ValidatePattern('^([01][0-9]|2[0-3]):[0-5][0-9]$')][string]$At = "22:35"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectDir).Path
$runner = Join-Path $root "scripts\run_forward_research_scoring.ps1"
$bundleCheckRoot = if ($DataRoot) { (Resolve-Path -LiteralPath $DataRoot).Path } else { $root }
foreach ($required in @(
    $runner,
    (Join-Path $root ".venv\Scripts\python.exe"),
    (Join-Path $bundleCheckRoot "data\research\research_lane\models\mid_small_beta_neutral_sec_xgb\_manifest.json")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Cannot install forward scoring task; missing file: $required"
    }
}

$taskName = "MarketPredictorForwardScoringPostClose"
$existing = @(Get-ScheduledTask | Where-Object { $_.TaskName -eq $taskName -and $_.TaskPath -eq "\" })
foreach ($task in $existing) {
    if ($task.State -eq "Running") {
        throw "Stop the running $($task.TaskName) job before updating its schedule."
    }
}

if ($PSCmdlet.ShouldProcess($root, "Install weekday post-close prospective forward scoring and outcome maturation task")) {
    $backup = Join-Path $root ("data\runtime\scheduler_migrations\" + (Get-Date -Format "yyyyMMddTHHmmss"))
    New-Item -ItemType Directory -Path $backup -Force | Out-Null
    foreach ($task in $existing) {
        Export-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath |
            Out-File -LiteralPath (Join-Path $backup ($task.TaskName + ".xml")) -Encoding utf8
    }

    $arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`" -ProjectDir `"$root`""
    if ($DataRoot) {
        $storage = (Resolve-Path -LiteralPath $DataRoot).Path
        if (-not (Test-Path -LiteralPath $storage -PathType Container)) {
            throw "Forward scoring data root is not a directory: $storage"
        }
        $arguments += " -DataRoot `"$storage`""
    }

    $action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Argument $arguments -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $At
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 30) -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskName $taskName -TaskPath "\" -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal `
        -Description "Weekday post-close prospective SEC catalyst XGBoost + OPRA straddle/skew scoring and outcome maturation." `
        -Force | Out-Null

    $installed = Get-ScheduledTask -TaskName $taskName -TaskPath "\"
    if ($installed.Actions[0].Arguments -ne $action.Arguments) {
        throw "New task verification failed for $taskName."
    }
    Write-Output "Installed $taskName (Mon-Fri at $At local time). Backup: $backup"
    Write-Output "Runs while this Windows user is signed in; missed runs catch up automatically when available."
}
