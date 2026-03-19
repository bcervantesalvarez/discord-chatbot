# start-chatbot.ps1 -- Watchdog script for Discord Chatbot
# Restarts the bot on crash with exponential backoff.
# Graceful shutdown on Ctrl+C or when the console is closed.

param(
    [int]$MaxBackoff = 60,        # max seconds between restart attempts
    [int]$BaseDelay  = 2,         # initial restart delay
    [int]$SuccessReset = 120      # seconds of uptime before resetting backoff
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BotScript = Join-Path $ScriptDir "bot.py"
$LogDir    = Join-Path $ScriptDir "logs"
$LogFile   = Join-Path $LogDir "watchdog.log"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts | WATCHDOG | $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

# Find python executable -- prefer venv if it exists
$VenvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
    Write-Log "Using venv Python: $Python"
} else {
    $Python = $null
    foreach ($candidate in @("python", "python3", "py")) {
        try {
            $ver = & $candidate --version 2>&1
            if ($ver -match "Python 3") {
                $Python = $candidate
                break
            }
        } catch {}
    }
    if (-not $Python) {
        Write-Log "ERROR: Python 3 not found on PATH. Exiting."
        exit 1
    }
    Write-Log "Using system Python: $Python"
}
Write-Log "Python version: $(& $Python --version 2>&1)"

# Track the bot process for graceful shutdown
$script:BotProcess = $null
$script:ShuttingDown = $false

# Register cleanup on Ctrl+C / console close
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    $script:ShuttingDown = $true
    if ($script:BotProcess -and -not $script:BotProcess.HasExited) {
        Write-Log "Shutting down Discord Chatbot (PID $($script:BotProcess.Id))..."
        try {
            $script:BotProcess.Kill()
            $script:BotProcess.WaitForExit(10000)
        } catch {}
    }
}

try {
    [Console]::TreatControlCAsInput = $false
} catch {}

trap {
    $script:ShuttingDown = $true
    if ($script:BotProcess -and -not $script:BotProcess.HasExited) {
        Write-Log "Interrupt received. Stopping Discord Chatbot (PID $($script:BotProcess.Id))..."
        try { $script:BotProcess.Kill() } catch {}
    }
    Write-Log "Watchdog exiting."
    break
}

# ---- Main restart loop -------------------------------------------------------

$delay = $BaseDelay
$restartCount = 0

Write-Log "Watchdog started. Bot script: $BotScript"

while (-not $script:ShuttingDown) {
    $restartCount++
    Write-Log "Starting Discord Chatbot (attempt #$restartCount)..."

    $startTime = Get-Date

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Python
    $psi.Arguments = "`"$BotScript`""
    $psi.WorkingDirectory = $ScriptDir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError  = $false

    try {
        $script:BotProcess = [System.Diagnostics.Process]::Start($psi)
        Write-Log "Discord Chatbot running (PID $($script:BotProcess.Id))"
        $script:BotProcess.WaitForExit()
        $exitCode = $script:BotProcess.ExitCode
    } catch {
        $exitCode = -1
        Write-Log "ERROR: Failed to start bot: $_"
    }

    if ($script:ShuttingDown) { break }

    $uptime = ((Get-Date) - $startTime).TotalSeconds
    Write-Log "Discord Chatbot exited with code $exitCode after $([math]::Round($uptime, 1))s"

    # If bot ran long enough, reset backoff
    if ($uptime -ge $SuccessReset) {
        $delay = $BaseDelay
        Write-Log "Uptime exceeded ${SuccessReset}s -- backoff reset."
    }

    # Exit cleanly on code 0 (intentional shutdown via !shutdown or similar)
    if ($exitCode -eq 0) {
        Write-Log "Clean exit. Watchdog stopping."
        break
    }

    Write-Log "Restarting in ${delay}s..."
    Start-Sleep -Seconds $delay

    # Exponential backoff: 2 -> 4 -> 8 -> 16 -> 32 -> 60 (capped)
    $delay = [math]::Min($delay * 2, $MaxBackoff)
}

Write-Log "Watchdog stopped."
