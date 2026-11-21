$ErrorActionPreference = 'Stop'
$taskDirectory = $PSScriptRoot
$runDirectory = (& python -c "import sys; sys.path.insert(0, r'$taskDirectory'); from paths import JOKER_RUN; JOKER_RUN.mkdir(parents=True, exist_ok=True); print(JOKER_RUN)").Trim()
if (Test-Path -LiteralPath (Join-Path $runDirectory 'running.lock')) {
    throw 'A running.lock exists. Check its PID before launching again.'
}
$taskPython = (Get-Command python -ErrorAction Stop).Source
$taskScript = Join-Path $taskDirectory 'train_joker.py'
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', ('"' + $taskScript + '"')) -WorkingDirectory $runDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDirectory 'stdout.log') -RedirectStandardError (Join-Path $runDirectory 'stderr.log') -PassThru
@{ pid = $taskProcess.Id; python = $taskPython; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDirectory 'launcher.json')
Write-Output ('Started context-35 Raven-corpus training, PID ' + $taskProcess.Id)
