# E2 WSL2 bootstrap. Run elevated only after the official MSI has been downloaded.
# Never restarts Windows and never starts model training.
$ErrorActionPreference = 'Stop'
$e2ProjectRoot = Split-Path -Parent $PSScriptRoot
$e2InstallDirectory = Join-Path $e2ProjectRoot 'results/SOTA_Baseline/external/opencarp/preparation'
$e2PackagePath = $null
$e2PackageCandidates = @('downloads/wsl.3.0.1.0.x64.msi', 'downloads/wsl.3.0.1.0.x64.verified.msi')
$e2ExpectedHash = '28b1a0d013640a2ac95898ea705fa186e5b4ff767a1c1b49257161bc106599c6'
$e2StatusPath = Join-Path $e2InstallDirectory 'WSL_INSTALL_STATUS.json'
$e2Status = [ordered]@{ completed = $false; reboot_required = $false; distribution_installed = $false; training_started = $false }
try {
    $e2Principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $e2Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'This bootstrap requires Windows administrator privileges.'
    }
    foreach ($e2RelativeCandidate in $e2PackageCandidates) {
        $e2Candidate = Join-Path $PSScriptRoot $e2RelativeCandidate
        if ((Test-Path -LiteralPath $e2Candidate) -and (Get-Item -LiteralPath $e2Candidate).Length -eq 367669248) {
            if ((Get-FileHash -LiteralPath $e2Candidate -Algorithm SHA256).Hash.ToLowerInvariant() -eq $e2ExpectedHash) {
                $e2PackagePath = $e2Candidate
                break
            }
        }
    }
    if (-not $e2PackagePath) { throw 'A complete official WSL MSI with the expected SHA256 is required.' }
    $e2Signature = Get-AuthenticodeSignature -LiteralPath $e2PackagePath
    if ($e2Signature.Status -ne 'Valid' -or $e2Signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
        throw 'WSL package does not have a valid Microsoft signature.'
    }
    $e2Feature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    $e2Status['virtual_machine_platform_before'] = [string]$e2Feature.State
    if ($e2Feature.State -ne 'Enabled') {
        $e2FeatureResult = Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
        $e2Status.reboot_required = [bool]$e2FeatureResult.RestartNeeded
    }
    $e2MsiLog = Join-Path $e2InstallDirectory 'wsl-msi-install.log'
    $e2MsiProcess = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/i', ('"' + $e2PackagePath + '"'), '/qn', '/norestart', '/L*v', ('"' + $e2MsiLog + '"')) -WindowStyle Hidden -PassThru -Wait
    $e2Status['msi_exit_code'] = $e2MsiProcess.ExitCode
    if ($e2MsiProcess.ExitCode -notin @(0, 3010)) { throw ('WSL MSI failed: ' + $e2MsiProcess.ExitCode) }
    if ($e2MsiProcess.ExitCode -eq 3010) { $e2Status.reboot_required = $true }
    $e2Status.completed = $true
} catch {
    $e2Status['error'] = $_.Exception.Message
} finally {
    $e2Status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $e2StatusPath -Encoding UTF8
}
if (-not $e2Status.completed) { exit 1 }
if ($e2Status.reboot_required) { exit 3010 }
exit 0
