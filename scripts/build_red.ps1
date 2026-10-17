# Requires lawful rights to build/use the game. No ROM or game assets are downloaded.
# Tool archives and pret source must already be prepared under ignored work/.
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$sourceRoot=Join-Path $projectRoot 'work\pokered'
$toolBin=Join-Path $projectRoot 'work\w64devkit\bin'
$rgbdsRoot=Join-Path $projectRoot 'work\rgbds-1.0.4'
$expectedSource='af519899719f0754965776faac0e836a3b906e6d'
$expectedRom='ea9bcae617fdf159b045185467ae58b2e4a48b9a'
$sourceCommit=(& git -C $sourceRoot rev-parse HEAD).Trim()
if($LASTEXITCODE -ne 0 -or $sourceCommit -ne $expectedSource){throw 'Unexpected pret/pokered source revision'}
$assemblerVersion=(& (Join-Path $rgbdsRoot 'rgbasm.exe') --version).Trim()
if($assemblerVersion -ne 'rgbasm v1.0.4'){throw 'RGBDS 1.0.4 required'}
$previousPath=$env:PATH
try {
    $env:PATH=$toolBin+';'+$rgbdsRoot+';'+$previousPath
    New-Item -ItemType Directory -Force (Join-Path $projectRoot 'artifacts') | Out-Null
    Push-Location -LiteralPath $sourceRoot
    try {
        $prefix=$rgbdsRoot.Replace('\','/')+'/'
        & (Join-Path $toolBin 'make.exe') -j2 red "RGBDS=$prefix" 2>&1 | Tee-Object -FilePath (Join-Path $projectRoot 'artifacts\build_red.log')
        if($LASTEXITCODE -ne 0){throw 'make red failed'}
    } finally { Pop-Location }
    $builtRom=Join-Path $sourceRoot 'pokered.gbc'
    $actual=(Get-FileHash -Algorithm SHA1 -LiteralPath $builtRom).Hash.ToLowerInvariant()
    if($actual -ne $expectedRom){throw "Built ROM SHA1 mismatch: $actual; no ROM copied"}
    $target=Join-Path $projectRoot 'local\PokemonRed.gb'
    if((Test-Path -LiteralPath $target) -and (Get-FileHash -Algorithm SHA1 -LiteralPath $target).Hash.ToLowerInvariant() -ne $expectedRom){
        throw 'Refusing to overwrite an unrelated existing local/PokemonRed.gb'
    }
    Copy-Item -LiteralPath $builtRom -Destination $target
    if((Get-FileHash -Algorithm SHA1 -LiteralPath $target).Hash.ToLowerInvariant() -ne $expectedRom){throw 'Copy hash mismatch'}
    $manifest=[ordered]@{
        source='https://github.com/pret/pokered';commit=$sourceCommit;rgbds=$assemblerVersion
        make=(& (Join-Path $toolBin 'make.exe') --version | Select-Object -First 1)
        gcc=(& (Join-Path $toolBin 'gcc.exe') --version | Select-Object -First 1)
        command='make -j2 red RGBDS=../rgbds-1.0.4/';rom_bytes=(Get-Item -LiteralPath $target).Length
        rom_sha1=$actual;target='local/PokemonRed.gb';bytes_modified=$false
    }
    $manifest | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $projectRoot 'artifacts\rom_build.json')
    Write-Output "Verified and copied English Red: $actual"
} finally { $env:PATH=$previousPath }
