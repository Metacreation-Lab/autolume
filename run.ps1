$currentDir = $PSScriptRoot
Push-Location $currentDir
$activatePath = [System.IO.Path]::Combine($currentDir, "venv\Scripts\Activate.ps1")
if ([System.IO.File]::Exists($activatePath)) {
    & $activatePath
    . python main.py
}
else
{
    throw "The project is not installed."
}