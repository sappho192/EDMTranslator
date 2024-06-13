Push-Location
Set-Location -Path ".\nuget"
if (Test-Path -Path "EDMTranslator.*") {
    Remove-Item "EDMTranslator.*" -Force
}
Pop-Location

Push-Location
Set-Location -Path ".\src\EDMTranslator"
if (Test-Path -Path ".\bin") {
    Remove-Item "bin" -Force -Recurse
}
if (Test-Path -Path ".\obj") {
    Remove-Item "obj" -Force -Recurse
}
Pop-Location
