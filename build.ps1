Push-Location

Set-Location -Path ".\src\EDMTranslator"
dotnet build --configuration Release
Copy-Item "bin\Release\EDMTranslator.*.nupkg" "..\..\nuget"
Pop-Location
