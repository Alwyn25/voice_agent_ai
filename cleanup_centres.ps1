# Deactivate the leftover test centres so locate_centre has a sane list to choose from.
#
# There is no DELETE for centres — only PATCH is_active — so this deactivates rather than
# removes. Deactivated centres stay out of availability and out of the agent's choices,
# but their history and appointments remain intact, which is the behaviour you want.

$ErrorActionPreference = "Stop"
$BaseUrl = "http://127.0.0.1:8000"
$AdminEmail = "rahul@advora.ai"
$AdminPassword = "rahul12345"
$KeepPrefix = "Motorworks"

$LoginBody = @{ email = $AdminEmail; password = $AdminPassword } | ConvertTo-Json
$Login = Invoke-RestMethod -Method Post -Uri "$BaseUrl/auth/login" -Body $LoginBody -ContentType "application/json"
$AuthH = @{ Authorization = "Bearer $($Login.access_token)" }

$Centres = Invoke-RestMethod -Uri "$BaseUrl/scheduling/centres" -Headers $AuthH
$Stale = $Centres | Where-Object { $_.name -notlike "$KeepPrefix*" -and $_.is_active }

if (-not $Stale) {
    Write-Host "Nothing to deactivate." -ForegroundColor Green
    exit 0
}

Write-Host "`nDeactivating $($Stale.Count) leftover centre(s):" -ForegroundColor Yellow
foreach ($Centre in $Stale) {
    Invoke-RestMethod -Method Patch -Uri "$BaseUrl/scheduling/centres/$($Centre.id)" -Headers $AuthH `
        -Body (@{ is_active = $false } | ConvertTo-Json) -ContentType "application/json" | Out-Null
    Write-Host "  - $($Centre.name)" -ForegroundColor DarkGray
}

$Remaining = (Invoke-RestMethod -Uri "$BaseUrl/scheduling/centres" -Headers $AuthH) |
    Where-Object { $_.is_active }
Write-Host "`nActive centres now:" -ForegroundColor Cyan
$Remaining | Format-Table name, city, region_code
