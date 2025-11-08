param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("build", "stop", "remove")]
    [string]$Action,

    [Parameter(Mandatory=$true)]
    [string]$ContainerName,

    [string]$ImageName = "$ContainerName-image"
)

function Build-And-Run {
    Write-Host "Building Docker image '$ImageName'..."
    docker build -t $ImageName .

    Write-Host "Running container '$ContainerName'..."
    docker run -d -p 8000:8000 --name $ContainerName $ImageName
    Write-Host "Container '$ContainerName' is running."
}

function Stop-Container {
    docker stop $ContainerName
    Write-Host "Stopped container '$ContainerName'"
}

function Remove-Container {
    docker rm $ContainerName
    Write-Host "Removed container '$ContainerName'"
}

# Check if container exists
$exists = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }

switch ($Action) {
    "build" {
        if ($exists) {
            Write-Host "Container '$ContainerName' already exists. Stopping and removing it first..."
            Stop-Container
            Remove-Container
        }
        Build-And-Run
    }
    "stop" {
        if (-not $exists) {
            Write-Host "Container '$ContainerName' does not exist." -ForegroundColor Red
            exit 1
        }
        Stop-Container
    }
    "remove" {
        if (-not $exists) {
            Write-Host "Container '$ContainerName' does not exist." -ForegroundColor Red
            exit 1
        }
        Remove-Container
    }
}