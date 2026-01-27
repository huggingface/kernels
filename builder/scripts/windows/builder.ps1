#Requires -Version 7.0

<#
.SYNOPSIS
    Kernel Builder - Modern PowerShell wrapper for build2cmake tool

.DESCRIPTION
    This script provides a modular interface to build2cmake for generating CMake
    structures from build.toml configuration files. Supports multiple backends
    including CUDA, ROCm, Metal, and XPU.

.PARAMETER SourceFolder
    Path to the folder containing build.toml file

.PARAMETER TargetFolder
    Optional destination folder for generated CMake files (defaults to SourceFolder)

.PARAMETER Backend
    Target backend: cuda, rocm, metal, xpu, or universal

.PARAMETER Build2CmakePath
    Path to build2cmake executable (auto-detected if not specified)

.PARAMETER Force
    Force overwrite existing files without prompting

.PARAMETER OpsId
    Optional unique identifier suffixed to kernel names (e.g., Git SHA)

.PARAMETER Clean
    Remove generated artifacts instead of building

.PARAMETER DryRun
    Show what would be done without executing (clean mode only)

.PARAMETER Validate
    Validate build.toml without generating files

.PARAMETER Build
    Build the project after generating CMake files

.PARAMETER BuildConfig
    CMake build configuration (Debug or Release, defaults to Release)

.PARAMETER ArchList
    GPU architectures to build for (backend-agnostic).
    For CUDA: e.g., "7.5 8.6" or "Turing Ampere"
    For ROCm: e.g., "gfx906;gfx908;gfx90a"
    For XPU: Currently not supported via environment variable

.PARAMETER LocalInstall
    Run CMake install target after building (installs to build/<variant>/<package>/ for local development)

.PARAMETER KernelsInstall
    Run kernels_install target after building (installs to CMAKE_INSTALL_PREFIX/<variant>/<package>/)

.PARAMETER InstallPrefix
    Installation prefix for kernels_install target (defaults to CMAKE_INSTALL_PREFIX)

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -Backend cuda -Force

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -TargetFolder ./build/relu -OpsId abc123

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -Clean -Force

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -Backend cuda -Build -BuildConfig Debug

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -Backend cuda -ArchList "7.5 8.6" -Build

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -Backend rocm -ArchList "gfx906;gfx908" -Build

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -Backend cuda -Build -LocalInstall

.EXAMPLE
    .\builder.ps1 -SourceFolder ./examples/relu -Backend cuda -Build -KernelsInstall -InstallPrefix "C:\kernels"
#>

[CmdletBinding(DefaultParameterSetName = 'Generate')]
param(
    [Parameter(Mandatory = $true, Position = 0, HelpMessage = "Folder containing build.toml")]
    [ValidateScript({ Test-Path $_ -PathType Container })]
    [string]$SourceFolder,

    [Parameter(ParameterSetName = 'Generate')]
    [ValidateScript({
        if ($_ -and !(Test-Path $_ -PathType Container)) {
            throw "Target folder does not exist: $_"
        }
        $true
    })]
    [string]$TargetFolder,

    [Parameter(ParameterSetName = 'Generate')]
    [ValidateSet('cuda', 'rocm', 'metal', 'xpu', 'universal')]
    [string]$Backend,

    [Parameter()]
    [string]$Build2CmakePath,

    [Parameter(ParameterSetName = 'Generate')]
    [switch]$Force,

    [Parameter(ParameterSetName = 'Generate')]
    [string]$OpsId,

    [Parameter(ParameterSetName = 'Generate')]
    [switch]$Build,

    [Parameter(ParameterSetName = 'Generate')]
    [ValidateSet('Debug', 'Release')]
    [string]$BuildConfig = 'Release',

    [Parameter(ParameterSetName = 'Generate')]
    [string]$ArchList,

    [Parameter(ParameterSetName = 'Generate')]
    [switch]$LocalInstall,

    [Parameter(ParameterSetName = 'Generate')]
    [switch]$KernelsInstall,

    [Parameter(ParameterSetName = 'Generate')]
    [string]$InstallPrefix,

    [Parameter(ParameterSetName = 'Clean', Mandatory = $true)]
    [switch]$Clean,

    [Parameter(ParameterSetName = 'Clean')]
    [switch]$DryRun,

    [Parameter(ParameterSetName = 'Validate', Mandatory = $true)]
    [switch]$Validate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

#region Helper Functions

function Write-Status {
    param([string]$Message, [string]$Type = 'Info')

    $colors = @{
        'Info'    = 'Cyan'
        'Success' = 'Green'
        'Warning' = 'Yellow'
        'Error'   = 'Red'
    }

    $prefix = switch ($Type) {
        'Info'    { '[*]' }
        'Success' { '[+]' }
        'Warning' { '[!]' }
        'Error'   { '[X]' }
    }

    Write-Host "$prefix $Message" -ForegroundColor $colors[$Type]
}

function Find-Build2Cmake {
    <#
    .SYNOPSIS
        Locates build2cmake executable in common locations
    #>

    # Check if provided path is valid
    if ($Build2CmakePath) {
        if (Test-Path $Build2CmakePath -PathType Leaf) {
            return $Build2CmakePath
        }
        throw "Specified build2cmake path not found: $Build2CmakePath"
    }

    # Search common locations
    $searchPaths = @(
        (Join-Path $PSScriptRoot '..' '..' '..' 'build2cmake' 'target' 'release' 'build2cmake.exe'),
        (Join-Path $PSScriptRoot '..' '..' '..' 'build2cmake' 'target' 'debug' 'build2cmake.exe'),
        'build2cmake.exe',
        'build2cmake'
    )

    foreach ($path in $searchPaths) {
        $resolved = if ([System.IO.Path]::IsPathRooted($path)) {
            $path
        } else {
            Join-Path $PWD $path
        }

        if (Test-Path $resolved -PathType Leaf) {
            Write-Status "Found build2cmake at: $resolved" -Type Info
            return $resolved
        }
    }

    # Try system PATH
    $cmd = Get-Command build2cmake -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Status "Using build2cmake from PATH: $($cmd.Source)" -Type Info
        return $cmd.Source
    }

    throw "build2cmake executable not found. Please build it or specify -Build2CmakePath"
}

function Get-BuildTomlPath {
    param([string]$Folder)

    $buildTomlPath = Join-Path $Folder 'build.toml'

    if (!(Test-Path $buildTomlPath -PathType Leaf)) {
        throw "build.toml not found in folder: $Folder"
    }

    return $buildTomlPath
}

function Invoke-Build2Cmake {
    param(
        [string]$Build2CmakeExe,
        [string[]]$Arguments
    )

    Write-Status "Executing: $Build2CmakeExe $($Arguments -join ' ')" -Type Info

    & $Build2CmakeExe @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "build2cmake failed with exit code $LASTEXITCODE"
    }
}

function Import-EnvironmentVariables {
    <#
    .SYNOPSIS
        Imports environment variables from a file
    #>
    param([string]$FilePath)

    Get-Content $FilePath | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
}

function Initialize-VSEnvironment {
    <#
    .SYNOPSIS
        Initializes Visual Studio build environment for MSBuild/CMake
    #>

    Write-Status "Initializing Visual Studio environment..." -Type Info

    # Check if already in VS environment
    if ($env:VSINSTALLDIR) {
        Write-Status "Visual Studio environment already initialized" -Type Info
        return
    }

    # Search for vswhere.exe
    $vswherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (!(Test-Path $vswherePath)) {
        throw "vswhere.exe not found. Please install Visual Studio 2017 or later."
    }

    # Find latest VS installation
    $vsPath = & $vswherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vsPath) {
        throw "Visual Studio with C++ tools not found. Please install Visual Studio with C++ workload."
    }

    Write-Status "Found Visual Studio at: $vsPath" -Type Info

    # Find vcvarsall.bat
    $vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (!(Test-Path $vcvarsPath)) {
        throw "vcvarsall.bat not found at expected location: $vcvarsPath"
    }

    # Execute vcvarsall and capture environment variables
    $tempFile = [System.IO.Path]::GetTempFileName()

    # Detect platform architecture
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
    $vcvarsArch = if ($arch -eq 'Arm64') { 'arm64' } else { 'x64' }

    Write-Status "Initializing VS environment for $vcvarsArch architecture" -Type Info

    # Run vcvarsall.bat and export environment to temp file
    cmd /c "`"$vcvarsPath`" $vcvarsArch && set > `"$tempFile`""

    if ($LASTEXITCODE -ne 0) {
        Remove-Item $tempFile -ErrorAction SilentlyContinue
        throw "Failed to initialize Visual Studio environment"
    }

    # Parse and apply environment variables
    Import-EnvironmentVariables -FilePath $tempFile

    Remove-Item $tempFile -ErrorAction SilentlyContinue

    Write-Status "Visual Studio environment initialized successfully" -Type Success
}

function Get-CMakeConfigureArgs {
    <#
    .SYNOPSIS
        Builds CMake configuration arguments
    #>
    param(
        [bool]$ShouldInstall,
        [string]$InstallPrefix,
        [string]$Backend
    )

    # For XPU backend, use Ninja generator with Intel compilers
    if ($Backend -and $Backend.ToLower() -eq 'xpu') {
        Write-Status "Using Ninja generator for XPU backend with Intel SYCL compilers" -Type Info
        
        $kwargs = @("..", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release")
        
        # Verify Intel compilers are available (CMakeLists.txt will set them correctly)
        $icx = Get-Command icx -ErrorAction SilentlyContinue
        
        if ($icx) {
            Write-Status "Found Intel compiler: $($icx.Source)" -Type Info
            Write-Status "CMakeLists.txt will configure icx for Windows (MSVC-compatible mode)" -Type Info
        } else {
            Write-Status "Intel compilers not found in PATH. Make sure oneAPI environment is initialized." -Type Error
            Write-Status "Run: & `"C:\Program Files (x86)\Intel\oneAPI\setvars.bat`"" -Type Error
            throw "Intel compilers (icx) are required for XPU backend but were not found in PATH. Please initialize oneAPI environment."
        }
    } else {
        # Use Visual Studio generator for other backends
        $arch = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
        $vsArch = if ($arch -eq 'Arm64') { 'ARM64' } else { 'x64' }

        Write-Status "Detected platform: $arch, using Visual Studio architecture: $vsArch" -Type Info

        $kwargs = @("..", "-G", "Visual Studio 17 2022", "-A", $vsArch)
    }

    # Build for all supported GPU archs, not just the detected arch.
    $kwargs += "-DBUILD_ALL_SUPPORTED_ARCHS=ON"

    # Detect Python from current environment
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pythonExe) {
        $kwargs += "-DPython3_EXECUTABLE=$pythonExe"
        Write-Status "Using Python from environment: $pythonExe" -Type Info

        # Verify Python and PyTorch version
        $torchVersion = & $pythonExe -c "import torch; print(torch.__version__)" 2>$null
        if ($torchVersion) {
            Write-Status "Detected PyTorch version: $torchVersion" -Type Info
        }
    } else {
        Write-Status "Python not found in PATH, CMake will auto-detect" -Type Warning
    }

    if ($ShouldInstall -and $InstallPrefix) {
        $kwargs += "-DCMAKE_INSTALL_PREFIX=$InstallPrefix"
        Write-Status "Setting CMAKE_INSTALL_PREFIX=$InstallPrefix" -Type Info
    }

    return $kwargs
}

function Invoke-CMakeTarget {
    <#
    .SYNOPSIS
        Executes a CMake build target
    #>
    param(
        [string]$Target,
        [string]$BuildConfig,
        [string]$DisplayName
    )

    Write-Status "Running $DisplayName..." -Type Info

    # Use cmake --build with proper escaping for Visual Studio generator
    cmake --build . --config $BuildConfig --target $Target

    if ($LASTEXITCODE -ne 0) {
        throw "$DisplayName failed with exit code $LASTEXITCODE"
    }

    Write-Status "$DisplayName completed successfully!" -Type Success
}

function Invoke-CMakeBuild {
    param(
        [string]$SourcePath,
        [string]$BuildConfig,
        [bool]$RunLocalInstall = $false,
        [bool]$RunKernelsInstall = $false,
        [string]$InstallPrefix = $null,
        [string]$Backend = $null
    )

    Write-Status "Building project with CMake..." -Type Info
    Write-Status "Configuration: $BuildConfig" -Type Info

    # Ensure VS environment is initialized (needed for Ninja and MSVC)
    Initialize-VSEnvironment

    # Create build directory
    $buildDir = Join-Path $SourcePath "build"
    if (!(Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
        Write-Status "Created build directory: $buildDir" -Type Info
    }

    # Configure with CMake
    Write-Status "Configuring CMake project..." -Type Info
    Push-Location $buildDir
    try {
        $configureArgs = Get-CMakeConfigureArgs -ShouldInstall ($RunKernelsInstall -or $RunLocalInstall) -InstallPrefix $InstallPrefix -Backend $Backend

        cmake @configureArgs

        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed with exit code $LASTEXITCODE"
        }

        # Build with CMake
        Write-Status "Building project..." -Type Info
        cmake --build . --config $BuildConfig

        if ($LASTEXITCODE -ne 0) {
            throw "CMake build failed with exit code $LASTEXITCODE"
        }

        Write-Status "Build completed successfully!" -Type Success

        # Run install target if requested
        if ($RunLocalInstall) {
            Invoke-CMakeTarget -Target 'install' -BuildConfig $BuildConfig -DisplayName 'install target (local development layout)'
        }

        if ($RunKernelsInstall) {
            Invoke-CMakeTarget -Target 'install' -BuildConfig $BuildConfig -DisplayName 'install target'
        }
    }
    finally {
        Pop-Location
    }
}

#endregion

#region Backend-Specific Functions

function Invoke-Backend {
    <#
    .SYNOPSIS
        Generates CMake files for specified backend
    #>
    param(
        [string]$Build2CmakeExe,
        [string]$BuildToml,
        [string]$Target,
        [hashtable]$Options,
        [string]$Backend
    )

    $backendName = if ($Backend -eq 'universal') { 'Universal' } else { $Backend.ToUpper() }
    Write-Status "Generating $backendName backend..." -Type Info

    $kwargs = @('generate-torch', $BuildToml)

    if ($Target) { $kwargs += $Target }
    if ($Options.Force) { $kwargs += '--force' }
    if ($Options.OpsId) { $kwargs += '--ops-id', $Options.OpsId }
    if ($Backend -and $Backend -ne 'universal') { $kwargs += '--backend', $Backend }

    Invoke-Build2Cmake -Build2CmakeExe $Build2CmakeExe -Arguments $kwargs
}

function Set-BackendArchitecture {
    <#
    .SYNOPSIS
        Configures backend-specific architecture environment variables
    #>
    param(
        [string]$Backend,
        [string]$ArchList
    )

    $archMappings = @{
        'cuda' = @{ Env = 'TORCH_CUDA_ARCH_LIST'; Supported = $true }
        'rocm' = @{ Env = 'PYTORCH_ROCM_ARCH'; Supported = $true }
        'xpu'  = @{ Env = $null; Supported = $false; Message = 'no standard environment variable' }
    }

    if ($mapping = $archMappings[$Backend.ToLower()]) {
        if ($mapping.Supported) {
            Set-Item "env:$($mapping.Env)" -Value $ArchList
            Write-Status "Set $($mapping.Env)=$ArchList" -Type Info
        } else {
            Write-Status "ArchList not supported for $Backend backend ($($mapping.Message))" -Type Warning
        }
    } else {
        Write-Status "ArchList not applicable for $Backend backend" -Type Warning
    }
}

#endregion

#region Main Logic

try {
    # Resolve paths
    $SourceFolder = Resolve-Path $SourceFolder -ErrorAction Stop
    $buildTomlPath = Get-BuildTomlPath -Folder $SourceFolder
    $build2cmakeExe = Find-Build2Cmake

    # Validate mode
    if ($Validate) {
        Write-Status "Validating $buildTomlPath..." -Type Info
        Invoke-Build2Cmake -Build2CmakeExe $build2cmakeExe -Arguments @('validate', $buildTomlPath)
        Write-Status "Validation successful!" -Type Success
        exit 0
    }

    # Clean mode
    if ($Clean) {
        Write-Status "Cleaning generated artifacts..." -Type Warning

        $kwargs = @('clean', $buildTomlPath)
        if ($TargetFolder) { $kwargs += $TargetFolder }
        if ($DryRun) { $kwargs += '--dry-run' }
        if ($Force) { $kwargs += '--force' }
        if ($OpsId) { $kwargs += '--ops-id', $OpsId }

        Invoke-Build2Cmake -Build2CmakeExe $build2cmakeExe -Arguments $kwargs
        Write-Status "Clean completed!" -Type Success
        exit 0
    }

    # Generate mode
    # Check for Metal backend on Windows
    if ($Backend -and $Backend.ToLower() -eq 'metal') {
        throw "Metal backend is not supported on Windows. Metal is only available on macOS."
    }

    $options = @{
        Force = $Force.IsPresent
        OpsId = $OpsId
    }

    # Set architecture environment variables if ArchList is provided
    if ($ArchList -and $Backend) {
        Set-BackendArchitecture -Backend $Backend -ArchList $ArchList
    }

    # Determine backend strategy
    if ($Backend) {
        # Explicit backend specified
        $targetPath = if ($TargetFolder) { Resolve-Path $TargetFolder } else { $null }
        Invoke-Backend -Build2CmakeExe $build2cmakeExe -BuildToml $buildTomlPath -Target $targetPath -Options $options -Backend $Backend.ToLower()
    } else {
        # Auto-detect backend from build.toml
        Write-Status "Auto-detecting backend from build.toml..." -Type Info

        $kwargs = @('generate-torch', $buildTomlPath)
        if ($TargetFolder) { $kwargs += (Resolve-Path $TargetFolder) }
        if ($Force) { $kwargs += '--force' }
        if ($OpsId) { $kwargs += '--ops-id', $OpsId }

        Invoke-Build2Cmake -Build2CmakeExe $build2cmakeExe -Arguments $kwargs
    }

    Write-Status "Generation completed successfully!" -Type Success

    # Build if requested (skip if no CMakeLists.txt exists, e.g., universal backend)
    if ($Build) {
        $buildPath = if ($TargetFolder) { $TargetFolder } else { $SourceFolder }
        $cmakeListsPath = Join-Path $buildPath "CMakeLists.txt"

        if (!(Test-Path $cmakeListsPath -PathType Leaf)) {
            Write-Status "No CMakeLists.txt found, skipping build (likely universal backend)" -Type Info
        } else {
            Invoke-CMakeBuild `
                -SourcePath $buildPath `
                -BuildConfig $BuildConfig `
                -RunLocalInstall $LocalInstall.IsPresent `
                -RunKernelsInstall $KernelsInstall.IsPresent `
                -InstallPrefix $InstallPrefix `
                -Backend $Backend
        }
    }

} catch {
    Write-Status "Error: $_" -Type Error
    exit 1
}

#endregion
