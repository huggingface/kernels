# shellcheck shell=bash

# Based on setup-cuda-hook.

# Only run the hook from nativeBuildInputs
(( "$hostOffset" == -1 && "$targetOffset" == 0)) || return 0

guard=Sourcing
reason=

[[ -n ${xpuSetupHookOnce-} ]] && guard=Skipping && reason=" because the hook has been propagated more than once"

if (( "${NIX_DEBUG:-0}" >= 1 )) ; then
    echo "$guard hostOffset=$hostOffset targetOffset=$targetOffset setup-xpu-hook$reason" >&2
else
    echo "$guard setup-xpu-hook$reason" >&2
fi

[[ "$guard" = Sourcing ]] || return 0

declare -g xpuSetupHookOnce=1
declare -Ag xpuHostPathsSeen=()
declare -Ag xpuOutputToPath=()

extendXpuHostPathsSeen() {
    (( "${NIX_DEBUG:-0}" >= 1 )) && echo "extendXpuHostPathsSeen $1" >&2

    local markerPath="$1/nix-support/include-in-xpu-root"
    [[ ! -f "${markerPath}" ]] && return 0
    [[ -v xpuHostPathsSeen[$1] ]] && return 0

    xpuHostPathsSeen["$1"]=1

    # E.g. cuda_cudart-lib
    local xpuOutputName
    # Fail gracefully if the file is empty.
    # One reason the file may be empty: the package was built with strictDeps set, but the current build does not have
    # strictDeps set.
    read -r xpuOutputName < "$markerPath" || return 0

    [[ -z "$xpuOutputName" ]] && return 0

    local oldPath="${xpuOutputToPath[$xpuOutputName]-}"
    [[ -n "$oldPath" ]] && echo "extendXpuHostPathsSeen: warning: overwriting $xpuOutputName from $oldPath to $1" >&2
    xpuOutputToPath["$xpuOutputName"]="$1"
}
addEnvHooks "$targetOffset" extendXpuHostPathsSeen

propagateXpuLibraries() {
    (( "${NIX_DEBUG:-0}" >= 1 )) && echo "propagateXpuLibraries: xpuPropagateToOutput=$xpuPropagateToOutput xpuHostPathsSeen=${!xpuHostPathsSeen[*]}" >&2

    [[ -z "${xpuPropagateToOutput-}" ]] && return 0

    mkdir -p "${!xpuPropagateToOutput}/nix-support"
    # One'd expect this should be propagated-bulid-build-deps, but that doesn't seem to work
    echo "@setupXpuHook@" >> "${!xpuPropagateToOutput}/nix-support/propagated-native-build-inputs"

    local propagatedBuildInputs=( "${!xpuHostPathsSeen[@]}" )
    for output in $(getAllOutputNames) ; do
        if [[ ! "$output" = "$xpuPropagateToOutput" ]] ; then
            appendToVar propagatedBuildInputs "${!output}"
        fi
        break
    done

    # One'd expect this should be propagated-host-host-deps, but that doesn't seem to work
    printWords "${propagatedBuildInputs[@]}" >> "${!xpuPropagateToOutput}/nix-support/propagated-build-inputs"
}
