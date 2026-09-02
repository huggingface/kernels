# shellcheck shell=bash

# Based on setup-cuda-hook.

# Only run the hook from nativeBuildInputs
(( "$hostOffset" == -1 && "$targetOffset" == 0)) || return 0

guard=Sourcing
reason=

[[ -n ${rocmSetupHookOnce-} ]] && guard=Skipping && reason=" because the hook has been propagated more than once"

if (( "${NIX_DEBUG:-0}" >= 1 )) ; then
    echo "$guard hostOffset=$hostOffset targetOffset=$targetOffset setup-rocm-hook$reason" >&2
else
    echo "$guard setup-rocm-hook$reason" >&2
fi

[[ "$guard" = Sourcing ]] || return 0

declare -g rocmSetupHookOnce=1
declare -Ag rocmHostPathsSeen=()
declare -Ag rocmOutputToPath=()

extendrocmHostPathsSeen() {
    (( "${NIX_DEBUG:-0}" >= 1 )) && echo "extendrocmHostPathsSeen $1" >&2

    local markerPath="$1/nix-support/include-in-rocm-root"
    [[ ! -f "${markerPath}" ]] && return 0
    [[ -v rocmHostPathsSeen[$1] ]] && return 0

    rocmHostPathsSeen["$1"]=1

    # E.g. cuda_cudart-lib
    local rocmOutputName
    # Fail gracefully if the file is empty.
    # One reason the file may be empty: the package was built with strictDeps set, but the current build does not have
    # strictDeps set.
    read -r rocmOutputName < "$markerPath" || return 0

    [[ -z "$rocmOutputName" ]] && return 0

    local oldPath="${rocmOutputToPath[$rocmOutputName]-}"
    [[ -n "$oldPath" ]] && echo "extendrocmHostPathsSeen: warning: overwriting $rocmOutputName from $oldPath to $1" >&2
    rocmOutputToPath["$rocmOutputName"]="$1"
}
addEnvHooks "$targetOffset" extendrocmHostPathsSeen

propagateRocmLibraries() {
    (( "${NIX_DEBUG:-0}" >= 1 )) && echo "propagateRocmLibraries: rocmPropagateToOutput=$rocmPropagateToOutput rocmHostPathsSeen=${!rocmHostPathsSeen[*]}" >&2

    [[ -z "${rocmPropagateToOutput-}" ]] && return 0

    mkdir -p "${!rocmPropagateToOutput}/nix-support"
    # One'd expect this should be propagated-bulid-build-deps, but that doesn't seem to work
    echo "@setupRocmHook@" >> "${!rocmPropagateToOutput}/nix-support/propagated-native-build-inputs"

    local propagatedBuildInputs=( "${!rocmHostPathsSeen[@]}" )
    for output in $(getAllOutputNames) ; do
        if [[ ! "$output" = "$rocmPropagateToOutput" ]] ; then
            appendToVar propagatedBuildInputs "${!output}"
        fi
        break
    done

    # One'd expect this should be propagated-host-host-deps, but that doesn't seem to work
    printWords "${propagatedBuildInputs[@]}" >> "${!rocmPropagateToOutput}/nix-support/propagated-build-inputs"
}
postFixupHooks+=(propagateRocmLibraries)
