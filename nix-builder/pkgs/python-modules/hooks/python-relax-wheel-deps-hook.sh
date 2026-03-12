# shellcheck shell=bash

# Setup hook that modifies Python dependencies versions.
#
# Example usage in a derivation:
#
#   { …, python3Packages, … }:
#
#   python3Packages.buildPythonPackage {
#     …
#     # This will relax the dependency restrictions
#     # e.g.: abc>1,<=2 -> abc
#     pythonRelaxWheelDeps = [ "abc" ];
#     # This will relax all dependencies restrictions instead
#     # pythonRelaxWheelDeps = true;
#     # This will remove the dependency
#     # e.g.: cde>1,<=2 -> <nothing>
#     pythonRemoveWheelDeps = [ "cde" ];
#     # This will remove all dependencies from the project
#     # pythonRemoveWheelDeps = true;
#     …
#   }
#
# IMPLEMENTATION NOTES:
#
# The "Requires-Dist" dependency specification format is described in PEP 508.
# Examples that the regular expressions in this hook needs to support:
#
# Requires-Dist: foo
#   -> foo
# Requires-Dist: foo[optional]
#   -> foo[optional]
# Requires-Dist: foo[optional]~=1.2.3
#   -> foo[optional]
# Requires-Dist: foo[optional, xyz] (~=1.2.3)
#   -> foo[optional, xyz]
# Requires-Dist: foo[optional]~=1.2.3 ; os_name = "posix"
#   -> foo[optional] ; os_name = "posix"
#
# Currently unsupported: URL specs (foo @ https://example.com/a.zip).

_pythonRelaxWheelDeps() {
    local -r metadata_file="$1"

    if [[ -z "${pythonRelaxWheelDeps[*]-}" ]] || [[ "$pythonRelaxWheelDeps" == 0 ]]; then
        return
    elif [[ "$pythonRelaxWheelDeps" == 1 ]]; then
        sed -i "$metadata_file" -r \
            -e 's/(Requires-Dist: [a-zA-Z0-9_.-]+\s*(\[[^]]+\])?)[^;]*(;.*)?/\1\3/'
    else
        # shellcheck disable=SC2048
        for dep in ${pythonRelaxWheelDeps[*]}; do
            sed -i "$metadata_file" -r \
                -e "s/(Requires-Dist: $dep\s*(\[[^]]+\])?)[^;]*(;.*)?/\1\3/i"
        done
    fi
}

_pythonRemoveWheelDeps() {
    local -r metadata_file="$1"

    if [[ -z "${pythonRemoveWheelDeps[*]-}" ]] || [[ "$pythonRemoveWheelDeps" == 0 ]]; then
        return
    elif [[ "$pythonRemoveWheelDeps" == 1 ]]; then
        sed -i "$metadata_file" \
            -e '/Requires-Dist:.*/d'
    else
        # shellcheck disable=SC2048
        for dep in ${pythonRemoveWheelDeps[*]-}; do
            sed -i "$metadata_file" \
                -e "/Requires-Dist: $dep/d"
        done
    fi

}

pythonRelaxWheelDepsHook() {
    local -r metadata_file="$out/@pythonSitePackages@"/*.dist-info/METADATA

    # Using no quotes on purpose since we need to expand the glob from `$metadata_file`
    # shellcheck disable=SC2086
    _pythonRelaxWheelDeps $metadata_file
    # shellcheck disable=SC2086
    _pythonRemoveWheelDeps $metadata_file

    if (("${NIX_DEBUG:-0}" >= 1)); then
        echo "pythonRelaxWheelDepsHook: resulting METADATA for '$wheel':"
        # shellcheck disable=SC2086
        cat $metadata_file
    fi
}

postInstall+=" pythonRelaxWheelDepsHook"
