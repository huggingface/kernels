# shellcheck shell=bash

# This is the imports check hook from nixpkgs, modified to fake a /sys path
# required for tcmalloc. Without this path, tcmalloc will crash:
#
#
# https://github.com/google/tcmalloc/issues/245

# Setup hook for checking whether Python imports succeed
echo "Sourcing python-kernel-imports-check-hook.sh"

pythonKernelImportsCheckPhase() {
    echo "Executing pythonKernelImportsCheckPhase"

    if [[ -n "${pythonKernelImportsCheck[*]-}" ]]; then
        echo "Check whether the following modules can be imported: ${pythonKernelImportsCheck[*]}"
        # shellcheck disable=SC2154
        pythonKernelImportsCheckOutput="$out"
        if [[ -n "${python-}" ]]; then
            echo "Using python specific output \$python for imports check"
            pythonKernelImportsCheckOutput=$python
        fi
        export PYTHONPATH="$pythonKernelImportsCheckOutput/@pythonSitePackages@:$PYTHONPATH"

        # Prepare fake /sys
        FAKESYS="$(mktemp -d)"
        trap 'rm -rf -- "${FAKESYS}"' EXIT
        mkdir -p "${FAKESYS}/devices/system/cpu"
        echo "0-1" > "${FAKESYS}/devices/system/cpu/possible"

        # Python modules and namespaces names are Python identifiers, which must not contain spaces.
        # See https://docs.python.org/3/reference/lexical_analysis.html
        # shellcheck disable=SC2048,SC2086
        (
          cd "$pythonKernelImportsCheckOutput" && 
          @proot@/bin/proot -b ${FAKESYS}:/sys \
          @pythonCheckInterpreter@ -c 'import sys; import importlib; list(map(lambda mod: importlib.import_module(mod), sys.argv[1:]))' ${pythonKernelImportsCheck[*]}
        )
    fi
}

if [[ -z "${dontUsePythonImportsCheck-}" ]]; then
    echo "Using pythonKernelImportsCheckPhase"
    appendToVar preDistPhases pythonKernelImportsCheckPhase
fi
