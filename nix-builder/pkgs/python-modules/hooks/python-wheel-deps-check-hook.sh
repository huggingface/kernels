# Setup hook for PyPA installer.
echo "Sourcing python-runtime-deps-check-hook"

pythonWheelDepsCheckHook() {
    echo "Executing pythonWheelDepsCheck"

    export PYTHONPATH="$out/@pythonSitePackages@:$PYTHONPATH"

    # Assumption: we don't have spaces in these paths.
    for metadata_file in $out/@pythonSitePackages@/*.dist-info/METADATA; do
      echo "Checking runtime dependencies for $metadata_file"
      @pythonInterpreter@ @hook@ "$metadata_file"
    done

    echo "Finished executing pythonWheelDepsCheck"
}

if [ -z "${dontCheckRuntimeDeps-}" ]; then
    echo "Using pythonWheelDepsCheckHook"
    # Ideally, this would be post-install, but we have to guarantee
    # that the relax hook runs before this, so we move it a phase
    # later.
    appendToVar preFixupPhases pythonWheelDepsCheckHook
fi
