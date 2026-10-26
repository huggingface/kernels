# Function for generating `src` paths for crates in the `kernels` tree.
{ lib, runCommand }:

{
  crates,
  tests ? [ ],
  extraFiles ? [ ],
}:

let
  inherit (lib) fileset;

  root = ../../..;
  crateFile = crate: path: root + "/${crate}/${path}";

  readManifest = crate: builtins.fromTOML (builtins.readFile (crateFile crate "Cargo.toml"));

  workspaceMembers = (builtins.fromTOML (builtins.readFile (root + "/Cargo.toml"))).workspace.members;

  # Workspace crates that a crate depends on through `path`.
  pathDependencies =
    crate:
    let
      manifest = readManifest crate;
      # Get all dependencies.
      dependencies = lib.concatMap (table: lib.attrValues (manifest.${table} or { })) [
        "dependencies"
        "build-dependencies"
        "dev-dependencies"
      ];
    in
    map
      # Get the crate name of a path dependency.
      (dependency: baseNameOf dependency.path)
      # Dependencies that are path dependencies.
      (builtins.filter (dep: dep ? path) dependencies);

  # Find the transitive closure workspace of workspace dependencies starting
  # at `crates`. These are dependencies that we need the full source of.
  buildCrates = map (crate: crate.key) (
    builtins.genericClosure {
      startSet = map (key: { inherit key; }) crates;
      operator = { key, ... }: map (key: { inherit key; }) (pathDependencies key);
    }
  );

  # Workspace crates that are not in the transitive closure of `crates` can
  # be stubbed. We need some files to make Cargo happy, but we don't need their
  # actual sources.
  stubCrates = lib.subtractLists buildCrates workspaceMembers;

  # Crate entry points (lib.rs, main.rs, or custom-defined binaries).
  entryPoints =
    crate:
    let
      manifest = readManifest crate;
    in
    builtins.filter (path: builtins.pathExists (crateFile crate path)) (
      lib.unique (
        [ (manifest.lib.path or "src/lib.rs") ]
        ++ map (bin: bin.path or "src/main.rs") (manifest.bin or [ ])
        ++ [ "src/main.rs" ]
      )
    );

  crateFileset =
    crate:
    fileset.unions (
      # Build files.
      [
        (crateFile crate "Cargo.toml")
        (fileset.maybeMissing (crateFile crate "build.rs"))
      ]
      # Entry points.
      ++ map (dir: fileset.maybeMissing (crateFile crate dir)) (
        lib.unique ([ "src" ] ++ map builtins.dirOf (entryPoints crate))
      )
      # Tests.
      ++ lib.optional (builtins.elem crate tests) (fileset.maybeMissing (crateFile crate "tests"))
    );

  # For stubs, we have to use the real Cargo.toml.
  stubManifests = fileset.unions (map (crate: crateFile crate "Cargo.toml") stubCrates);

  # Entrypoints that we need to stub.
  stubEntryPoints = lib.concatMap (
    crate: map (path: "${crate}/${path}") (entryPoints crate)
  ) stubCrates;

  # Tie it all together.
  source = fileset.toSource {
    inherit root;
    fileset = fileset.unions (
      [
        (root + "/Cargo.lock")
        (root + "/Cargo.toml")
        stubManifests
      ]
      ++ map crateFileset buildCrates
      ++ map (path: root + "/${path}") extraFiles
    );
  };
in
assert lib.assertMsg (lib.all (crate: builtins.elem crate workspaceMembers) (
  crates ++ tests
)) "mkKernelsRustSrc: all of `crates` and `tests` must be workspace members";
runCommand "source" { } ''
  cp -r --no-preserve=mode ${source} $out
  ${lib.concatMapStringsSep "\n" (
    path: "install -D /dev/null $out/${lib.escapeShellArg path}"
  ) stubEntryPoints}
''
