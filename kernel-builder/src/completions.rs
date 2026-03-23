use std::io::stdout;

use clap::Command;
use clap_complete::{generate, Shell};

pub fn print_completions(cmd: &mut Command, shell: Shell) {
    // Putting the binary name as a literal here, `cmd.get_name()` returns
    // the crate name and `cmd.get_bin_name()` is unreliable.
    generate(shell, cmd, "kernel-builder", &mut stdout());
}
