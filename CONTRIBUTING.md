# Contributing to `kernels`

Thanks for your interest in contributing to `kernels`! We love contributions
from the community!

## Ways to contribute

Contributions aren't only code. Filing a bug report, requesting a feature, or
sharing general feedback is just as valuable — use the
[issue templates](https://github.com/huggingface/kernels/issues/new/choose) to get
started.

## Open an issue first

**Always open an issue to discuss the scope of your change before opening a PR.**
Use the [issue templates](https://github.com/huggingface/kernels/issues/new/choose)
to file a bug report or feature request, and wait for a maintainer to agree on the
approach before you start writing code.

It helps us align early, avoid duplicate or unmergeable work, and manage our review bandwidth amidst a steady flux of agent-generated PRs.

### Vouching

We use the [vouch](https://github.com/mitchellh/vouch) workflow to gate PRs:

- PRs from contributors who are **not vouched for are automatically closed**.
- Regular contributors are added to the vouch list
  ([`.github/VOUCHED.td`](.github/VOUCHED.td)).

If your PR is auto-closed, that's expected — open an issue, discuss the change, and
a maintainer will follow up.

### LLM-generated changes

If your change is mostly or entirely LLM-generated, we prefer that you share your
**prompt in an issue** rather than opening a PR with the generated diff. Either way,
please fill out the LLM disclosure section in the PR template honestly.

## Development

The Python package lives in [`kernels/`](kernels/). Install it with the dev extras:

```bash
pip install -e "kernels/[dev]"
```

Before pushing, format and lint:

```bash
make style    # auto-format and fix
make quality  # check formatting and lint (what CI runs)
```

Run the tests:

```bash
cd kernels && pytest
```

New or changed functionality should come with test coverage.

## Opening the PR

Once a maintainer has approved the approach on the issue:

1. Branch off `main` and make your change.
2. Link the PR to its issue (`Closes #123`) and fill out the PR template.
3. Make sure `make quality` and the tests pass.

We'll take it from there. Thanks for contributing!
