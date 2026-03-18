### kernels copy-readme

Use `kernels copy-readme <repo_id>` to copy the most recent version's README
to the `main` branch of a kernel repository, so visitors landing on the
default branch see the latest documentation.

By default, the README is printed to stdout. Use `--push-to-hub` to upload
it directly to the `main` branch on the Hub.

```bash
# Print
kernels copy-readme kernels-community/activation

# Push to the Hub
kernels copy-readme kernels-community/activation --push-to-hub
```
