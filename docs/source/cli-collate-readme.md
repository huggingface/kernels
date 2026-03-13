### kernels collate-readme

Use `kernels collate-readme <repo_id>` to generate a README for the `main`
branch of a kernel repository. This README lists all available version
branches (e.g., `v1`, `v2`) with links, so visitors landing on
the default branch can discover the kernel's versions.

By default, the README is printed to stdout. Use `--push-to-hub` to upload
it directly to the `main` branch on the Hub.

```bash
# Print
kernels collate-readme kernels-community/activation

# Push to the Hub
kernels collate-readme kernels-community/activation --push-to-hub
```
