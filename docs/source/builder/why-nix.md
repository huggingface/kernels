# Why Nix?

The Kernel Builder project uses Nix to build custom kernels designed specifically for PyTorch.

Here’s why we chose Nix and why it's particularly suited to our workflow:

## 1. Consistent and Reproducible Builds

Nix guarantees deterministic evaluation, ensuring that every kernel is built identically, regardless of the host environment. This consistency prevents "it works on my machine" problems, making debugging and deployment straightforward.

## 2. Simplified Dependency Management

Compiling PyTorch kernels often involves complex dependencies such as CUDA versions, PyTorch APIs, and various C++ toolchains. Nix explicitly defines and manages these dependencies, eliminating version conflicts and making maintenance easier.

## 3. Declarative Configuration

Nix’s declarative approach clearly specifies exactly what each kernel build needs. This transparency aids collaboration, speeds up troubleshooting, and makes it easy to document the build process.

## 4. Isolated, Reliable Builds

Each kernel build with Nix runs in a fully isolated sandbox, removing any uncertainty about external state. This isolation ensures clean builds, free of unexpected side effects.

## 5. Efficient Caching and CI Integration

Kernel compilation can be resource-intensive. Nix leverages efficient caching of build artifacts, significantly reducing build times and optimizing continuous integration workflows.

## 6. Easy Experimentation and Rollbacks

Nix allows you to experiment with different kernel configurations, PyTorch versions, or CUDA toolkits easily. If a change introduces an issue, reverting to a previous state is quick and effortless.

Overall, Nix streamlines the Kernel Builder workflow, allowing us to efficiently and reliably manage complex machine learning kernel builds.

---

If you want to learn more about Nix, check out the following resources:

## References

- **The Official Nix Manual:**
  - The definitive source for all things Nix, providing comprehensive coverage of its features, commands, and ecosystem.
  - Link: [Nix Manual (nixos.org)](https://nixos.org/manual/nix/stable/)
- **Nix Pills:**
  - A series of blog posts breaking down complex Nix concepts into digestible pieces, ideal for a structured, tutorial-style approach.
  - Link: [Nix Pills (nixos.org)](https://nixos.org/guides/nix-pills/)
- **nix.dev**:
  - Home of official documentation for the Nix ecosystem.
  - Link [nix.dev](https://nix.dev/)
- **NixOS Wiki:**
  - A community-driven wiki with a wealth of information, including tips, tricks, and tutorials, covering a wide range of topics, including NixOS-specific information.
  - Link: [NixOS Wiki](https://nixos.wiki/wiki/Main_Page)

