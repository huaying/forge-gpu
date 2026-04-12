# Contributing to Forge

Thank you for your interest in contributing to Forge! Here's how to get started.

## Development Setup

### Prerequisites

- **Rust 1.78+** — install via [rustup](https://rustup.rs/)
- **CUDA Toolkit 12.0+** — for GPU kernel compilation
- **A CUDA-capable GPU** — minimum compute capability 7.0 (RTX 2000 series+)

### Building

```bash
git clone https://github.com/huaying/forge-gpu.git
cd forge-gpu
cargo build
cargo test
```

### Project Structure

```
forge-gpu/
├── forge/               # Top-level crate (re-exports + prelude)
├── forge-core/          # Math types: Vec2/3/4, Mat22/33/44, Quat, scalars
├── forge-codegen/       # CUDA type mapping utilities
├── forge-macros/        # #[kernel] and #[func] proc macros, Rust→CUDA emitter
├── forge-runtime/       # CUDA context, Array<T>, kernel compilation, launch
├── examples/            # Runnable examples (particles, kernel_demo)
├── tests/               # Integration tests (kernel_macro_test, func_macro_test)
├── DESIGN.md            # Architecture & design decisions (aspirational — not all implemented)
├── ROADMAP.md           # Development milestones
└── PROJECT_PLAN.md      # Sprint tracking
```

## Development Guidelines

### 📏 Rule #1: Docs Must Match Code

**Every commit that changes behavior must update docs in the same commit.**

This means:
- If you add a feature → update README.md "What Works Today" section
- If you change an API → update all code examples that reference it
- If you implement something from DESIGN.md → move it from "planned" to "implemented"
- If you defer something → note it explicitly ("not yet implemented")

**README.md is the source of truth for what works.** DESIGN.md and ROADMAP.md describe the vision (what we're building toward). Keep the distinction clear:
- README: "this is what you can do right now"
- DESIGN/ROADMAP: "this is where we're going"

**Before every release/push, ask:** does `cargo run --example particles` match what README shows? If not, fix it.

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- Write doc comments (`///`) for all public items
- Prefer `// SAFETY:` comments for any `unsafe` blocks

### Testing

- Unit tests go in the same file (`#[cfg(test)]`)
- Integration tests go in `tests/`
- GPU tests should be marked with `#[cfg(feature = "cuda")]`
- Test against Warp's output for math correctness when applicable

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(core): add Vec4 type with full operator overloading
fix(runtime): prevent double-free on Array drop
docs(design): update autodiff section
test(codegen): add proc macro expansion tests
```

### Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests for your changes
4. Ensure `cargo test` passes
5. Open a PR with a clear description

## Areas for Contribution

### 🟢 Good First Issues

- Add missing math builtins (see Warp's builtins list in DESIGN.md)
- Improve error messages in proc macros
- Write examples
- Documentation improvements

### 🟡 Medium

- Implement new types (spatial vectors, transforms)
- Add sparse matrix operations
- CPU backend via Rayon
- Benchmark suite against Warp

### 🔴 Advanced

- Autodiff adjoint kernel generation
- Multi-backend codegen (ROCm, Metal)
- CUDA graph capture
- FEM integration framework

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
