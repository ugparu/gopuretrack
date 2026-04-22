# Contributing to GoPureTrack

Thanks for considering a contribution. Bug reports, performance
regressions, and clarifying PRs are all welcome. This document covers
the practical bits — environment, checks to run before pushing, and the
conventions used throughout the repo.

## Prerequisites

- **Go 1.23+** (the module targets 1.23; CI runs 1.23.x and 1.24.x).
- **golangci-lint v2.11+** — install via
  [the official installer](https://golangci-lint.run/welcome/install/):

  ```bash
  go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.11.4
  ```

No other tooling is required — there is no cgo, no code generation, no
Makefile.

## Running the checks

Before opening a PR, please run:

```bash
go build ./...
go vet ./...
go test -race ./...
golangci-lint run ./...
```

All four must pass on your change. CI runs the same commands on
Linux / macOS / Windows × Go 1.23 / 1.24.

### Benchmarks

If your change touches a hot path, include before/after numbers:

```bash
go test -bench . -benchmem -run=^$ -benchtime=3s ./puretrack > new.txt
git stash
go test -bench . -benchmem -run=^$ -benchtime=3s ./puretrack > old.txt
git stash pop
benchstat old.txt new.txt
```

### Parity

The `puretrack/parity_test.go` suite replays MOT17 sequences against a
stored reference and fails on any per-frame deviation. If your change is
expected to shift outputs, the reference fixtures under
`puretrack/testdata/reference/` must be regenerated — please call it out
in the PR description.

## Pull requests

- **One logical change per PR.** Refactors, feature work, and fixes go
  in separate PRs so they can be reviewed and reverted independently.
- **Commit messages are English, imperative, concise.** «Fix negative
  IoU on zero-size boxes» beats «fixes».
- **Keep the diff tight.** No unrelated formatting churn — `gofmt` and
  `goimports` are part of the lint config and catch the rest.
- **Public API changes** (new exported type, method, field, or constant)
  must be documented with a godoc comment and ideally covered by an
  example under `puretrack/example_test.go`.
- **Semantic Versioning is binding.** The project follows
  [SemVer](https://semver.org/): breaking changes to the exported API
  require a major-version bump. Renaming or removing anything public,
  changing a function's signature, or changing the semantics of a
  `Config` field in an incompatible way — all require a `v2`. Prefer
  additive changes (new method, new optional `Config` field) and
  deprecation notices over removal.
- **Comments explain the *why***, not the *what*. The existing code
  style prefers self-documenting identifiers and reserves comments for
  non-obvious invariants, numerical precision concerns, and algorithm
  references.

## Issues

A good bug report includes:

- Go version (`go version`) and OS.
- A minimal reproduction — the detection stream, config, and expected
  vs. actual track ids / boxes.
- Whether the parity suite still passes (`go test ./puretrack -run Parity`).

Feature proposals are welcome as issues; please start a discussion
before writing the patch so scope is agreed up-front.

## License

By contributing you agree that your contribution will be released under
the project's [MIT License](./LICENSE).
