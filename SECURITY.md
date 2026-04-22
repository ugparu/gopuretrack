# Security Policy

## Supported versions

Security fixes are released against the **latest `1.x` tag** and
`master`. Older minor versions within the `1.x` line are not
back-ported — upgrade to the newest `1.x` to receive fixes. The API is
stable under [Semantic Versioning](https://semver.org/), so an upgrade
within `1.x` should never require code changes.

## Scope

The tracker reads in-memory detection structs provided by the caller
and produces in-memory track structs in return. It does not perform
network I/O, file I/O (outside test fixtures), or subprocess execution.
Accordingly, the practical threat surface is:

- Panics or unbounded allocation triggered by malformed caller input
  (NaN / Inf boxes, zero-size or negative-dimension detections, very
  large detection counts).
- Numerical divergence from the reference under adversarial inputs
  that nonetheless parse as valid detections.

Reports in either category are welcome.
