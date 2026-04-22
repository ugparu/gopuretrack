# Security Policy

## Supported versions

Security fixes are released against the **latest `1.x` tag** and
`master`. Older minor versions within the `1.x` line are not
back-ported — upgrade to the newest `1.x` to receive fixes. The API is
stable under [Semantic Versioning](https://semver.org/), so an upgrade
within `1.x` should never require code changes.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Use GitHub's private vulnerability-reporting channel instead:

1. Go to the [Security tab](https://github.com/ugparu/gopuretrack/security)
   of this repository.
2. Click **Report a vulnerability**.
3. Provide a description, reproduction steps, and — if you have one —
   a suggested fix.

You should receive an acknowledgment within **72 hours**. If the report
is confirmed, a fix will be prepared, a GitHub Security Advisory
published, and a patched release tagged. Credit will be given in the
advisory unless you prefer to remain anonymous.

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
