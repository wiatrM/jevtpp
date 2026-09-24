# Security policy

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting for
[wiatrm/jevtpp](https://github.com/wiatrm/jevtpp/security/advisories/new).
Include affected versions, impact, reproduction steps and any mitigation.

Security fixes are applied to the latest released minor version. Before 1.0,
use the latest published version or current default branch when validating a
report.

The diagnostics HTTP server does not provide TLS or authentication. Keep its
default loopback binding or place it behind an authenticated reverse proxy and
network policy. Never attach secrets or raw customer input as diagnostic tags.
Treat model files and backend responses as untrusted inputs.

