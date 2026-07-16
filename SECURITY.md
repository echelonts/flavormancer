# Security & verification

## Verifying signatures

Every commit and every release tag in this repository is **signed** (GPG/DCO). The public
key is committed as [`KEYS`](KEYS) so you can verify them yourself.

```sh
gpg --import KEYS                      # trust the project key once
git verify-commit HEAD                 # verify a commit
git verify-tag v0.1.0                  # verify a release tag
```

**Signing key**

- Fingerprint: `D905 5DE5 AF81 7666 49AF 4183 675F 9696 4677 1BCB`
- Type: ed25519 (sign + certify)

Datasets and trained models are **never** committed (see `.gitignore`); the training
scripts regenerate them from their cited sources, so there are no binary artifacts to
sign beyond the source tree itself (GitHub's auto-generated release archives correspond
to the signed tag).

## Reporting a vulnerability

Flavormancer predicts *flavor properties only* and is **not** a safety, toxicity, GRAS,
regulatory, or stability determination — a prediction is never a clearance to consume.

For security issues in the code (not model outputs), please open a private report via
GitHub Security Advisories, or email **admin@echelonts.net**. Please do not file
public issues for suspected vulnerabilities.
