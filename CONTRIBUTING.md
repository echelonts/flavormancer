# Contributing to Flavormancer

> **Note:** Flavormancer is currently built by a fixed internal team and is **not
> accepting external contributions** at this stage. Unsolicited pull requests will be
> closed (no reflection on the work). This guide is for the project's own team.

Thanks for contributing. This guide covers how we branch, commit, and review so
the history stays clean and the work stays legible.

## Workflow

Trunk-based with short-lived branches. `main` is protected — no direct pushes.
Branch → keep it small → open a PR → get it reviewed → **squash-merge** → delete
the branch.

## Branch naming

`type/scope-desc`, e.g. `feat/api-predict-endpoint`, `feat/ui-taste-meters`,
`chore/infra-compose`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), scope = module:

```
feat(api): add /predict endpoint backed by ONNX Runtime
fix(training): correct InChIKey dedup collision
docs(data): document ChemTastesDB column names
```

- **Types:** `feat fix docs chore refactor test perf`
- **Scopes:** `training aroma api serving ui infra data docs`
- Commits must be **signed** (GPG or SSH). PRs with unsigned commits won't merge.

## Pull requests

- **Small** — one logical change, readable as a sequence of decisions.
- **Link the issue:** `Closes #42`.
- **CI green** before merge.
- **One approval required**, routed by [`CODEOWNERS`](CODEOWNERS).
- **Squash-merge**; the branch is deleted on merge. Full commit-by-commit detail
  and the review thread stay preserved in the PR.
- For genuine pair work, add `Co-authored-by:` trailers so everyone gets credit.

## Definition of done

CI green · reviewed and approved · issue linked · docs updated if behavior changed.

## Project board

`Backlog → Ready → In progress → In review → Done`

## Labels

`area:training` `area:aroma` `area:api` `area:serving` `area:ui` `area:infra`
`area:data`, plus `good-first-issue`, `bug`, `enhancement`.

## Local setup

Datasets and trained models are **never** committed — the training scripts
download their sources and `.gitignore` keeps artifacts out of the repo. See
`docs/` for environment and run steps.
