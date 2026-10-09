<!-- >>> dotfiles cursor-cloud >>> -->
<!-- Managed by `dotfiles install` / `dotfiles install-cloud-agent`. Do not edit by hand. -->

## Cursor Cloud specific instructions

- **dotfiles** lives at `$HOME/dotfiles` (override with `DOTFILES_HOME`).
- **DKR** lives at `$HOME/DKR` (override with `DKR_HOME`).
- Before writing or pushing code, run `dotfiles setup --non-interactive` (or `bash .cursor/bootstrap.sh` on a cold Build).
- If either repo is missing after bootstrap, **stop and report** — do not continue, invent a path, or skip setup.
- Generator-owned paths (`.cursor/`, and when present `.claude/` / `.github/` from the CLI) must not be hand-edited in this product repo; change templates in `avoidTheLite/dotfiles` and re-run `dotfiles install-cloud-agent` (or regenerate).

<!-- <<< dotfiles cursor-cloud <<< -->
