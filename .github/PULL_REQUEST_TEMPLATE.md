<!--
Thanks for contributing to Autolume!

Please make sure your PR title follows the Conventional Commits style without a scope:

    <type>: <subject>

For example: `feat: add OSC mapping for noise widget`

See CONTRIBUTING.md for the full convention and the list of allowed types.
-->

## Summary

<!-- What does this PR change, and why? 1–3 sentences. -->

## Related issue

<!-- e.g. `Closes #42`, `Refs #17`. Delete if not applicable. -->

## Type of change

- [ ] Feature
- [ ] Fix
- [ ] Other (docs, refactor, chore, ci, etc.)

## How was this tested?

<!--
Autolume has no automated test suite, so manual verification matters.
Describe the steps you took to confirm the change works:
- Which command did you run? (e.g. `uv run main.py`)
- Which UI path or workflow did you exercise?
- What did you observe?
- For UI changes, attach a screenshot or short GIF.
-->

## Checklist

- [ ] PR title follows Conventional Commits without a scope (`<type>: <subject>`)
- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] Documentation in `docs/` or `README.md` is updated if user-visible behavior changed
- [ ] If this PR adds new runtime files, they are included in `release.bat` (as a `--add-binary`/`--add-data` flag or an `xcopy` step) so the Windows release still ships them
- [ ] Screenshots or a short clip are attached for UI changes
