---
name: Python runtime setup
description: Environment-specific setup needed for Python packages in this workspace
---

Use the supported Python tools module before installing Python packages; the system Python is immutable and does not provide a usable pip installation.

**Why:** Direct pip installation against the system interpreter is blocked by the externally-managed-environment policy.

**How to apply:** For future Python apps, check available Python modules, install a supported tools module such as python-3.11, then install packages through the managed package flow.