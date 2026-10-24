# CVE-2026-111497: Pillow's bundled Brotli decoder

Some Pillow wheels, including the inspected macOS arm64 Pillow 13.0.0 wheel, still contain an affected Google Brotli decoder. A Pillow version bump alone is not proof of remediation. Existing Pillow pins are retained to preserve the project's dependency resolution.

After creating or syncing an environment, stop workloads that use it and run with that environment's Python (Python 3.9+):

```sh
python scripts/repair_pillow_brotli.py --check
python scripts/repair_pillow_brotli.py --apply --backup-dir /absolute/path/outside-the-venv/pillow-brotli-backup
python scripts/repair_pillow_brotli.py --check
```

Choose a new backup directory for each actual repair. Re-running against a fixed decoder does not change files or create a backup. Restart affected workers after repairing.

The helper applies a temporary native-library repair on macOS/Linux. It downloads immutable Google Brotli 1.2.1 source, verifies its SHA-256, compiles a self-contained decoder, checks existing public symbols and normal decompression, backs up the decoder and RECORD with verified hashes, atomically replaces the decoder, and updates RECORD hashes. The receipt identifies the source commit and installed hashes. A C compiler and, on macOS, functioning Command Line Tools and code signing are required. It refuses signed app bundles, ambiguous decoder copies and paths escaping site-packages.

For an explicitly selected environment, `--site-packages /absolute/path/to/site-packages` is supported. Use matching platform/architecture. Older Python versions may require a separate supported Python for the helper and independent application compatibility testing.

`no_vendored_decoder` means no separately packaged decoder was found in Pillow's file list; it is not a declaration that every native library or the whole environment is safe. System libraries, statically embedded copies, Node.js, containers and other packages require separate checks. Windows is not repaired by this helper; use a vendor-fixed build.

The added CI job tests the repair helper against an isolated old wheel on Linux/macOS. It does not automatically repair every deployed environment or run the project's full application tests. Invoke the helper after every relevant dependency sync/install; package reinstalls can restore the vulnerable wheel. Merge/deploy changes and inspect CI results before treating a workload as remediated.

To roll back for a compatibility failure, stop the workload and restore the original decoder and RECORD from the backup receipt, then verify hashes. This reinstates the affected library and is not a security fix.

References:
- https://github.com/google/brotli/releases/tag/v1.2.1
- https://nodejs.org/en/blog/vulnerability/october-2026-security-releases
