# 10xgraph 0.10.2

Smaller installs: Pillow is no longer a required dependency. No API changes.

```bash
pip install -U "10xgraph==0.10.2"
```

## Changed

- **Pillow moved to the `images` extra.**
  - Every install and Docker image is about 20 MB smaller.
  - With Pillow installed, the media processor still resizes images, fixes EXIF orientation and converts formats.
  - Without it, images pass through unchanged.
  - If you rely on image resizing or conversion, install the extra: `pip install -U "10xgraph[images]==0.10.2"` (also part of `10xgraph[all]`).

See `CHANGELOG.md` for details.
