# App Folder Structure

Use `START_APP.bat` as the only user-facing entry point.

```text
App/
  START_APP.bat
  requirements-qt.txt

  config/       JSON config and examples
  maintenance/  old one-off patch/bootstrap files
  runtime/      watcher logs and locks
  src/          runtime Python modules
  tools/        optional helper tools
  vendor/       vendored Pupil Labs libraries
```

## Runtime Modules

- `src/app.py` - desktop app and watcher manager.
- `src/watcher.py` - background recording scanner.
- `src/analyzer.py` - AOI detection and export pipeline.
- `src/precheck.py` - optional sampled recording precheck.
- `src/masks.py` - AOI mask config loading.
- `src/paths.py` - shared folder paths.

## Config And Runtime Files

- App settings: `config/settings.json`
- Optional custom masks: `config/aoi_masks.json`
- Watcher log: `runtime/watcher.log`
- Watcher lock: `runtime/watcher.lock`
