# Pretrained Checkpoint Configs

These files were generated from Hugging Face FAIR Chemistry checkpoints with:

```bash
bash -lc 'source setup_cuda.sh && FAIRCHEM_CACHE_DIR="$PWD/.cache/fairchem" .venv/bin/python tools/inspect_pretrained_checkpoints.py'
```

Each model YAML contains:

- Hugging Face source metadata and local checkpoint path
- checkpoint checksum and size
- lightweight state-dict statistics
- extracted `model_config`
- extracted `tasks_config`

The checkpoint weights themselves are cached under `.cache/fairchem/` and are
not committed.
