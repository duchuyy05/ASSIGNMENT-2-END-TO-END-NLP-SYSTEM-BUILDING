# VNU/UET RAG Data Builder

Build the current manual dataset from the curated CSV files:

```powershell
python scripts/build_manual_dataset.py --reuse-documents
```

If `data/processed/documents.json` is missing or you want to re-fetch source
pages, run:

```powershell
python scripts/build_manual_dataset.py --timeout 12
```

Generated files are written under `data/`. The original CSV files in
`data/manual_annotations/` are treated as source files and are not modified by
the build script.
