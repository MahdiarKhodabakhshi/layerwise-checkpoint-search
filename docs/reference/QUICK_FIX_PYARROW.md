# Quick Fix for pyarrow Error

If you see `❌ MTEB error: No module named 'pyarrow'`, run this:

```bash
cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate
pip install pyarrow
```

Or load the arrow module (if available):

```bash
module load arrow/21.0.0
```

The updated test script will automatically install pyarrow if it's not found.
