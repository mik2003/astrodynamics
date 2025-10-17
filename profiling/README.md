profiling — cProfile viewer
============================

Create a binary profile (from repo root):

```powershell
py -m cProfile -o profiling\profile.bin -m project
```

View and optionally write a timestamped textual dump:

```powershell
py -m profiling --units us --lines 50 --timestamp
```

The textual dump will be written into `profiling/` by default.
