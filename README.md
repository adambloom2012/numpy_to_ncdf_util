# npy2nc

Config-driven `.npy` → NetCDF converter. When your model's output layout changes
(new species, extra channel, different shape/dtype), you edit a few lines of TOML
instead of Rust or Python.

## Build

```bash
# needs libnetcdf (macOS: brew install netcdf; Ubuntu: apt install libnetcdf-dev)
cargo build --release          # binary: target/release/npy2nc

# or, no system libs (needs cmake + C compiler; slower first build):
cargo build --release --features static
```
Requires Rust ≥ 1.80 (`rustup update`).

## Commands

```bash
npy2nc inspect  *.npy                 # dtype, shape, min/max/mean/NaN count (--fast: header only)
npy2nc example-config > cfg.toml      # commented starting point
npy2nc run -c cfg.toml --dry-run      # load + validate everything, print resulting variables/shapes
npy2nc run -c cfg.toml                # convert (parallel across samples)
npy2nc run -c cfg.toml --only '2019072' --output-dir /tmp/x --jobs 4

# one-off, no config:  NAME=FILE:dim1,dim2[:units]  (repeatable)
npy2nc quick -o out.nc --template tpl.nc4 --date 2019-07-20 \
  --var "NO2=pred.npy:time,lat,lon:molec cm-2"
```

## How a config works

Per sample (`stem`), each `[[variable]]` does:

`load {stem}-file  →  select  →  split  →  permute  →  scale/offset  →  write`

| Key | Meaning |
|---|---|
| `file` | path relative to `input.dir`; `{stem}` substituted |
| `dims` | dim names of the array **after** select/split/permute |
| `select = [{axis=0, index=0}]` | fix an axis (drops it). Negative index counts from end |
| `[variable.split]` `axis`, `labels`, `long_names` | one npy axis → many variables. `name` must contain `{label}`; `units`/`long_name` may use `{label}`, `{long}` |
| `permute = [1,0]` | reorder axes to match `dims` |
| `scale`, `offset` | `x*scale+offset` (unit conversion) |
| `dtype` | `"f32"` (default) or `"f64"` on output. Input dtype can be anything |
| `optional = true` | skip if file is absent |
| `attrs = {k = "v"}` | extra attributes |

Other sections: `[template]` (coords copied **by dim name** from any NetCDF, incl. their
`units`/`long_name`), `[coords.<dim>]` (`from_template`, or `start`/`step`, plus `attrs`),
`[time]` (axis from a date parsed out of the stem, or fixed `start`), `[global_attrs]`
(`{stem}`, `{date}` available), `[output]` (`pattern`, `compression`, `shuffle`, `overwrite`).

Unknown/misspelled config keys are errors, and a shape mismatch tells you the input shape,
the shape after select/split, and what `dims` you gave.

See `examples/multispecies_surface.toml` for a config that reproduces
`numpy_to_netcdf.py --file_type multi_species` (VCD + surface, 7 species).

## Adapting to changed outputs — cheat sheet

- **Added a species:** append to `labels` (and optionally `long_names`).
- **Shape (24,1,H,W) instead of (24,H,W):** add `select = [{axis=1, index=0}]`.
- **Different date format in filenames:** change `date_regex` / `date_format`.
- **A new output file per sample:** copy a `[[variable]]` block, change `file`/`name`.
- **`_pred.npy` also matching `_foo_pred.npy`:** add to `exclude_suffixes` (or use `--only`).

## Notes / limitations

- Samples are converted in parallel, but libnetcdf/HDF5 serialize their own calls behind a
  global lock, so the win is in `.npy` decoding/reshaping, not in compression. If you're
  compression-bound, lower `compression` (1–2 is much faster than 4–6).
- Supported `.npy` dtypes: f2/f4/f8, i1–i8, u1–u8, bool; either endianness; C or Fortran order.
  Not supported: `.npz`, structured/object arrays.
- Time is written CF-style (`hours since YYYY-MM-DD 00:00:00`) rather than epoch-1970; xarray
  decodes both to the same datetimes.
