//! Sample discovery, array preparation, and NetCDF writing.

use crate::config::*;
use crate::npy::{self, Elem};
use anyhow::{anyhow, bail, Context, Result};
use chrono::NaiveDate;
use ndarray::{ArrayD, Axis, IxDyn};
use parking_lot::ReentrantMutex;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// libnetcdf/HDF5 are not safe to call from multiple threads at once, even
/// though the `netcdf` crate's own internal locking suggests otherwise (see
/// https://github.com/georust/netcdf/issues/43). `.npy` decoding runs freely
/// in parallel across samples (that's where the real CPU cost is - see
/// README), but every call into the netcdf/HDF5 libraries - opening the
/// template, creating a file, writing a variable - is funneled through this
/// single process-wide lock. It's reentrant so `write()` can hold it across
/// its call into `TemplateCache::get()` without deadlocking.
static NC_LOCK: ReentrantMutex<()> = ReentrantMutex::new(());

// ───────────────────────── helpers ─────────────────────────

pub fn subst(s: &str, kv: &[(&str, &str)]) -> String {
    let mut out = s.to_string();
    for (k, v) in kv {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub stem: String,
    pub date_str: String,
    pub date: Option<NaiveDate>,
}

impl Sample {
    fn kv(&self) -> Vec<(&str, &str)> {
        vec![("stem", &self.stem), ("date", &self.date_str)]
    }
}

/// Find sample stems in `input.dir`.
pub fn discover(cfg: &Config, only: Option<&Regex>) -> Result<Vec<String>> {
    let mut stems = Vec::new();
    let rd = std::fs::read_dir(&cfg.input.dir)
        .with_context(|| format!("reading {}", cfg.input.dir.display()))?;
    for e in rd {
        let name = e?.file_name().to_string_lossy().into_owned();
        if cfg.input.exclude_suffixes.iter().any(|x| name.ends_with(x)) {
            continue;
        }
        if let Some(stem) = name.strip_suffix(&cfg.input.discover) {
            if only.map_or(true, |r| r.is_match(stem)) {
                stems.push(stem.to_string());
            }
        }
    }
    stems.sort();
    Ok(stems)
}

pub fn make_sample(cfg: &Config, stem: &str, date_re: Option<&Regex>) -> Result<Sample> {
    let Some(t) = &cfg.time else {
        return Ok(Sample {
            stem: stem.into(),
            date_str: String::new(),
            date: None,
        });
    };
    if let Some(re) = date_re {
        let caps = re
            .captures(stem)
            .ok_or_else(|| anyhow!("date_regex did not match stem '{stem}'"))?;
        let m = caps
            .get(1)
            .or_else(|| caps.get(0))
            .unwrap()
            .as_str()
            .to_string();
        let d = NaiveDate::parse_from_str(&m, &t.date_format)
            .with_context(|| format!("parsing date '{m}' with '{}'", t.date_format))?;
        Ok(Sample {
            stem: stem.into(),
            date_str: m,
            date: Some(d),
        })
    } else {
        let s = t.start.as_ref().unwrap();
        let d = NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .with_context(|| format!("time.start '{s}' must be YYYY-MM-DD"))?;
        Ok(Sample {
            stem: stem.into(),
            date_str: s.replace('-', ""),
            date: Some(d),
        })
    }
}

// ───────────────────────── prepared data ─────────────────────────

pub enum Arr {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}
impl Arr {
    pub fn shape(&self) -> &[usize] {
        match self {
            Arr::F32(a) => a.shape(),
            Arr::F64(a) => a.shape(),
        }
    }
}

pub struct Prepared {
    pub name: String,
    pub dims: Vec<String>,
    pub data: Arr,
    pub attrs: Vec<(String, String)>,
}

fn build<T: Elem>(
    v: &Variable,
    path: &Path,
    sample: &Sample,
    wrap: fn(ArrayD<T>) -> Arr,
) -> Result<Vec<Prepared>> {
    let mut arr: ArrayD<T> = npy::read(path)?;
    let orig_shape = arr.shape().to_vec();

    for s in &v.select {
        if s.axis >= arr.ndim() {
            bail!(
                "select axis {} out of range for shape {:?}",
                s.axis,
                arr.shape()
            );
        }
        let len = arr.shape()[s.axis] as i64;
        let idx = if s.index < 0 { len + s.index } else { s.index };
        if idx < 0 || idx >= len {
            bail!(
                "select index {} out of range for axis {} (len {len})",
                s.index,
                s.axis
            );
        }
        arr = arr.index_axis_move(Axis(s.axis), idx as usize);
    }

    // (label, long, array)
    let mut parts: Vec<(Option<String>, String, ArrayD<T>)> = Vec::new();
    if let Some(sp) = &v.split {
        if sp.axis >= arr.ndim() {
            bail!(
                "split axis {} out of range for shape {:?}",
                sp.axis,
                arr.shape()
            );
        }
        let len = arr.shape()[sp.axis];
        if len != sp.labels.len() {
            bail!(
                "split axis {} has length {len} but {} labels were given (input shape {:?})",
                sp.axis,
                sp.labels.len(),
                orig_shape
            );
        }
        for (i, l) in sp.labels.iter().enumerate() {
            let long = sp.long_names.get(l).cloned().unwrap_or_else(|| l.clone());
            parts.push((
                Some(l.clone()),
                long,
                arr.index_axis(Axis(sp.axis), i).to_owned(),
            ));
        }
    } else {
        parts.push((None, String::new(), arr));
    }

    let mut out = Vec::new();
    for (label, long, mut a) in parts {
        if let Some(p) = &v.permute {
            if p.len() != a.ndim() {
                bail!(
                    "permute has {} entries but array has {} axes",
                    p.len(),
                    a.ndim()
                );
            }
            a = a.permuted_axes(IxDyn(p));
        }
        if a.ndim() != v.dims.len() {
            bail!(
                "input shape {:?} -> {:?} after select/split has {} axes, \
                 but dims = {:?} lists {}. Adjust `select`, `split`, or `dims`.",
                orig_shape,
                a.shape(),
                a.ndim(),
                v.dims,
                v.dims.len()
            );
        }
        let mut a = a.as_standard_layout().into_owned();
        if v.scale.is_some() || v.offset.is_some() {
            let (s, o) = (v.scale.unwrap_or(1.0), v.offset.unwrap_or(0.0));
            a.mapv_inplace(|x| T::from_f64(x.to_f64() * s + o));
        }

        let lab = label.clone().unwrap_or_default();
        let mut kv = sample.kv();
        kv.push(("label", &lab));
        kv.push(("long", &long));
        let mut attrs = Vec::new();
        if let Some(u) = &v.units {
            attrs.push(("units".to_string(), subst(u, &kv)));
        }
        if let Some(l) = &v.long_name {
            attrs.push(("long_name".to_string(), subst(l, &kv)));
        }
        for (k, val) in &v.attrs {
            attrs.push((k.clone(), subst(val, &kv)));
        }
        out.push(Prepared {
            name: subst(&v.name, &kv),
            dims: v.dims.clone(),
            data: wrap(a),
            attrs,
        });
    }
    Ok(out)
}

/// Load and transform every variable for one sample (pure Rust, no NetCDF calls).
pub fn prepare(cfg: &Config, sample: &Sample) -> Result<Vec<Prepared>> {
    let mut all = Vec::new();
    for v in &cfg.variables {
        let path = cfg.input.dir.join(subst(&v.file, &sample.kv()));
        if !path.exists() {
            if v.optional {
                eprintln!("  [skip] optional file missing: {}", path.display());
                continue;
            }
            bail!("missing input file {}", path.display());
        }
        let r = match v.dtype {
            DType::F32 => build::<f32>(v, &path, sample, Arr::F32),
            DType::F64 => build::<f64>(v, &path, sample, Arr::F64),
        };
        all.extend(r.with_context(|| format!("variable '{}' from {}", v.name, path.display()))?);
    }
    Ok(all)
}

// ───────────────────────── coordinates ─────────────────────────

#[derive(Clone)]
pub struct CoordVar {
    pub values: Vec<f64>,
    pub attrs: Vec<(String, String)>,
}

/// Lazily reads 1-D variables out of the template file (cached).
pub struct TemplateCache {
    path: Option<PathBuf>,
    cache: Mutex<HashMap<String, Option<CoordVar>>>,
}

impl TemplateCache {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn get(&self, name: &str) -> Result<Option<CoordVar>> {
        let Some(path) = &self.path else {
            return Ok(None);
        };
        if let Some(hit) = self.cache.lock().unwrap().get(name) {
            return Ok(hit.clone());
        }
        let _guard = NC_LOCK.lock(); // covers netcdf::open + the reads below
                                     // Re-check: another thread may have populated the cache while we
                                     // were waiting on the lock.
        if let Some(hit) = self.cache.lock().unwrap().get(name) {
            return Ok(hit.clone());
        }
        let f =
            netcdf::open(path).with_context(|| format!("opening template {}", path.display()))?;
        let cv = match f.variable(name) {
            Some(var) if var.dimensions().len() == 1 => {
                let values = var.get_values::<f64, _>(..)?;
                let mut attrs = Vec::new();
                for a in var.attributes() {
                    if let Ok(netcdf::AttributeValue::Str(s)) = a.value() {
                        attrs.push((a.name().to_string(), s));
                    }
                }
                Some(CoordVar { values, attrs })
            }
            _ => None,
        };
        self.cache
            .lock()
            .unwrap()
            .insert(name.to_string(), cv.clone());
        Ok(cv)
    }
}

fn time_coord(t: &TimeCfg, date: NaiveDate, n: usize) -> CoordVar {
    CoordVar {
        values: (0..n).map(|i| i as f64 * t.step).collect(),
        attrs: vec![
            ("standard_name".into(), "time".into()),
            ("long_name".into(), "time".into()),
            (
                "units".into(),
                format!(
                    "{} since {} {}",
                    t.unit,
                    date.format("%Y-%m-%d"),
                    t.start_time
                ),
            ),
            ("calendar".into(), t.calendar.clone()),
        ],
    }
}

fn resolve_coord(
    cfg: &Config,
    tpl: &TemplateCache,
    dim: &str,
    n: usize,
    sample: &Sample,
) -> Result<Option<CoordVar>> {
    if let Some(t) = &cfg.time {
        if t.dim == dim {
            let d = sample
                .date
                .ok_or_else(|| anyhow!("no date for sample {}", sample.stem))?;
            return Ok(Some(time_coord(t, d, n)));
        }
    }
    let cc = cfg.coords.get(dim);
    let mut cv = match cc {
        Some(c) if c.start.is_some() || c.step.is_some() => {
            let (s, st) = (c.start.unwrap_or(0.0), c.step.unwrap_or(1.0));
            Some(CoordVar {
                values: (0..n).map(|i| s + i as f64 * st).collect(),
                attrs: vec![],
            })
        }
        Some(Coord {
            from_template: Some(name),
            ..
        }) => Some(tpl.get(name)?.ok_or_else(|| {
            anyhow!("template has no 1-D variable '{name}' (needed for dim '{dim}')")
        })?),
        _ => tpl.get(dim)?,
    };
    if let Some(c) = cv.as_mut() {
        if c.values.len() != n {
            bail!(
                "coordinate '{dim}' has {} values but the data has {n} along that dimension \
                 — is the template grid the same as your model output?",
                c.values.len()
            );
        }
        if let Some(cc) = cc {
            for (k, v) in &cc.attrs {
                c.attrs.retain(|(kk, _)| kk != k);
                c.attrs.push((k.clone(), v.clone()));
            }
        }
    }
    Ok(cv)
}

// ───────────────────────── writing ─────────────────────────

fn put_var<T: netcdf::NcTypeDescriptor + Copy>(
    file: &mut netcdf::FileMut,
    p: &Prepared,
    data: &ArrayD<T>,
    out: &Output,
) -> Result<()> {
    let dims: Vec<&str> = p.dims.iter().map(String::as_str).collect();
    let mut var = file.add_variable::<T>(&p.name, &dims)?;
    if out.compression > 0 && !dims.is_empty() {
        var.set_compression(out.compression, out.shuffle)?;
    }
    for (k, v) in &p.attrs {
        var.put_attribute(k, v.as_str())?;
    }
    let vals: Vec<T> = data.iter().copied().collect(); // already C-order
    var.put_values(&vals, ..)?;
    Ok(())
}

pub fn write(
    cfg: &Config,
    tpl: &TemplateCache,
    sample: &Sample,
    prepared: &[Prepared],
    out_path: &Path,
) -> Result<()> {
    // Collect dimension sizes in order of first appearance, checking consistency.
    let mut dims: Vec<(String, usize)> = Vec::new();
    for p in prepared {
        for (name, &len) in p.dims.iter().zip(p.data.shape()) {
            match dims.iter().find(|(n, _)| n == name) {
                Some((_, l)) if *l != len => bail!(
                    "dimension '{name}' has length {l} elsewhere but {len} in variable '{}'",
                    p.name
                ),
                Some(_) => {}
                None => dims.push((name.clone(), len)),
            }
        }
    }

    if out_path.exists() && !cfg.output.overwrite {
        bail!("{} exists (overwrite = false)", out_path.display());
    }
    // Serialize the whole file lifetime (create -> write vars -> drop/close)
    // against every other thread's netcdf/HDF5 calls, including the
    // TemplateCache::get() call below (safe: NC_LOCK is reentrant).
    let _guard = NC_LOCK.lock();
    let mut file = netcdf::create(out_path)?;
    for (n, l) in &dims {
        file.add_dimension(n, *l)?;
    }

    for (n, l) in &dims {
        if let Some(cv) = resolve_coord(cfg, tpl, n, *l, sample)? {
            let mut v = file.add_variable::<f64>(n, &[n.as_str()])?;
            for (k, val) in &cv.attrs {
                v.put_attribute(k, val.as_str())?;
            }
            v.put_values(&cv.values, ..)?;
        }
    }

    for p in prepared {
        match &p.data {
            Arr::F32(a) => put_var::<f32>(&mut file, p, a, &cfg.output)?,
            Arr::F64(a) => put_var::<f64>(&mut file, p, a, &cfg.output)?,
        }
    }

    let kv = sample.kv();
    for (k, v) in &cfg.global_attrs {
        file.add_attribute(k, subst(v, &kv).as_str())?;
    }
    Ok(())
}

pub fn out_path(cfg: &Config, sample: &Sample) -> PathBuf {
    cfg.output
        .dir
        .join(subst(&cfg.output.pattern, &sample.kv()))
}
