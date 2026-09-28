//! TOML configuration schema.

use anyhow::{bail, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub input: Input,
    pub output: Output,
    /// Optional NetCDF file supplying coordinate arrays (lat/lon/...).
    pub template: Option<Template>,
    /// Per-dimension coordinate definitions, keyed by dimension name.
    #[serde(default)]
    pub coords: BTreeMap<String, Coord>,
    /// Optional time axis, generated from a date parsed out of the sample name.
    pub time: Option<TimeCfg>,
    /// Global attributes. `{stem}` and `{date}` are substituted.
    #[serde(default)]
    pub global_attrs: BTreeMap<String, String>,
    #[serde(rename = "variable")]
    pub variables: Vec<Variable>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub dir: PathBuf,
    /// Samples are discovered as files in `dir` ending with this suffix;
    /// the rest of the filename is the sample `stem`. e.g. "_pred.npy".
    pub discover: String,
    /// Skip discovered files ending with any of these suffixes.
    #[serde(default)]
    pub exclude_suffixes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub dir: PathBuf,
    /// Output filename pattern. `{stem}` and `{date}` are substituted.
    #[serde(default = "d_pattern")]
    pub pattern: String,
    /// zlib level 0-9 (0 disables compression).
    #[serde(default = "d_compression")]
    pub compression: i32,
    #[serde(default = "d_true")]
    pub shuffle: bool,
    #[serde(default = "d_true")]
    pub overwrite: bool,
}
fn d_pattern() -> String { "{stem}.nc4".into() }
fn d_compression() -> i32 { 4 }
fn d_true() -> bool { true }

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Template {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coord {
    /// Copy this 1-D variable from the template (default: same name as the dim,
    /// if the template has one).
    pub from_template: Option<String>,
    /// Or generate `start + i*step`.
    pub start: Option<f64>,
    pub step: Option<f64>,
    /// Attributes to write (override anything copied from the template).
    #[serde(default)]
    pub attrs: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeCfg {
    #[serde(default = "d_time_dim")]
    pub dim: String,
    /// Regex applied to the sample stem; group 1 (or whole match) is the date.
    pub date_regex: Option<String>,
    /// Fixed start date (`YYYY-MM-DD`) if it is not encoded in the filename.
    pub start: Option<String>,
    #[serde(default = "d_date_fmt")]
    pub date_format: String,
    #[serde(default = "d_one")]
    pub step: f64,
    /// seconds | minutes | hours | days
    #[serde(default = "d_hours")]
    pub unit: String,
    #[serde(default = "d_start_time")]
    pub start_time: String,
    #[serde(default = "d_calendar")]
    pub calendar: String,
}
fn d_time_dim() -> String { "time".into() }
fn d_date_fmt() -> String { "%Y%m%d".into() }
fn d_one() -> f64 { 1.0 }
fn d_hours() -> String { "hours".into() }
fn d_start_time() -> String { "00:00:00".into() }
fn d_calendar() -> String { "standard".into() }

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variable {
    /// Output variable name. With `split`, must contain `{label}`.
    pub name: String,
    /// Input file relative to `input.dir`. `{stem}` is substituted.
    pub file: String,
    /// Dimension names of the array *after* select/split/permute.
    pub dims: Vec<String>,
    /// f32 (default) or f64
    #[serde(default)]
    pub dtype: DType,
    /// Fix an axis at an index (removing it), applied in order.
    #[serde(default)]
    pub select: Vec<Select>,
    /// Split one axis into several variables, one per label.
    pub split: Option<Split>,
    /// Reorder remaining axes (numpy `transpose` semantics) to match `dims`.
    pub permute: Option<Vec<usize>>,
    /// value = value * scale + offset  (applied last, e.g. unit conversion)
    pub scale: Option<f64>,
    pub offset: Option<f64>,
    /// Attribute templates. `{label}` / `{long}` available with `split`.
    pub units: Option<String>,
    pub long_name: Option<String>,
    #[serde(default)]
    pub attrs: BTreeMap<String, String>,
    /// If the file is missing, skip this variable instead of failing the sample.
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    #[default]
    F32,
    F64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Select {
    pub axis: usize,
    /// Negative counts from the end.
    pub index: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub axis: usize,
    pub labels: Vec<String>,
    /// Optional label -> descriptive name, available as `{long}`.
    #[serde(default)]
    pub long_names: BTreeMap<String, String>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.variables.is_empty() {
            bail!("config has no [[variable]] entries");
        }
        for v in &self.variables {
            if v.split.is_some() && !v.name.contains("{label}") {
                bail!("variable '{}' uses split but its name has no {{label}}", v.name);
            }
            if let Some(p) = &v.permute {
                if p.len() != v.dims.len() {
                    bail!("variable '{}': permute length must equal dims length", v.name);
                }
            }
        }
        if let Some(t) = &self.time {
            if t.date_regex.is_none() && t.start.is_none() {
                bail!("[time] needs either date_regex or start");
            }
        }
        Ok(())
    }
}

pub const EXAMPLE: &str = r#"# npy2nc example config: multi-species column + surface predictions.
# Run:  npy2nc run -c config.toml

[input]
dir = "/data/preds"
discover = "_pred.npy"                 # samples = files ending in this; stem = the rest
# NB: "_pred.npy" also matches "*_surface_pred.npy" / "*_emis_pred.npy" -> exclude them
exclude_suffixes = ["_surface_pred.npy", "_emis_pred.npy"]

[output]
dir = "/data/preds/netcdf"
pattern = "{stem}.nc4"
compression = 4

[template]                             # lat/lon copied from here
path = "/data/GEOSChem.MetVars.20200305_0000z.nc4"

# Optional: override / generate coordinates. By default any dim whose name
# exists as a 1-D variable in the template is copied automatically.
# [coords.lat]
# from_template = "lat"
# [coords.z]
# start = 0
# step = 1
# attrs = { units = "level" }

[time]                                 # hourly axis, date parsed from the stem
date_regex = '(\d{8})$'
date_format = "%Y%m%d"
step = 1
unit = "hours"

[global_attrs]
title = "DeepMMF multi-species predictions"
date = "{date}"
Conventions = "CF-1.8"

# (24, 7, H, W) -> 7 variables: NO2_VCD_pred, O3_VCD_pred, ...
[[variable]]
name = "{label}_VCD_pred"
file = "{stem}_pred.npy"
dims = ["time", "lat", "lon"]
units = "molec cm-2"
long_name = "Model predicted {long} vertical column density"
[variable.split]
axis = 1
labels = ["NO2", "O3", "SO2", "CO", "CH2O", "OH", "ISOP"]
long_names = { NO2 = "nitrogen dioxide", O3 = "ozone", SO2 = "sulfur dioxide" }

[[variable]]
name = "{label}_VCD_target"
file = "{stem}_target.npy"
dims = ["time", "lat", "lon"]
dtype = "f64"                          # store as double
units = "molec cm-2"
long_name = "GEOS-Chem simulated {long} vertical column density"
[variable.split]
axis = 1
labels = ["NO2", "O3", "SO2", "CO", "CH2O", "OH", "ISOP"]

# (1, H, W) single channel: pick index 0 of axis 0 -> (H, W)
[[variable]]
name = "NOx_emis_pred"
file = "{stem}_emis_pred.npy"
dims = ["lat", "lon"]
select = [{ axis = 0, index = 0 }]     # (1,H,W) -> (H,W)
units = "kg m-2 s-1"
optional = true
"#;
