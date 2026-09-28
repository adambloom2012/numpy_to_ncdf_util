mod config;
mod convert;
mod npy;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use config::*;
use rayon::prelude::*;
use regex::Regex;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "npy2nc", version, about = "Convert NumPy .npy files to NetCDF")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Batch-convert samples described by a TOML config
    Run {
        #[arg(short, long)]
        config: PathBuf,
        /// Override input.dir
        #[arg(long)]
        input_dir: Option<PathBuf>,
        /// Override output.dir
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
        /// Override template.path
        #[arg(long)]
        template: Option<PathBuf>,
        /// Only convert stems matching this regex
        #[arg(long)]
        only: Option<String>,
        /// Parallel worker threads (default: all cores)
        #[arg(short, long)]
        jobs: Option<usize>,
        /// Load + validate everything and print a summary, but write nothing
        #[arg(long)]
        dry_run: bool,
    },
    /// One-off conversion without a config file
    Quick {
        /// Output .nc file
        #[arg(short, long)]
        output: PathBuf,
        /// NAME=FILE.npy:dim1,dim2,...[:units]   (repeatable)
        #[arg(long = "var", required = true)]
        vars: Vec<String>,
        /// Template NetCDF for lat/lon etc. (dims copied by name)
        #[arg(long)]
        template: Option<PathBuf>,
        /// Start date YYYY-MM-DD, enables a `time` coordinate
        #[arg(long)]
        date: Option<String>,
        #[arg(long, default_value = "hours")]
        time_unit: String,
        #[arg(long, default_value_t = 1.0)]
        time_step: f64,
        #[arg(long, default_value = "time")]
        time_dim: String,
        #[arg(long, default_value_t = 4)]
        compression: i32,
        #[arg(long, default_value = "f32")]
        dtype: String,
    },
    /// Print dtype / shape / basic stats of .npy files
    Inspect {
        files: Vec<PathBuf>,
        /// Header only (skip min/max/mean scan)
        #[arg(long)]
        fast: bool,
    },
    /// Print a fully commented example config
    ExampleConfig,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::ExampleConfig => {
            print!("{EXAMPLE}");
            Ok(())
        }
        Cmd::Inspect { files, fast } => inspect(files, fast),
        Cmd::Quick { output, vars, template, date, time_unit, time_step, time_dim, compression, dtype } => {
            let cfg = quick_config(output, vars, template, date, time_unit, time_step, time_dim, compression, &dtype)?;
            let stem = cfg.output.pattern.rsplit_once('.').map_or(cfg.output.pattern.clone(), |x| x.0.to_string());
            run_config(cfg, None, Some(vec![stem]), false)
        }
        Cmd::Run { config, input_dir, output_dir, template, only, jobs, dry_run } => {
            let txt = std::fs::read_to_string(&config)
                .with_context(|| format!("reading {}", config.display()))?;
            let mut cfg: Config =
                toml::from_str(&txt).with_context(|| format!("parsing {}", config.display()))?;
            if let Some(d) = input_dir { cfg.input.dir = d; }
            if let Some(d) = output_dir { cfg.output.dir = d; }
            if let Some(t) = template { cfg.template = Some(Template { path: t }); }
            if let Some(j) = jobs {
                rayon::ThreadPoolBuilder::new().num_threads(j).build_global()?;
            }
            let only = only.map(|s| Regex::new(&s)).transpose()?;
            run_config(cfg, only, None, dry_run)
        }
    }
}

fn run_config(cfg: Config, only: Option<Regex>, fixed_stems: Option<Vec<String>>, dry_run: bool) -> Result<()> {
    cfg.validate()?;
    if let Some(t) = &cfg.time {
        if !["seconds", "minutes", "hours", "days"].contains(&t.unit.as_str()) {
            bail!("time.unit must be seconds|minutes|hours|days, got '{}'", t.unit);
        }
    }
    let date_re = cfg
        .time
        .as_ref()
        .and_then(|t| t.date_regex.as_deref())
        .map(Regex::new)
        .transpose()
        .context("invalid time.date_regex")?;

    let stems = match fixed_stems {
        Some(s) => s,
        None => convert::discover(&cfg, only.as_ref())?,
    };
    if stems.is_empty() {
        bail!(
            "no files in {} ending with '{}'",
            cfg.input.dir.display(),
            cfg.input.discover
        );
    }
    if !dry_run {
        std::fs::create_dir_all(&cfg.output.dir)?;
    }
    let tpl = convert::TemplateCache::new(cfg.template.as_ref().map(|t| t.path.clone()));
    println!("{} sample(s) found{}", stems.len(), if dry_run { " (dry run)" } else { "" });

    let t0 = Instant::now();
    let results: Vec<(String, Result<()>)> = stems
        .par_iter()
        .map(|stem| {
            let r = (|| -> Result<()> {
                let sample = convert::make_sample(&cfg, stem, date_re.as_ref())?;
                let prepared = convert::prepare(&cfg, &sample)?;
                if dry_run {
                    println!("{stem}:");
                    for p in &prepared {
                        println!("  {:<28} dims={:?} shape={:?}", p.name, p.dims, p.data.shape());
                    }
                    return Ok(());
                }
                let out = convert::out_path(&cfg, &sample);
                convert::write(&cfg, &tpl, &sample, &prepared, &out)?;
                println!("  {stem} -> {}", out.display());
                Ok(())
            })();
            (stem.clone(), r)
        })
        .collect();

    let failed: Vec<_> = results.iter().filter(|(_, r)| r.is_err()).collect();
    for (stem, r) in &failed {
        eprintln!("[FAILED] {stem}: {:#}", r.as_ref().unwrap_err());
    }
    println!(
        "done: {} ok, {} failed in {:.2}s",
        results.len() - failed.len(),
        failed.len(),
        t0.elapsed().as_secs_f64()
    );
    if failed.is_empty() { Ok(()) } else { Err(anyhow!("{} sample(s) failed", failed.len())) }
}

#[allow(clippy::too_many_arguments)]
fn quick_config(
    output: PathBuf,
    vars: Vec<String>,
    template: Option<PathBuf>,
    date: Option<String>,
    time_unit: String,
    time_step: f64,
    time_dim: String,
    compression: i32,
    dtype: &str,
) -> Result<Config> {
    let dtype = match dtype {
        "f32" => DType::F32,
        "f64" => DType::F64,
        o => bail!("--dtype must be f32 or f64, got {o}"),
    };
    let mut variables = Vec::new();
    for spec in vars {
        let (name, rest) = spec.split_once('=').ok_or_else(|| anyhow!("bad --var '{spec}'"))?;
        let mut parts = rest.splitn(3, ':');
        let file = parts.next().unwrap();
        let dims = parts
            .next()
            .ok_or_else(|| anyhow!("--var '{spec}' needs :dim1,dim2,..."))?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        variables.push(Variable {
            name: name.into(),
            file: file.into(),
            dims,
            dtype,
            select: vec![],
            split: None,
            permute: None,
            scale: None,
            offset: None,
            units: parts.next().map(String::from),
            long_name: None,
            attrs: BTreeMap::new(),
            optional: false,
        });
    }
    let stem = output.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let dir = output.parent().map(|p| p.to_path_buf()).filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("."));
    let fname = output.file_name().unwrap().to_string_lossy().into_owned();
    Ok(Config {
        // quick mode passes its stem explicitly, so `discover` is unused
        input: Input { dir: PathBuf::from("."), discover: String::new(), exclude_suffixes: vec![] },
        output: Output { dir, pattern: fname, compression, shuffle: true, overwrite: true },
        template: template.map(|path| Template { path }),
        coords: BTreeMap::new(),
        time: date.map(|d| TimeCfg {
            dim: time_dim,
            date_regex: None,
            start: Some(d),
            date_format: "%Y%m%d".into(),
            step: time_step,
            unit: time_unit,
            start_time: "00:00:00".into(),
            calendar: "standard".into(),
        }),
        global_attrs: BTreeMap::from([("source".into(), format!("npy2nc quick ({stem})"))]),
        variables,
    })
}

fn inspect(files: Vec<PathBuf>, fast: bool) -> Result<()> {
    for f in files {
        match npy::peek(&f) {
            Err(e) => println!("{}: ERROR {e:#}", f.display()),
            Ok(h) => {
                print!("{}: dtype={} shape={:?}{}", f.display(), h.descr, h.shape,
                       if h.fortran_order { " (fortran order)" } else { "" });
                if !fast {
                    let a = npy::read::<f64>(&f)?;
                    let (mut mn, mut mx, mut sum, mut nan, mut n) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize, 0usize);
                    for &x in a.iter() {
                        if x.is_nan() { nan += 1; continue; }
                        mn = mn.min(x); mx = mx.max(x); sum += x; n += 1;
                    }
                    print!("\n    min={mn:.6e} max={mx:.6e} mean={:.6e} nan={nan}", sum / n.max(1) as f64);
                }
                println!();
            }
        }
    }
    Ok(())
}
