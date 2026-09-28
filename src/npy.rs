//! Minimal, dtype-agnostic `.npy` reader.
//!
//! Reads any numeric dtype (f2/f4/f8, i1-i8, u1-u8, bool), either byte order,
//! C or Fortran layout, and converts to the requested output element type.

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use regex::Regex;
use std::path::Path;

/// Output element types we can convert into.
pub trait Elem: Copy + Send + Sync + 'static {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}
impl Elem for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl Elem for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
    kind: char,
    size: usize,
    big_endian: bool,
    data_offset: usize,
}

pub fn parse_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file (bad magic)");
    }
    let major = bytes[6];
    let (hlen, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated header");
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => bail!("unsupported .npy version {v}"),
    };
    let end = start + hlen;
    if bytes.len() < end {
        bail!("truncated header");
    }
    let text = std::str::from_utf8(&bytes[start..end]).context("header is not utf-8")?;

    let descr = Regex::new(r#"'descr'\s*:\s*'([^']*)'"#)?
        .captures(text)
        .ok_or_else(|| anyhow!("unsupported dtype (structured/object arrays are not supported)"))?[1]
        .to_string();
    let fortran_order = &Regex::new(r"'fortran_order'\s*:\s*(True|False)")?
        .captures(text)
        .ok_or_else(|| anyhow!("missing fortran_order"))?[1]
        == "True";
    let shape_txt = Regex::new(r"'shape'\s*:\s*\(([^)]*)\)")?
        .captures(text)
        .ok_or_else(|| anyhow!("missing shape"))?[1]
        .to_string();
    let shape = shape_txt
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().context("bad shape entry"))
        .collect::<Result<Vec<_>>>()?;

    let mut chars = descr.chars();
    let first = chars.next().ok_or_else(|| anyhow!("empty descr"))?;
    let (big_endian, rest) = match first {
        '<' | '|' | '=' => (first == '=' && cfg!(target_endian = "big"), &descr[1..]),
        '>' => (true, &descr[1..]),
        _ => (cfg!(target_endian = "big"), &descr[..]),
    };
    let kind = rest.chars().next().ok_or_else(|| anyhow!("bad descr {descr}"))?;
    let size: usize = rest[1..].parse().with_context(|| format!("bad descr {descr}"))?;
    if !matches!(kind, 'f' | 'i' | 'u' | 'b') {
        bail!("unsupported dtype '{descr}' (only numeric/bool arrays are supported)");
    }
    Ok(Header { descr, fortran_order, shape, kind, size, big_endian, data_offset: end })
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 => if frac == 0.0 { sign * f64::INFINITY } else { f64::NAN },
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

fn decode<T: Elem>(h: &Header, raw: &[u8]) -> Result<Vec<T>> {
    let be = h.big_endian;
    macro_rules! dec {
        ($ty:ty, $n:literal, $conv:expr) => {
            raw.chunks_exact($n)
                .map(|c| {
                    let a: [u8; $n] = c.try_into().unwrap();
                    let v = if be { <$ty>::from_be_bytes(a) } else { <$ty>::from_le_bytes(a) };
                    T::from_f64($conv(v))
                })
                .collect()
        };
    }
    Ok(match (h.kind, h.size) {
        ('f', 2) => dec!(u16, 2, f16_to_f64),
        ('f', 4) => dec!(f32, 4, |v: f32| v as f64),
        ('f', 8) => dec!(f64, 8, |v: f64| v),
        ('i', 1) => dec!(i8, 1, |v: i8| v as f64),
        ('i', 2) => dec!(i16, 2, |v: i16| v as f64),
        ('i', 4) => dec!(i32, 4, |v: i32| v as f64),
        ('i', 8) => dec!(i64, 8, |v: i64| v as f64),
        ('u', 1) | ('b', 1) => dec!(u8, 1, |v: u8| v as f64),
        ('u', 2) => dec!(u16, 2, |v: u16| v as f64),
        ('u', 4) => dec!(u32, 4, |v: u32| v as f64),
        ('u', 8) => dec!(u64, 8, |v: u64| v as f64),
        _ => bail!("unsupported dtype '{}'", h.descr),
    })
}

/// Read an `.npy` file into a C-ordered array of `T`.
pub fn read<T: Elem>(path: &Path) -> Result<ArrayD<T>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let h = parse_header(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    let n: usize = h.shape.iter().product();
    let need = h.data_offset + n * h.size;
    if bytes.len() < need {
        bail!("{}: file truncated ({} < {} bytes)", path.display(), bytes.len(), need);
    }
    let data = decode::<T>(&h, &bytes[h.data_offset..need])?;
    let arr = if h.fortran_order {
        ArrayD::from_shape_vec(IxDyn(&h.shape).f(), data)?
            .as_standard_layout()
            .into_owned()
    } else {
        ArrayD::from_shape_vec(IxDyn(&h.shape), data)?
    };
    Ok(arr)
}

/// Header-only peek (used by `inspect`).
pub fn peek(path: &Path) -> Result<Header> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 65536];
    let n = f.read(&mut buf)?;
    parse_header(&buf[..n])
}
