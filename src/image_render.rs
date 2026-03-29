use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use image::ExtendedColorType;
use image::codecs::webp::WebPEncoder;
use kira_spatial_3d::{ScalarField, SpatialDomain, build_heightmap_mesh_mapped};

use crate::cli::{HeightModeArg, LaplacianVizMode, RunArgs, SignalKind};
use crate::error::OrchestratorError;
use crate::pipeline::{GridMapping, build_height_spec, percentile_from_sorted};

pub(crate) fn write_per_gene_signal_webps(
    args: &RunArgs,
    grid: &GridMapping,
    per_gene_signal_dense: &[(String, Vec<f32>)],
) -> Result<(), OrchestratorError> {
    let width = u32::try_from(grid.nx)
        .map_err(|_| OrchestratorError::InvalidInput("image width does not fit u32"))?;
    let height = u32::try_from(grid.ny)
        .map_err(|_| OrchestratorError::InvalidInput("image height does not fit u32"))?;
    let height_spec = build_height_spec(args)?;

    for (gene, signal_dense) in per_gene_signal_dense {
        if signal_dense.len() != grid.nx * grid.ny {
            return Err(OrchestratorError::InvalidInput(
                "per-gene signal length does not match grid shape",
            ));
        }
        let signal_for_image = if args.signal == SignalKind::Laplacian && args.laplacian_sigma > 0.0
        {
            gaussian_blur_2d(signal_dense, grid.nx, grid.ny, args.laplacian_sigma)
        } else {
            signal_dense.clone()
        };

        let mesh_domain =
            SpatialDomain::new(grid.nx, grid.ny, grid.origin_x, grid.origin_y, 1.0, 1.0)?;
        let dense_field = ScalarField::new(mesh_domain, &signal_for_image)?;
        let mesh = build_heightmap_mesh_mapped(&dense_field, height_spec)?;
        let mapped: Vec<f32> = mesh.vertices.iter().map(|p| p[2]).collect();
        let (vmin, vmax) = robust_range(&mapped, 0.5, 99.5);
        let denom = (vmax - vmin).max(1e-6);
        let mut t_map = vec![0.0_f32; grid.nx * grid.ny];
        for (i, &v) in mapped.iter().enumerate() {
            t_map[i] = if v.is_finite() {
                ((v - vmin) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        let flat_mapped = denom <= 1e-6 || t_map.iter().all(|&t| t <= 1e-6);
        if flat_mapped {
            t_map = sparse_positive_unit_map(signal_dense, args.height_mode);
            eprintln!(
                "[timing] gene_webp_sparse_fallback gene={} active={}",
                gene,
                t_map.iter().any(|&t| t > 0.0)
            );
        }
        let nnz = t_map.iter().filter(|&&t| t > 0.0).count();
        let sparse_ratio = nnz as f32 / (t_map.len().max(1) as f32);
        if sparse_ratio < 0.05 {
            let dilated = max_filter_radius(&t_map, grid.nx, grid.ny, 3);
            t_map = box_blur3x3(&dilated, grid.nx, grid.ny);
        }
        let (shade_values, shade_vmin, shade_denom): (Vec<f32>, f32, f32) = if flat_mapped {
            (t_map.clone(), 0.0, 1.0)
        } else {
            (mapped, vmin, denom)
        };
        let mut lap_scale = 1.0_f32;
        if args.signal == SignalKind::Laplacian {
            let mut abs_vals = signal_for_image
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .map(f32::abs)
                .filter(|v| *v > 0.0)
                .collect::<Vec<_>>();
            if !abs_vals.is_empty() {
                abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                lap_scale = percentile_from_sorted(&abs_vals, 99.5).max(1e-6);
            }
        }
        let lap_edge_eps = (lap_scale * 0.01).max(1e-6);

        let mut rgb = vec![0_u8; grid.nx * grid.ny * 3];
        let light = normalize3([0.5, 0.8, 0.3]);
        let view_dir = [0.0_f32, 0.0, 1.0];
        let slope_p90 =
            estimate_slope_p90(&shade_values, grid.nx, grid.ny, shade_vmin, shade_denom);
        let slope_gain = (0.22 / slope_p90.max(1e-4)).clamp(1.0, 80.0);
        let z_exag = (args.z_scale.max(1e-3) * 0.20 * slope_gain).clamp(0.5, 4096.0);
        eprintln!(
            "[timing] gene_webp_shading gene={} slope_p90={:.6} z_exag={:.3}",
            gene, slope_p90, z_exag
        );
        for y in 0..grid.ny {
            for x in 0..grid.nx {
                let i = y * grid.nx + x;
                let t = t_map[i];
                let zero_cross = args.signal == SignalKind::Laplacian
                    && args.laplacian_viz != LaplacianVizMode::Diverging
                    && has_zero_crossing(&signal_for_image, grid.nx, grid.ny, x, y, lap_edge_eps);
                let base = if args.signal == SignalKind::Laplacian {
                    match args.laplacian_viz {
                        LaplacianVizMode::ZeroCrossings => {
                            if zero_cross {
                                [0.95, 0.96, 0.99]
                            } else {
                                [0.10, 0.11, 0.13]
                            }
                        }
                        LaplacianVizMode::Diverging | LaplacianVizMode::Both => {
                            let s = (signal_for_image[i] / lap_scale).clamp(-1.0, 1.0);
                            let mut c = laplacian_diverging_colormap(s, t);
                            if args.laplacian_viz == LaplacianVizMode::Both && zero_cross {
                                c = lerp3(c, [0.98, 0.98, 1.0], 0.55);
                            }
                            c
                        }
                    }
                } else {
                    let t_vis = ((1.0 + 40.0 * t).ln() / (1.0 + 40.0_f32).ln()).clamp(0.0, 1.0);
                    viewer_colormap_f32(t_vis, 0.9)
                };
                let h_l = normalized_height_at(
                    &shade_values,
                    grid.nx,
                    grid.ny,
                    x.saturating_sub(1),
                    y,
                    shade_vmin,
                    shade_denom,
                );
                let h_r = normalized_height_at(
                    &shade_values,
                    grid.nx,
                    grid.ny,
                    (x + 1).min(grid.nx.saturating_sub(1)),
                    y,
                    shade_vmin,
                    shade_denom,
                );
                let h_d = normalized_height_at(
                    &shade_values,
                    grid.nx,
                    grid.ny,
                    x,
                    y.saturating_sub(1),
                    shade_vmin,
                    shade_denom,
                );
                let h_u = normalized_height_at(
                    &shade_values,
                    grid.nx,
                    grid.ny,
                    x,
                    (y + 1).min(grid.ny.saturating_sub(1)),
                    shade_vmin,
                    shade_denom,
                );
                let dzdx = (h_r - h_l) * z_exag;
                let dzdy = (h_u - h_d) * z_exag;
                let n = normalize3([-dzdx, -dzdy, 1.0]);

                let lambert = dot3(n, light).max(0.0);
                let half_vec = normalize3([
                    light[0] + view_dir[0],
                    light[1] + view_dir[1],
                    light[2] + view_dir[2],
                ]);
                let spec = dot3(n, half_vec).max(0.0).powf(32.0) * 0.18;
                let fresnel = (1.0 - dot3(n, view_dir).max(0.0)).powf(3.0) * 0.2;
                let shade = if args.signal == SignalKind::Laplacian
                    && args.laplacian_viz == LaplacianVizMode::ZeroCrossings
                {
                    1.0
                } else {
                    (0.16 + lambert * 1.45 + spec * 1.1 + fresnel * 0.6).clamp(0.0, 2.2)
                };

                let lit = [
                    (base[0] * shade).clamp(0.0, 1.0),
                    (base[1] * shade).clamp(0.0, 1.0),
                    (base[2] * shade).clamp(0.0, 1.0),
                ];
                let off = i * 3;
                rgb[off] = (lit[0] * 255.0).clamp(0.0, 255.0).round() as u8;
                rgb[off + 1] = (lit[1] * 255.0).clamp(0.0, 255.0).round() as u8;
                rgb[off + 2] = (lit[2] * 255.0).clamp(0.0, 255.0).round() as u8;
            }
        }

        let path = args
            .out_dir
            .join(format!("gene.{}.webp", sanitize_gene_filename(gene)));
        save_lossless_webp(&path, &rgb, width, height)?;
    }
    Ok(())
}

fn save_lossless_webp(
    path: &Path,
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<(), OrchestratorError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = WebPEncoder::new_lossless(writer);
    encoder.encode(rgb, width, height, ExtendedColorType::Rgb8)?;
    Ok(())
}

fn sanitize_gene_filename(gene: &str) -> String {
    let mut out = String::with_capacity(gene.len());
    for ch in gene.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "gene".to_string()
    } else {
        out
    }
}

fn viewer_colormap_f32(t_in: f32, gamma: f32) -> [f32; 3] {
    let t = t_in.clamp(0.0, 1.0).powf(gamma.max(0.05));
    let c0 = [0.12_f32, 0.19, 0.72];
    let c1 = [0.11_f32, 0.72, 0.71];
    let c2 = [0.97_f32, 0.90, 0.23];
    let c3 = [0.86_f32, 0.12, 0.14];
    if t < 0.33 {
        lerp3(c0, c1, t / 0.33)
    } else if t < 0.66 {
        lerp3(c1, c2, (t - 0.33) / 0.33)
    } else {
        lerp3(c2, c3, (t - 0.66) / 0.34)
    }
}

fn laplacian_diverging_colormap(signed_norm: f32, amp: f32) -> [f32; 3] {
    let s = signed_norm.clamp(-1.0, 1.0);
    let a = s.abs().powf(0.85);
    let amp_gain = (0.35 + 0.65 * amp.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let neutral = [0.44_f32, 0.45, 0.48];
    let neg = [0.12_f32, 0.37, 0.94];
    let pos = [0.95_f32, 0.22, 0.18];
    let target = if s < 0.0 { neg } else { pos };
    lerp3(neutral, target, (a * amp_gain).clamp(0.0, 1.0))
}

fn has_zero_crossing(field: &[f32], nx: usize, ny: usize, x: usize, y: usize, eps: f32) -> bool {
    if nx == 0 || ny == 0 {
        return false;
    }
    let i = y * nx + x;
    let c = field[i];
    if !c.is_finite() {
        return false;
    }
    let check_neighbor = |xx: usize, yy: usize| -> bool {
        let n = field[yy * nx + xx];
        n.is_finite() && ((c > eps && n < -eps) || (c < -eps && n > eps))
    };
    let xl = x.saturating_sub(1);
    let xr = (x + 1).min(nx.saturating_sub(1));
    let yd = y.saturating_sub(1);
    let yu = (y + 1).min(ny.saturating_sub(1));
    check_neighbor(xl, y) || check_neighbor(xr, y) || check_neighbor(x, yd) || check_neighbor(x, yu)
}

fn gaussian_blur_2d(src: &[f32], nx: usize, ny: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 || nx == 0 || ny == 0 || src.is_empty() {
        return src.to_vec();
    }
    let radius = (sigma * 3.0).ceil() as usize;
    if radius == 0 {
        return src.to_vec();
    }
    let mut kernel = Vec::<f32>::with_capacity(radius * 2 + 1);
    let mut sum = 0.0_f32;
    for k in 0..=(radius * 2) {
        let dx = (k as isize - radius as isize) as f32;
        let w = (-0.5 * (dx / sigma).powi(2)).exp();
        kernel.push(w);
        sum += w;
    }
    for w in &mut kernel {
        *w /= sum.max(1e-12);
    }

    let mut tmp = vec![0.0_f32; src.len()];
    for y in 0..ny {
        for x in 0..nx {
            let mut acc = 0.0_f32;
            for (k, &w) in kernel.iter().enumerate() {
                let off = k as isize - radius as isize;
                let xx = (x as isize + off).clamp(0, nx.saturating_sub(1) as isize) as usize;
                let v = src[y * nx + xx];
                if v.is_finite() {
                    acc += v * w;
                }
            }
            tmp[y * nx + x] = acc;
        }
    }

    let mut out = vec![0.0_f32; src.len()];
    for y in 0..ny {
        for x in 0..nx {
            let mut acc = 0.0_f32;
            for (k, &w) in kernel.iter().enumerate() {
                let off = k as isize - radius as isize;
                let yy = (y as isize + off).clamp(0, ny.saturating_sub(1) as isize) as usize;
                let v = tmp[yy * nx + x];
                if v.is_finite() {
                    acc += v * w;
                }
            }
            out[y * nx + x] = acc;
        }
    }
    out
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let s = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    ]
}

fn normalized_height_at(
    mapped: &[f32],
    nx: usize,
    ny: usize,
    x: usize,
    y: usize,
    vmin: f32,
    denom: f32,
) -> f32 {
    let xx = x.min(nx.saturating_sub(1));
    let yy = y.min(ny.saturating_sub(1));
    let v = mapped[yy * nx + xx];
    if v.is_finite() {
        ((v - vmin) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn estimate_slope_p90(mapped: &[f32], nx: usize, ny: usize, vmin: f32, denom: f32) -> f32 {
    if nx == 0 || ny == 0 {
        return 0.0;
    }
    let mut mags = Vec::<f32>::with_capacity(nx.saturating_mul(ny));
    for y in 0..ny {
        for x in 0..nx {
            let h_l = normalized_height_at(mapped, nx, ny, x.saturating_sub(1), y, vmin, denom);
            let h_r = normalized_height_at(
                mapped,
                nx,
                ny,
                (x + 1).min(nx.saturating_sub(1)),
                y,
                vmin,
                denom,
            );
            let h_d = normalized_height_at(mapped, nx, ny, x, y.saturating_sub(1), vmin, denom);
            let h_u = normalized_height_at(
                mapped,
                nx,
                ny,
                x,
                (y + 1).min(ny.saturating_sub(1)),
                vmin,
                denom,
            );
            let dx = h_r - h_l;
            let dy = h_u - h_d;
            mags.push((dx * dx + dy * dy).sqrt());
        }
    }
    if mags.is_empty() {
        return 0.0;
    }
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_from_sorted(&mags, 90.0)
}

fn max_filter_radius(src: &[f32], nx: usize, ny: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; src.len()];
    for y in 0..ny {
        for x in 0..nx {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(nx.saturating_sub(1));
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius).min(ny.saturating_sub(1));
            let mut m = 0.0_f32;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    m = m.max(src[yy * nx + xx]);
                }
            }
            out[y * nx + x] = m;
        }
    }
    out
}

fn box_blur3x3(src: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; src.len()];
    for y in 0..ny {
        for x in 0..nx {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(ny.saturating_sub(1));
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(nx.saturating_sub(1));
            let mut sum = 0.0_f32;
            let mut n = 0_u32;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    sum += src[yy * nx + xx];
                    n += 1;
                }
            }
            out[y * nx + x] = if n > 0 { sum / n as f32 } else { 0.0 };
        }
    }
    out
}

fn robust_range(values: &[f32], lo_pct: f32, hi_pct: f32) -> (f32, f32) {
    let mut finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    let lo_i = ((lo_pct.clamp(0.0, 100.0) / 100.0) * (n.saturating_sub(1) as f32)).round() as usize;
    let hi_i = ((hi_pct.clamp(0.0, 100.0) / 100.0) * (n.saturating_sub(1) as f32)).round() as usize;
    let lo = finite[lo_i.min(n - 1)];
    let hi = finite[hi_i.min(n - 1)].max(lo + 1e-6);
    (lo, hi)
}

fn sparse_positive_unit_map(values: &[f32], mode: HeightModeArg) -> Vec<f32> {
    let transformed: Vec<f32> = values
        .iter()
        .copied()
        .map(|v| {
            if !v.is_finite() {
                return 0.0;
            }
            let base = match mode {
                HeightModeArg::Raw | HeightModeArg::Signed => v,
                HeightModeArg::Abs => v.abs(),
            };
            if base.is_finite() && base > 0.0 {
                base
            } else {
                0.0
            }
        })
        .collect();

    let mut nz = transformed
        .iter()
        .copied()
        .filter(|v| *v > 0.0 && v.is_finite())
        .collect::<Vec<_>>();
    if nz.is_empty() {
        return vec![0.0; values.len()];
    }
    nz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = percentile_from_sorted(&nz, 5.0);
    let hi = percentile_from_sorted(&nz, 99.5).max(lo + 1e-6);
    let scale = (hi - lo).max(1e-6);
    transformed
        .into_iter()
        .map(|v| {
            if v > 0.0 {
                ((v - lo) / scale).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}
