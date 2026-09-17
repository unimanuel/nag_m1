"""Create a high-contrast PDF summary of the Waymo ellipsoid optimisation study."""

from pathlib import Path


OUT = Path(__file__).with_name("2026-09-17_ellipsoid_optimization_report.pdf")
W, H = 792, 612
LEFT, RIGHT = 36, 756
NAVY = (0.06, 0.18, 0.34)
BLACK = (0.03, 0.03, 0.03)
GREY = (0.26, 0.26, 0.26)


def esc(value):
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text(x, y, value, size=9, bold=False, color=None):
    font = "F2" if bold else "F1"
    color = color or BLACK
    return (
        "BT\n"
        + "{:.3f} {:.3f} {:.3f} rg\n".format(*color)
        + "/{} {} Tf\n{:.1f} {:.1f} Td\n({}) Tj\nET\n".format(font, size, x, y, esc(value))
    )


def line(x1, y1, x2, y2, width=0.5, gray=0.72):
    return "{:.3f} G\n{:.2f} w\n{:.1f} {:.1f} m\n{:.1f} {:.1f} l\nS\n".format(
        gray, width, x1, y1, x2, y2
    )


def fill(x, y, width, height, gray):
    return "{:.3f} g\n{:.1f} {:.1f} {:.1f} {:.1f} re\nf\n".format(gray, x, y, width, height)


def wrapped(x, y, value, width, size=9, leading=12):
    """Draw short ASCII prose with an intentionally conservative wrap estimate."""
    capacity = max(12, int(width / (size * 0.51)))
    words = value.split()
    rows, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if current and len(candidate) > capacity:
            rows.append(current)
            current = word
        else:
            current = candidate
    if current:
        rows.append(current)
    output = ""
    for row in rows:
        output += text(x, y, row, size)
        y -= leading
    return output, y


def section(output, y, title):
    return output + text(LEFT, y, title, 12, True, NAVY), y - 17


def table(y, headers, rows, widths, font_size=7.6, row_height=20):
    """Draw black table text over white and very light grey rows."""
    total = sum(widths)
    assert total <= RIGHT - LEFT
    output = fill(LEFT, y - row_height + 2, total, row_height, 0.82)
    x = LEFT + 4
    for header, width in zip(headers, widths):
        output += text(x, y - 12, header, font_size, True, BLACK)
        x += width
    y -= row_height
    for index, row in enumerate(rows):
        if index % 2:
            output += fill(LEFT, y - row_height + 2, total, row_height, 0.96)
        x = LEFT + 4
        for value, width in zip(row, widths):
            output += text(x, y - 12, value, font_size, False, BLACK)
            x += width
        output += line(LEFT, y - row_height + 2, LEFT + total, y - row_height + 2, 0.3, 0.82)
        y -= row_height
    return output, y


def footer(output, page, note):
    output += line(LEFT, 42, RIGHT, 42, 0.45, 0.72)
    output += text(LEFT, 28, note, 7.5, color=GREY)
    output += text(RIGHT - 75, 28, "Page {} of 6".format(page), 7.5, color=GREY)
    return output


pages = []

# 1. Scope and conclusion.
p = ""
p += text(LEFT, 568, "Waymo ego-background ellipsoid optimisation report", 19, True, NAVY)
p += text(LEFT, 545, "Results through 17 September 2026 - high-contrast reference report", 10, color=GREY)
p += line(LEFT, 535, RIGHT, 535, 1.2, 0.25)
y = 511
p, y = section(p, y, "Purpose and scope")
paragraph, y = wrapped(
    LEFT + 12, y,
    "This report tracks the progression from the original point-centred PCA ellipsoid to the latest "
    "multi-frame reprojection-depth fits. It compares the newest source-scale-limited geometry with "
    "the established sphere, AR3, raw v4, and Moderate-Y alternatives. Metrics are final test-image "
    "means: PSNR and SSIM are higher-is-better; LPIPS is lower-is-better.",
    696, 9.5, 13,
)
p += paragraph
y -= 12
p, y = section(p, y, "Executive conclusion")
paragraph, y = wrapped(
    LEFT + 12, y,
    "The new radius-scale prior solved the catastrophic scale expansion in all three pathological "
    "intervals. It improved every metric relative to the original reprojection-depth v1 fit, by "
    "0.409-1.282 dB PSNR, 0.009-0.031 SSIM, and 0.023-0.036 LPIPS. This is a genuine geometric "
    "improvement, not merely a cosmetic PLY change.",
    696, 9.5, 13,
)
p += paragraph
y -= 9
paragraph, y = wrapped(
    LEFT + 12, y,
    "It is not yet the universal winner. Across all seven ego intervals, v4 Moderate-Y remains the "
    "best ellipsoid on the three macro metrics, while the sphere remains the safest simple baseline. "
    "The original reprojection-depth objective is very strong for some non-pathological scenes but is "
    "not safe without a scale prior or candidate-selection fallback.",
    696, 9.5, 13,
)
p += paragraph
y -= 15
p, y = section(p, y, "Latest six-run test: three formerly pathological intervals")
summary_rows = [
    ("Original reprojection v1, regular", "27.871", ".8509", ".3170"),
    ("Scale-limited reprojection, regular", "28.356", ".8637", ".2920"),
    ("Original reprojection v1, Visible-UV", "27.695", ".8455", ".3153"),
    ("Scale-limited reprojection, Visible-UV", "28.450", ".8656", ".2875"),
]
part, y = table(y, ["Three-interval macro average", "PSNR", "SSIM", "LPIPS"], summary_rows,
                [390, 105, 105, 105], 8.6, 25)
p += part
y -= 17
p, y = section(p, y, "Evidence boundaries")
for claim in [
    "Direct same-run-root comparisons are used for the six new scale-limited jobs against their original v1 counterparts.",
    "The seven-sequence macro comparison uses the same metric convention, but every cell is one training run rather than a statistical replication.",
    "Older plane results are historical context only because the training code and setup evolved; they are not direct September head-to-head evidence.",
]:
    part, y = wrapped(LEFT + 12, y, "- " + claim, 690, 9.1, 12)
    p += part
    y -= 3
pages.append(footer(p, 1, "Source roots: runs/2026-09-12_reprojection_depth_v1 and runs/2026-09-13_reprojection_depth_radius_scale."))

# 2. Chronology.
p = ""
p += text(LEFT, 568, "Ellipsoid methods: chronology and intent", 18, True, NAVY)
p += text(LEFT, 545, "Names below are used consistently in the remaining tables.", 9.5, color=GREY)
y = 520
method_rows = [
    ("Original PCA", "Point-centred PCA axes; all-point extrema", "Can be off-camera and very thin", "Generally weak in ego scenes"),
    ("Centered / AR3", "Mean-camera centre; PCA; 3:1 ratio cap", "Prevent pathological flattening", "Stable but needs clipping"),
    ("Sphere-clipped AR3", "AR3 radii capped by matching sphere", "Safety geometry / strong baseline", "Near sphere quality overall"),
    ("v3 depth proxy", "Front-camera ray depth to static candidates", "Surface near visible geometry", "Often vertically too thin"),
    ("v4 depth proxy", "Iterative multi-ray depth fit, road axes", "Improved robust depth target", "Strong, but can be flat"),
    ("v4 Moderate-Y", "v4 plus physical vertical-radius floor", "Avoid ill-conditioned flat surface", "Best macro ellipsoid"),
    ("Reprojection-depth v1", "Pixel reprojection plus weak log-depth", "Target moving-camera parallax", "Good scenes; 3 scale failures"),
    ("Scale-limited reprojection", "v1 plus soft source-scale prior", "Control excessive surface scale", "Fixes failures, not universal"),
]
part, y = table(
    y,
    ["Method", "Core construction", "Why it was introduced", "Observed result"],
    method_rows, [142, 235, 170, 173], 7.1, 38,
)
p += part
y -= 14
p, y = section(p, y, "Coordinate and optimisation context")
part, y = wrapped(
    LEFT + 12, y,
    "The local Waymo / NAG convention used for these PLYs is X forward, Y vertical, and Z lateral. "
    "A fitted ellipsoid is rendered with a 2 percent runtime radius margin. The v4 family uses fixed "
    "road-aligned axes; it does not infer arbitrary PCA axes during the final depth fit. The later "
    "reprojection fits retain that fixed orientation, but optimise centre and radii.",
    696, 9.2, 13,
)
p += part
y -= 9
part, y = wrapped(
    LEFT + 12, y,
    "A more sphere-like result is not automatically evidence that an ellipsoid failed. Sphere clipping, "
    "camera centring, and vertical thickening are deliberate safety mechanisms. The key question is "
    "whether the resulting surface reduces background-flow strain without putting cameras near the surface.",
    696, 9.2, 13,
)
p += part
pages.append(footer(p, 2, "The entries summarise actual tested variants; they are not mutually exclusive code paths."))

# 3. Full seven-sequence macro comparison.
p = ""
p += text(LEFT, 568, "Full seven-interval ego benchmark: macro averages", 18, True, NAVY)
p += text(LEFT, 545, "Equal weight for W191 0-40, 40-128, 128-198 and W254 0-37, 37-81, 86-137, 139-195.", 9, color=GREY)
y = 520
p, y = section(p, y, "Regular mapping")
regular_rows = [
    ("Sphere", "28.081", ".8436", ".3175", "Strong reference"),
    ("Sphere-clipped AR3", "28.030", ".8427", ".3166", "Near sphere"),
    ("Raw v4 depth proxy", "27.982", ".8463", ".3115", "Better perceptual average"),
    ("v4 Moderate-Y", "28.112", ".8489", ".3099", "Best macro ellipsoid"),
    ("Original reprojection-depth v1", "27.919", ".8434", ".3205", "Pathologies pull mean down"),
]
part, y = table(y, ["Background geometry", "PSNR", "SSIM", "LPIPS", "Reading"], regular_rows,
                [300, 90, 90, 90, 150], 8.3, 27)
p += part
y -= 24
p, y = section(p, y, "Visible-UV mapping")
uv_rows = [
    ("Sphere", "28.075", ".8433", ".3140", "Strong reference"),
    ("Sphere-clipped AR3", "28.052", ".8430", ".3141", "Near sphere"),
    ("Raw v4 depth proxy", "27.968", ".8451", ".3111", "Better perceptual average"),
    ("v4 Moderate-Y", "28.100", ".8484", ".3084", "Best macro ellipsoid"),
    ("Original reprojection-depth v1", "27.842", ".8408", ".3193", "Pathologies pull mean down"),
]
part, y = table(y, ["Background geometry", "PSNR", "SSIM", "LPIPS", "Reading"], uv_rows,
                [300, 90, 90, 90, 150], 8.3, 27)
p += part
y -= 18
p, y = section(p, y, "Reading the macro result")
for item in [
    "Moderate-Y v4 is the current all-sequence leader: it has the highest macro PSNR and SSIM and the lowest macro LPIPS in both mappings among the listed methods.",
    "Visible-UV changes image-space sample density and texture layout. It is a separate rendering choice, not a correction inside the geometry-only fit; it helps LPIPS more consistently than PSNR here.",
    "The scale-limited reprojection method has not been trained on all seven intervals, so it is intentionally excluded from this full-set table. Page 4 gives its direct six-run evidence.",
]:
    part, y = wrapped(LEFT + 12, y, "- " + item, 690, 9.1, 12)
    p += part
    y -= 3
pages.append(footer(p, 3, "All values are final test-image means. Differences below roughly 0.05 dB should be treated as ties until replicated."))

# 4. Latest direct test.
p = ""
p += text(LEFT, 568, "Latest optimisation: source-scale-limited reprojection fit", 18, True, NAVY)
p += text(LEFT, 545, "Direct comparison of the six new jobs with their original reprojection-depth v1 counterparts.", 9.5, color=GREY)
y = 520
p, y = section(p, y, "Per-interval outcomes")
scale_rows = [
    ("W191 0-40", "Regular", "28.062 / .8230 / .3454", "28.568 / .8374 / .3208", "+.505 / +.0144 / -.0246"),
    ("W191 0-40", "Visible-UV", "28.036 / .8202 / .3414", "28.500 / .8357 / .3189", "+.463 / +.0155 / -.0225"),
    ("W254 0-37", "Regular", "27.989 / .8697 / .3070", "28.529 / .8840 / .2794", "+.541 / +.0143 / -.0276"),
    ("W254 0-37", "Visible-UV", "28.168 / .8739 / .2993", "28.687 / .8872 / .2742", "+.519 / +.0133 / -.0251"),
    ("W254 139-195", "Regular", "27.562 / .8601 / .2985", "27.971 / .8696 / .2758", "+.409 / +.0095 / -.0227"),
    ("W254 139-195", "Visible-UV", "26.882 / .8425 / .3053", "28.164 / .8738 / .2695", "+1.282 / +.0313 / -.0358"),
]
part, y = table(
    y,
    ["Interval", "Mapping", "Original v1: P / S / L", "Scale-limited: P / S / L", "Scale minus original"],
    scale_rows, [113, 88, 170, 170, 179], 7.45, 30,
)
p += part
y -= 17
p, y = section(p, y, "What changed geometrically")
geometry_rows = [
    ("W191 0-40", "[41.66, 157.59, 17.15]", "[42.38, 66.36, 17.15]", "66.35"),
    ("W254 0-37", "[35.09, 171.45, 17.14]", "[35.31, 61.45, 17.14]", "61.41"),
    ("W254 139-195", "[41.33, 171.46, 17.15]", "[41.49, 73.67, 17.15]", "73.60"),
]
part, y = table(
    y,
    ["Interval", "Original radii [X, Y, Z] m", "Scale-limited radii m", "Reference maximum m"],
    geometry_rows, [150, 205, 205, 160], 8.0, 28,
)
p += part
y -= 15
part, y = wrapped(
    LEFT + 12, y,
    "All three retained a lateral-major orientation. That is important evidence: orientation itself was "
    "not the dominant failure. Constraining the enormous lateral radius to the safe raw-v4 source scale "
    "removed the harmful surface displacement and improved every metric.",
    696, 9.3, 13,
)
p += part
pages.append(footer(p, 4, "P / S / L = PSNR / SSIM / LPIPS. X forward, Y vertical, Z lateral in the PLY convention."))

# 5. Comparators on the three paths.
p = ""
p += text(LEFT, 568, "Pathological intervals: best-established comparison set", 18, True, NAVY)
p += text(LEFT, 545, "The new method fixes v1, but this benchmark tests whether it has surpassed practical alternatives.", 9.2, color=GREY)
y = 520
p, y = section(p, y, "Regular mapping")
regular_compare = [
    ("W191 0-40", "28.062 / .823 / .345", "28.568 / .837 / .321", "28.269 / .830 / .323", "28.712 / .847 / .306", "28.670 / .850 / .296"),
    ("W254 0-37", "27.989 / .870 / .307", "28.529 / .884 / .279", "29.106 / .889 / .274", "28.951 / .890 / .270", "28.864 / .889 / .271"),
    ("W254 139-195", "27.562 / .860 / .298", "27.971 / .870 / .276", "28.586 / .877 / .264", "28.962 / .884 / .260", "28.313 / .872 / .280"),
]
part, y = table(
    y,
    ["Interval", "Original v1", "Scale-limited", "Sphere", "Raw v4", "Moderate-Y"],
    regular_compare, [104, 123, 123, 123, 123, 124], 7.1, 30,
)
p += part
y -= 22
p, y = section(p, y, "Visible-UV mapping")
uv_compare = [
    ("W191 0-40", "28.036 / .820 / .341", "28.500 / .836 / .319", "28.197 / .828 / .321", "28.691 / .847 / .305", "28.622 / .848 / .294"),
    ("W254 0-37", "28.168 / .874 / .299", "28.687 / .887 / .274", "29.124 / .890 / .268", "28.997 / .890 / .267", "28.958 / .890 / .268"),
    ("W254 139-195", "26.882 / .842 / .305", "28.164 / .874 / .269", "28.404 / .873 / .265", "28.962 / .884 / .259", "28.325 / .875 / .269"),
]
part, y = table(
    y,
    ["Interval", "Original v1", "Scale-limited", "Sphere", "Raw v4", "Moderate-Y"],
    uv_compare, [104, 123, 123, 123, 123, 124], 7.1, 30,
)
p += part
y -= 20
p, y = section(p, y, "Comparison conclusion")
for item in [
    "The scale-limited fit is a decisive improvement over original v1, and beats the matching current sphere on every metric for W191 0-40.",
    "It has not yet consistently beaten raw v4 or Moderate-Y. For W254 0-37 its Visible-UV LPIPS is nearly tied with the best AR3 / v4-family alternatives, but sphere still leads PSNR.",
    "For W254 139-195 the scale-limited Visible-UV result approaches Moderate-Y, but remains behind raw v4 and the historical plane reference. This supports a better candidate-selection objective, not another arbitrary global radius rule.",
]:
    part, y = wrapped(LEFT + 12, y, "- " + item, 690, 8.9, 12)
    p += part
    y -= 3
pages.append(footer(p, 5, "Rows show PSNR / SSIM / LPIPS. Raw-v4 and Moderate-Y values are prior established runs, not retrained in the latest job folder."))

# 6. Math and next steps.
p = ""
p += text(LEFT, 568, "Latest regulariser and recommended next work", 18, True, NAVY)
p += text(LEFT, 545, "Keep fitting, diagnostics, and training validation separate so selection does not overfit a named sequence.", 9.2, color=GREY)
y = 520
p, y = section(p, y, "Original reprojection-depth objective")
part, y = wrapped(
    LEFT + 12, y,
    "For a LiDAR point P viewed from source camera i, the fitter intersects the source ray with the ellipsoid at Q_i, "
    "projects P and Q_i into useful target cameras j, and robustly penalises their image displacement. A weaker log-depth "
    "term anchors source-ray depth. In shorthand: L_original = L_reprojection + lambda_d L_raydepth + L_safety. It is "
    "iterative and uses static / visible valid observations, baseline weighting, Huber robustness, and camera-safety constraints.",
    696, 9.1, 13,
)
p += part
y -= 12
p, y = section(p, y, "New source-scale penalty")
part, y = wrapped(
    LEFT + 12, y,
    "Let r_k be a candidate radius and r_ref be the maximum radius of the safety-projected raw-v4 initializer. The added "
    "term is L_scale = 20 * (1/3) sum_k H_0.20(max(0, log(r_k / r_ref))). Radii at or below r_ref are unpenalised; a "
    "radius that grows beyond the trusted source scale pays a smooth robust cost. The total is L_new = L_original + L_scale.",
    696, 9.1, 13,
)
p += part
y -= 13
p, y = section(p, y, "Why it worked, and what it does not solve")
part, y = wrapped(
    LEFT + 12, y,
    "The original pixel/depth objective was nearly indifferent to large lateral expansions in the three difficult road geometries. "
    "The scale term restores a prior that the surface should not become larger than the safe raw-v4 starting scale, while still "
    "allowing the data term to choose centre and smaller dimensions. It does not prove global optimality, and it does not compensate "
    "for Visible-UV texture densification; Visible-UV is a separate layout-dependent rendering choice.",
    696, 9.1, 13,
)
p += part
y -= 13
p, y = section(p, y, "Recommended next experiment")
for item in [
    "Build a generic candidate portfolio per interval: raw v4, Moderate-Y, sphere-equivalent, sphere-clipped AR3, original reprojection v1, and scale-limited reprojection. Safety-project each candidate before scoring.",
    "Measure before training: pixel reprojection error, target-camera angular error, normalised ellipsoid angular correspondence, and mapping-specific texture-texel displacement. First test which diagnostics correlate with PSNR, SSIM, LPIPS, and background spatial error.",
    "Use original v1 geometry as one candidate / warm start so a scale-limited search cannot needlessly move a previously good basin such as W254 86-137. Select by the validated pre-training score, then train only the selected candidate in regular and Visible-UV.",
]:
    part, y = wrapped(LEFT + 12, y, "- " + item, 690, 8.85, 12)
    p += part
    y -= 3
pages.append(footer(p, 6, "The recommendation is a generic portfolio with validated diagnostics; it does not claim a single ellipsoid is inherently insufficient."))


def build_pdf(streams):
    """Assemble a minimal, dependency-free, vector-text PDF."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    page_ids = []
    for stream in streams:
        encoded = stream.encode("latin-1")
        content_id = len(objects) + 1
        objects.append("<< /Length {} >>\nstream\n{}endstream".format(len(encoded), stream))
        page_id = len(objects) + 1
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources "
            "<< /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {} 0 R >>".format(W, H, content_id)
        )
        page_ids.append(page_id)
    objects[1] = "<< /Type /Pages /Kids [{}] /Count {} >>".format(
        " ".join("{} 0 R".format(number) for number in page_ids), len(page_ids)
    )
    body = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(body.encode("latin-1")))
        body += "{} 0 obj\n{}\nendobj\n".format(number, obj)
    xref = len(body.encode("latin-1"))
    # The xref fields do not require the optional trailing separator space.
    # Omitting it also keeps a generated binary PDF clean under git diff --check.
    body += "xref\n0 {}\n0000000000 65535 f\n".format(len(objects) + 1)
    body += "".join("{:010d} 00000 n\n".format(offset) for offset in offsets[1:])
    body += "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(len(objects) + 1, xref)
    return body.encode("latin-1")


OUT.write_bytes(build_pdf(pages))
print(OUT)
