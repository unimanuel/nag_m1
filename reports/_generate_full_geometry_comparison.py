"""Create the standalone PDF report requested for the Waymo background study."""

from pathlib import Path


# The original PDF may remain open in a Windows viewer, which locks that exact
# path through the mounted drive.  Write the corrected revision alongside it.
OUT = Path(__file__).with_name("2026-08-21_full_geometry_comparison_high_contrast.pdf")
W, H = 792, 612  # US Letter, landscape, in PostScript points.


def esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text(x, y, value, size=9, bold=False, color=None):
    font = "F2" if bold else "F1"
    # PDF's non-stroking colour persists across drawing operations.  Always
    # set an explicit foreground colour so table text cannot inherit a row's
    # light-gray background fill.
    color = color or (0.04, 0.04, 0.04)
    color_command = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg\n"
    return f"BT\n{color_command}/{font} {size} Tf\n{x:.1f} {y:.1f} Td\n({esc(value)}) Tj\nET\n"


def line(x1, y1, x2, y2, width=0.5, gray=0.72):
    return f"{gray:.3f} G\n{width:.2f} w\n{x1:.1f} {y1:.1f} m\n{x2:.1f} {y2:.1f} l\nS\n"


def fill_rect(x, y, width, height, gray):
    return f"{gray:.3f} g\n{x:.1f} {y:.1f} {width:.1f} {height:.1f} re\nf\n"


def wrapped(x, y, value, width, size=9, leading=12, bold=False):
    words = value.split()
    current, rows = "", []
    # Helvetica averages about 0.50em per character at the small report sizes.
    capacity = max(12, int(width / (size * 0.50)))
    for word in words:
        candidate = (current + " " + word).strip()
        if current and len(candidate) > capacity:
            rows.append(current)
            current = word
        else:
            current = candidate
    if current:
        rows.append(current)
    commands = ""
    for row in rows:
        commands += text(x, y, row, size=size, bold=bold)
        y -= leading
    return commands, y


def table(y, columns, rows, widths, font_size=7.4, row_height=19):
    x = 36
    commands = fill_rect(x, y - row_height + 2, sum(widths), row_height, 0.84)
    cursor = x + 4
    for column, width in zip(columns, widths):
        commands += text(cursor, y - 12, column, font_size, bold=True)
        cursor += width
    y -= row_height
    for number, row in enumerate(rows):
        if number % 2 == 1:
            commands += fill_rect(x, y - row_height + 2, sum(widths), row_height, 0.96)
        cursor = x + 4
        for item, width in zip(row, widths):
            commands += text(cursor, y - 12, item, font_size)
            cursor += width
        commands += line(x, y - row_height + 2, x + sum(widths), y - row_height + 2, 0.3, 0.84)
        y -= row_height
    return commands, y


pages = []

# Page 1: purpose and top-level conclusion.
p = ""
p += text(36, 568, "Waymo background geometry comparison", 20, bold=True, color=(0.08, 0.20, 0.38))
p += text(36, 544, "Controlled geometry ablation and complete ego-sequence benchmark", 10, color=(0.25, 0.25, 0.25))
p += line(36, 535, 756, 535, 1.2, 0.25)
y = 510
paragraphs = [
    ("Scope", "This report includes the original thin PCA ellipsoid, the later centered and clipped variants, the W254 86-137 cross-section ablations, spheres, and the plane baseline. Metrics are test-image averages: higher PSNR and SSIM are better; lower LPIPS is better."),
    ("Bottom line", "The camera-centered robust AR3 ellipsoid, with each radius capped by its matching sphere radius, is the best ellipsoid tested. It is effectively tied with the sphere overall and gives the best LPIPS on the controlled W254 86-137 interval. Sphere Visible-UV remains the safest all-sequence default."),
]
for heading, paragraph in paragraphs:
    p += text(36, y, heading, 11, bold=True, color=(0.08, 0.20, 0.38))
    y -= 16
    piece, y = wrapped(48, y, paragraph, 690, size=9.3, leading=13)
    p += piece
    y -= 12

p += text(36, y, "All seven large-ego sequences: equal-sequence macro average", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 10
summary_rows = [
    ("Sphere", "28.031", ".8424", ".3189", "Stable baseline"),
    ("Sphere Visible-UV", "28.075", ".8433", ".3140", "Best aggregate PSNR / SSIM"),
    ("AR3 clipped ellipsoid", "28.030", ".8427", ".3166", "Better LPIPS than regular sphere"),
    ("AR3 clipped ellipsoid Visible-UV", "28.052", ".8430", ".3141", "0.023 dB behind Sphere Visible-UV"),
]
piece, y = table(y, ["Mode", "PSNR", "SSIM", "LPIPS", "Reading"], summary_rows,
                 [235, 70, 70, 70, 275], font_size=8.3, row_height=24)
p += piece
y -= 23
p += text(36, y, "Interpretation", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 16
bullets = [
    "The difference between the two Visible-UV leaders is too small to call a practical winner from one training run.",
    "The final AR3 geometry removes the large losses caused by the original 5.96 m thin axis and the uncapped 89.12 m long axis.",
    "In W191 frames 128-198, sphere clipping degenerates the ellipsoid to a near-sphere with a camera nearly on its surface; that interval needs a camera-clearance fallback.",
]
for bullet in bullets:
    piece, y = wrapped(48, y, "- " + bullet, 680, size=9.2, leading=13)
    p += piece
    y -= 4
p += text(36, 28, "Generated 2026-08-21.  See pages 2-3 for all tested geometry variants and experimental interpretation.", 8, color=(0.35, 0.35, 0.35))
pages.append(p)

# Page 2: controlled table.
p = ""
p += text(36, 568, "Controlled W254 86-137 geometry ablation", 18, bold=True, color=(0.08, 0.20, 0.38))
p += text(36, 545, "Effective radii include the configured runtime margin and any sphere clipping. UV means Visible-UV.", 9.5, color=(0.25, 0.25, 0.25))
y = 522
geometry_rows = [
    ("Plane", "-", "24.329 / .777 / .386", "-"),
    ("Original point-centered PCA ellipsoid", "70.81, 37.26, 5.96", "26.020 / .820 / .345", "26.195 / .824 / .345"),
    ("Centered robust AR3, not sphere-clipped", "89.12, 29.71, 29.71", "26.275 / .822 / .343", "26.500 / .826 / .342"),
    ("Centered all-point enclosing, sphere-clipped", "58.62, 49.01, 7.01", "26.780 / .835 / .336", "26.734 / .832 / .334"),
    ("Narrow, thin cross-section", "58.62, 29.71, 7.01", "27.012 / .838 / .331", "27.097 / .839 / .328"),
    ("Halfway geometry", "73.87, 44.16, 44.16", "26.641 / .825 / .337", "26.730 / .826 / .335"),
    ("Sphere-equivalent ellipsoid", "58.62, 58.62, 58.62", "27.081 / .835 / .336", "27.221 / .837 / .332"),
    ("True sphere background", "58.62, 58.62, 58.62", "27.207 / .836 / .335", "27.399 / .840 / .329"),
    ("Centered AR3, sphere-clipped (selected)", "58.62, 29.71, 29.71", "27.320 / .839 / .324", "27.383 / .840 / .323"),
    ("Wide, thick cross-section", "58.62, 49.01, 29.71", "27.322 / .838 / .326", "27.274 / .837 / .326"),
]
piece, y = table(y, ["Geometry", "Effective radii (m)", "Regular: P / S / L", "UV: P / S / L"], geometry_rows,
                 [280, 155, 140, 140], font_size=7.2, row_height=29)
p += piece
y -= 20
p += text(36, y, "What the ablation establishes", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 16
findings = [
    "Centering alone was insufficient: the uncapped AR3 long axis was 89.12 m and remained substantially worse than the sphere.",
    "Sphere clipping was necessary. It brought the AR3 result from 26.275 to 27.320 regular PSNR.",
    "Thickness was necessary. Raising the third radius from 7.01 m to 29.71 m improved PSNR by 0.31-0.54 dB.",
    "Making the second radius wider did not help: Wide-Thick only ties in regular mode and loses in Visible-UV.",
]
for finding in findings:
    piece, y = wrapped(48, y, "- " + finding, 680, size=9, leading=12)
    p += piece
    y -= 3
p += text(36, 28, "P / S / L = PSNR / SSIM / LPIPS.  Higher P and S are better; lower L is better.", 8, color=(0.35, 0.35, 0.35))
pages.append(p)

# Page 3: broader history and recommendation.
p = ""
p += text(36, 568, "Broader experiment history and recommendation", 18, bold=True, color=(0.08, 0.20, 0.38))
p += text(36, 545, "W254 aggregate covers four ego intervals; the static W125 control uses frames 0-40.", 9.5, color=(0.25, 0.25, 0.25))
y = 520
w254_rows = [
    ("Original non-centered ellipsoid", "27.1315", "27.0566", "Clearly behind sphere"),
    ("Plane", "26.2217", "-", "Worst broad baseline"),
    ("Sphere", "28.0810", "28.0892", "Strong stable baseline"),
    ("Final AR3 clipped ellipsoid", "28.0441", "28.0721", "Tied with sphere in practice"),
]
p += text(36, y, "W254 four-interval macro PSNR", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 10
piece, y = table(y, ["Background family", "Regular", "Visible-UV", "Conclusion"], w254_rows,
                 [280, 105, 105, 225], font_size=8.3, row_height=25)
p += piece
y -= 22
p += text(36, y, "Static-camera W125 control", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 16
control_text = "On W125 frames 0-40, ellipsoid Visible-UV produced 33.247 PSNR, sphere regular 33.234, sphere Visible-UV 33.092, and plane 32.563. This verifies that ellipsoids are not intrinsically harmful; geometry selection is the deciding factor in ego sequences."
piece, y = wrapped(48, y, control_text, 680, size=9.2, leading=13)
p += piece
y -= 15
p += text(36, y, "Recommended interpretation", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 16
recommendations = [
    "Production-safe default: Sphere Visible-UV. It has the highest aggregate PSNR and SSIM across all seven large-ego sequences.",
    "Best ellipsoid candidate: camera-centered robust AR3, 2% margin, then radius-wise sphere clipping. Its aggregate quality is virtually tied with Sphere Visible-UV and it is best on controlled perceptual quality (LPIPS).",
    "Do not reuse the original point-centered thin PCA geometry, the uncapped centered AR3 geometry, or a thin third axis. The controlled evidence shows each is materially worse.",
    "Before using AR3 automatically for every sequence, add a camera-clearance fallback for cases such as W191 128-198, where sphere clipping places a camera on or nearly on the surface.",
]
for item in recommendations:
    piece, y = wrapped(48, y, "- " + item, 680, size=9.2, leading=13)
    p += piece
    y -= 5
p += text(36, 28, "All comparisons are single training runs from the listed experiment folders; sub-0.05 dB differences should be treated as ties until replicated.", 8, color=(0.35, 0.35, 0.35))
pages.append(p)


def build_pdf(streams):
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
    page_ids = []
    for stream in streams:
        content_id = len(objects) + 1
        encoded = stream.encode("latin-1")
        objects.append(f"<< /Length {len(encoded)} >>\nstream\n{stream}endstream")
        page_id = len(objects) + 1
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>")
        page_ids.append(page_id)
    objects[1] = "<< /Type /Pages /Kids [" + " ".join(f"{n} 0 R" for n in page_ids) + f"] /Count {len(page_ids)} >>"
    body = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(body.encode("latin-1")))
        body += f"{number} 0 obj\n{obj}\nendobj\n"
    xref = len(body.encode("latin-1"))
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    body += "".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return body.encode("latin-1")


OUT.write_bytes(build_pdf(pages))
print(OUT)
