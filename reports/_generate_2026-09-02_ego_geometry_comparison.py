"""Create the consolidated Waymo ego-background comparison PDF.

All values are final means read from the completed NAG runs.  The report
intentionally preserves the individual-sequence rows: the macro average alone
can hide a mode that wins on a small number of intervals but loses elsewhere.
"""

from pathlib import Path


OUT = Path(__file__).with_name("2026-09-02_ego_geometry_comparison.pdf")
W, H = 792, 612  # Letter landscape.
NAVY = (0.07, 0.20, 0.38)
BLACK = (0.04, 0.04, 0.04)


def esc(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text(x, y, value, size=9, bold=False, color=BLACK):
    font = "F2" if bold else "F1"
    r, g, b = color
    return f"BT\n{r:.3f} {g:.3f} {b:.3f} rg\n/{font} {size} Tf\n{x:.1f} {y:.1f} Td\n({esc(value)}) Tj\nET\n"


def fill(x, y, width, height, gray):
    return f"{gray:.3f} g\n{x:.1f} {y:.1f} {width:.1f} {height:.1f} re\nf\n"


def line(x1, y1, x2, y2, width=.5, gray=.72):
    return f"{gray:.3f} G\n{width:.2f} w\n{x1:.1f} {y1:.1f} m\n{x2:.1f} {y2:.1f} l\nS\n"


def wrap(x, y, value, width, size=9, leading=12):
    capacity = max(16, int(width / (size * .50)))
    words, row, result = value.split(), "", ""
    for word in words:
        candidate = (row + " " + word).strip()
        if row and len(candidate) > capacity:
            result += text(x, y, row, size)
            y -= leading
            row = word
        else:
            row = candidate
    if row:
        result += text(x, y, row, size)
        y -= leading
    return result, y


def table(y, columns, rows, widths, size=7.4, height=23):
    x = 36
    command = fill(x, y - height + 2, sum(widths), height, .84)
    cursor = x + 4
    for column, width in zip(columns, widths):
        command += text(cursor, y - 13, column, size, bold=True)
        cursor += width
    y -= height
    for number, row in enumerate(rows):
        if number % 2:
            command += fill(x, y - height + 2, sum(widths), height, .96)
        cursor = x + 4
        for item, width in zip(row, widths):
            command += text(cursor, y - 14, item, size)
            cursor += width
        command += line(x, y - height + 2, x + sum(widths), y - height + 2, .3, .84)
        y -= height
    return command, y


def title(page, heading, subtitle):
    return (page + text(36, 568, heading, 18, bold=True, color=NAVY)
            + text(36, 545, subtitle, 9.3, color=(.25, .25, .25))
            + line(36, 535, 756, 535, 1.1, .25))


# In every triplet: PSNR (higher), SSIM (higher), LPIPS (lower).
# Sphere UV is used as the common sphere reference, because the historic run
# set has no regular-sphere result for the three W191 intervals.
DATA = {
    "W191 0-40": {
        "Sphere UV": (28.197100, .827780, .320605), "AR3": (28.230772, .833962, .315698),
        "AR3 UV": (28.261862, .833747, .314102), "Moderate-Y": (28.621964, .849472, .297921),
        "Moderate-Y UV": (28.670374, .850159, .296243), "Ray-depth": (28.712271, .846649, .306212),
    },
    "W191 40-128": {
        "Sphere UV": (29.910645, .876719, .272625), "AR3": (29.806711, .874960, .278843),
        "AR3 UV": (29.858500, .875770, .275488), "Moderate-Y": (29.804617, .877552, .271623),
        "Moderate-Y UV": (29.854305, .877760, .270039), "Ray-depth": (29.591150, .874105, .276616),
    },
    "W191 128-198": {
        "Sphere UV": (26.063364, .762385, .423533), "AR3": (25.995184, .759124, .425481),
        "AR3 UV": (25.955662, .758968, .423434), "Moderate-Y": (26.013540, .764649, .416078),
        "Moderate-Y UV": (26.063341, .766838, .414340), "Ray-depth": (25.829252, .757986, .422015),
    },
    "W254 0-37": {
        "Sphere UV": (29.123531, .889625, .267810), "AR3": (28.991995, .887048, .278191),
        "AR3 UV": (29.034340, .887555, .274220), "Moderate-Y": (28.883076, .886878, .279862),
        "Moderate-Y UV": (28.826880, .885680, .276968), "Ray-depth": (28.967548, .890197, .274521),
    },
    "W254 37-81": {
        "Sphere UV": (27.430573, .833122, .319411), "AR3": (27.339977, .828885, .325543),
        "AR3 UV": (27.293554, .828048, .323951), "Moderate-Y": (27.859316, .844358, .313570),
        "Moderate-Y UV": (27.726147, .840850, .312894), "Ray-depth": (27.876034, .850506, .294362),
    },
    "W254 86-137": {
        "Sphere UV": (27.398512, .840473, .328928), "AR3": (27.320404, .838828, .324450),
        "AR3 UV": (27.382591, .839847, .322950), "Moderate-Y": (27.261190, .844381, .320807),
        "Moderate-Y UV": (27.235680, .843135, .319977), "Ray-depth": (26.950788, .837661, .331641),
    },
    "W254 139-195": {
        "Sphere UV": (28.404366, .873300, .265242), "AR3": (28.524195, .875789, .267684),
        "AR3 UV": (28.577812, .876894, .264214), "Moderate-Y": (28.342157, .874986, .269385),
        "Moderate-Y UV": (28.324766, .874621, .268519), "Ray-depth": (27.946692, .867258, .275008),
    },
}


def triplet(values):
    return f"{values[0]:.3f} / {values[1]:.3f} / {values[2]:.3f}"


def macro(mode, names=DATA):
    values = [DATA[name][mode] for name in names]
    return tuple(sum(x[i] for x in values) / len(values) for i in range(3))


pages = []
p = title("", "Waymo ego-background comparison through 2026-09-02",
          "Final test-image means from completed runs. Higher PSNR/SSIM and lower LPIPS are better.")
y = 510
for heading, paragraph in [
    ("Question answered", "This revision brings the current ray-depth v4 experiments into the earlier sphere and camera-centred AR3 comparison. It explicitly compares the moderate-Y ellipsoid with the common Sphere Visible-UV reference and both AR3 modes for all three metrics on every one of the seven ego intervals."),
    ("Definitions", "AR3 is the camera-centred, robust 3:1 aspect-ratio ellipsoid with per-axis sphere clipping. Moderate-Y starts from the v4 ray-depth proxy fit and raises its physical vertical radius to equal its lateral radius. UV denotes the visible-UV training configuration."),
    ("Primary result", "Moderate-Y is not a universal winner in PSNR: relative to Sphere UV, it wins on only two of seven sequences (with one practical tie in UV mode). Its small macro PSNR gain is driven by those two wins. It nevertheless improves macro SSIM and LPIPS, so its evidence is mixed rather than a clean replacement for the sphere."),
]:
    p += text(36, y, heading, 11, bold=True, color=NAVY); y -= 16
    piece, y = wrap(48, y, paragraph, 680, size=9.3, leading=13); p += piece; y -= 10
p += text(36, y, "Seven-sequence equal-weight macro average", 12, bold=True, color=NAVY); y -= 10
rows = []
for mode, reading in [
    ("Sphere UV", "Common all-sequence reference"), ("AR3", "Regular clipped AR3"),
    ("AR3 UV", "Visible-UV clipped AR3"), ("Ray-depth", "Raw v4 proxy; vertically thin"),
    ("Moderate-Y", "v4 proxy, vertical radius = lateral"), ("Moderate-Y UV", "Visible-UV moderate-Y"),
]:
    a, b, c = macro(mode)
    label = "Sphere Visible-UV" if mode == "Sphere UV" else mode
    rows.append((label, f"{a:.3f}", f"{b:.4f}", f"{c:.4f}", reading))
piece, y = table(y, ["Mode", "PSNR", "SSIM", "LPIPS", "Interpretation"], rows,
                 [190, 85, 85, 85, 275], size=8.2, height=25); p += piece
p += text(36, 28, "Macro averages give each interval equal weight; they are not frame-weighted. Single-run differences below about 0.05 dB are best treated as ties.", 8, color=(.35, .35, .35))
pages.append(p)

for cohort, names in [("W191", list(DATA)[:3]), ("W254", list(DATA)[3:])]:
    p = title("", f"Individual {cohort} sequence comparison",
              "Each cell is PSNR / SSIM / LPIPS. The highlighted purpose is direct Moderate-Y versus Sphere and AR3 comparison.")
    y = 510
    rows = []
    for name in names:
        d = DATA[name]
        rows.append((name, triplet(d["Sphere UV"]), triplet(d["AR3"]), triplet(d["AR3 UV"]),
                     triplet(d["Moderate-Y"]), triplet(d["Moderate-Y UV"])))
    piece, y = table(y, ["Sequence", "Sphere UV", "AR3", "AR3 UV", "Moderate-Y", "Moderate-Y UV"], rows,
                     [105, 122, 122, 122, 122, 122], size=7.2, height=30); p += piece
    y -= 20
    p += text(36, y, "PSNR reading", 11, bold=True, color=NAVY); y -= 16
    observations = {
        "W191": [
            "0-40: both Moderate-Y modes beat Sphere UV and AR3 decisively; raw ray-depth is the highest PSNR but has worse LPIPS than Moderate-Y UV.",
            "40-128: Sphere UV is highest PSNR; Moderate-Y UV is only 0.056 dB behind and has the best SSIM and LPIPS.",
            "128-198: Moderate-Y UV is a PSNR tie with Sphere UV (-0.00002 dB) and has materially better SSIM and LPIPS.",
        ],
        "W254": [
            "0-37: Sphere UV remains best on all three metrics; Moderate-Y is clearly behind.",
            "37-81: raw ray-depth is highest PSNR and best SSIM/LPIPS; both Moderate-Y modes also clearly beat Sphere and AR3.",
            "86-137: Sphere UV is highest PSNR, while Moderate-Y UV has the best LPIPS; AR3 is the best regular ellipsoid PSNR.",
            "139-195: AR3 UV is highest PSNR and LPIPS; Moderate-Y is below Sphere UV.",
        ],
    }
    for observation in observations[cohort]:
        piece, y = wrap(48, y, "- " + observation, 680, size=9, leading=12); p += piece; y -= 4
    p += text(36, 28, "Sphere regular is absent for the W191 historic runs, so Sphere UV is the fair common reference here. No metric has been inferred from a missing run.", 8, color=(.35, .35, .35))
    pages.append(p)

p = title("", "Moderate-Y: direct comparison and interpretation",
          "Moderate-Y regular is compared with Sphere UV; this is the only sphere baseline available across all seven intervals.")
y = 510
delta_rows = []
for name, d in DATA.items():
    m, s = d["Moderate-Y"], d["Sphere UV"]
    delta_rows.append((name, f"{m[0]-s[0]:+.3f}", f"{m[1]-s[1]:+.4f}", f"{m[2]-s[2]:+.4f}",
                       "better" if m[0] > s[0] else "worse"))
piece, y = table(y, ["Sequence", "d PSNR", "d SSIM", "d LPIPS", "PSNR result"], delta_rows,
                 [180, 120, 120, 120, 180], size=8.2, height=25); p += piece
y -= 18
p += text(36, y, "What the deltas mean", 12, bold=True, color=NAVY); y -= 16
for statement in [
    "Moderate-Y regular is higher in PSNR on W191 0-40 and W254 37-81, and lower on the other five intervals. Those two gains are large enough to yield a +0.037 dB seven-sequence macro advantage over Sphere UV.",
    "It improves SSIM on six of seven intervals and LPIPS on five of seven. Thus the average gain is not merely a PSNR accident, but the instability across intervals prevents a production-default recommendation.",
    "Visible-UV is not consistently additive: it helps Moderate-Y in W191 0-40 and W191 128-198 but decreases its PSNR on five intervals. It is worthwhile as a separate experiment mode, not as an assumed upgrade.",
]:
    piece, y = wrap(48, y, "- " + statement, 680, size=9.2, leading=13); p += piece; y -= 5
p += text(36, 28, "For LPIPS, a negative delta is an improvement. Delta columns are Moderate-Y regular minus Sphere UV.", 8, color=(.35, .35, .35))
pages.append(p)

p = title("", "Earlier geometry experiments and what they established",
          "Controlled W254 86-137 ablations isolate the effect of centering, clipping, and cross-section thickness.")
y = 510
rows = [
    ("Plane", "-", "24.329 / .777 / .386", "Worst broad baseline"),
    ("Original point-centred PCA", "70.81, 37.26, 5.96", "26.020 / .820 / .345", "Thin height and off-centre geometry fail"),
    ("Centred robust AR3, uncapped", "89.12, 29.71, 29.71", "26.275 / .822 / .343", "Centering alone is insufficient"),
    ("Centred all-point, sphere-clipped", "58.62, 49.01, 7.01", "26.780 / .835 / .336", "Clipping helps; thin axis still costly"),
    ("Narrow thin", "58.62, 29.71, 7.01", "27.012 / .838 / .331", "Still too thin"),
    ("True sphere", "58.62, 58.62, 58.62", "27.207 / .836 / .335", "Strong stable reference"),
    ("AR3 centred, sphere-clipped", "58.62, 29.71, 29.71", "27.320 / .839 / .324", "Selected early ellipsoid"),
    ("Wide thick", "58.62, 49.01, 29.71", "27.322 / .838 / .326", "No benefit from wider lateral radius"),
]
piece, y = table(y, ["W254 86-137 regular mode", "Effective radii (m)", "PSNR / SSIM / LPIPS", "Conclusion"], rows,
                 [220, 180, 165, 155], size=7.3, height=28); p += piece
y -= 16
p += text(36, y, "Ray-depth v4 changes the question", 12, bold=True, color=NAVY); y -= 16
for statement in [
    "The v4 ray-depth proxy places the surface using FRONT camera rays and background LiDAR depth, rather than PCA point extents. Its raw fit is vertically flat; Moderate-Y is the controlled attempt to remove that thin vertical dimension while retaining the fitted forward/lateral placement.",
    "Proxy ray-depth error does not predict NAG quality reliably: W254 0-37 and 139-195 have low depth error yet Sphere UV wins, while W254 37-81 has a higher proxy error but ray-depth and Moderate-Y perform best. The current fit objective is therefore not a complete proxy for multi-view rendering difficulty.",
    "The next principled fitting direction is a visibility-aware, multi-frame reprojection or parallax-consistency objective, tested first on intervals where the present modes disagree. It should be treated as a new experiment, not a conclusion from these single-run comparisons.",
]:
    piece, y = wrap(48, y, "- " + statement, 680, size=9.1, leading=13); p += piece; y -= 5
p += text(36, 28, "All listed results are final training means from the named run folders. This report records evidence, not a statistically replicated ranking.", 8, color=(.35, .35, .35))
pages.append(p)


def build_pdf(streams):
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
    page_ids = []
    for stream in streams:
        content_id = len(objects) + 1
        encoded = stream.encode("latin-1")
        objects.append(f"<< /Length {len(encoded)} >>\nstream\n{stream}endstream")
        page_id = len(objects) + 1
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>")
        page_ids.append(page_id)
    objects[1] = "<< /Type /Pages /Kids [" + " ".join(f"{i} 0 R" for i in page_ids) + f"] /Count {len(page_ids)} >>"
    result, offsets = "%PDF-1.4\n%\\xe2\\xe3\\xcf\\xd3\n", [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(result.encode("latin-1")))
        result += f"{number} 0 obj\n{obj}\nendobj\n"
    start = len(result.encode("latin-1"))
    result += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f\n"
    result += "".join(f"{offset:010d} 00000 n\n" for offset in offsets[1:])
    result += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n"
    return result.encode("latin-1")


OUT.write_bytes(build_pdf(pages))
print(OUT)
