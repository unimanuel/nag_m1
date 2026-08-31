"""Generate a math reference for the camera-centred AR3 clipped ellipsoid."""

from pathlib import Path


OUT = Path(__file__).with_name("2026-08-21_ar3_centered_clipped_ellipsoid_math.pdf")
W, H = 612, 792  # Letter portrait


def esc(value):
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text(x, y, value, size=9, bold=False, mono=False, color=None):
    font = "F3" if mono else ("F2" if bold else "F1")
    color = color or (0.04, 0.04, 0.04)
    return (
        "BT\n"
        f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg\n"
        f"/{font} {size} Tf\n{x:.1f} {y:.1f} Td\n({esc(value)}) Tj\nET\n"
    )


def fill(x, y, width, height, gray):
    return f"{gray:.3f} g\n{x:.1f} {y:.1f} {width:.1f} {height:.1f} re\nf\n"


def line(x1, y1, x2, y2, width=0.6, gray=0.72):
    return f"{gray:.3f} G\n{width:.2f} w\n{x1:.1f} {y1:.1f} m\n{x2:.1f} {y2:.1f} l\nS\n"


def wrap(x, y, content, width=520, size=9.4, leading=13, indent=""):
    capacity = max(16, int(width / (size * 0.50)))
    words, row, result = content.split(), "", ""
    for word in words:
        candidate = (row + " " + word).strip()
        if row and len(candidate) > capacity:
            result += text(x, y, indent + row, size)
            y -= leading
            row = word
        else:
            row = candidate
    if row:
        result += text(x, y, indent + row, size)
        y -= leading
    return result, y


def equation(x, y, content):
    return fill(x - 7, y - 8, 536, 22, 0.95) + text(x, y, content, size=9.3, mono=True, color=(0.02, 0.10, 0.25))


pages = []

p = ""
p += text(42, 744, "Camera-centred AR3 sphere-clipped ellipsoid", 18, bold=True, color=(0.08, 0.20, 0.38))
p += text(42, 722, "Mathematical construction used by the revised ego-background experiment", 9.5, color=(0.28, 0.28, 0.28))
p += line(42, 712, 570, 712, 1.1, 0.25)
y = 688
p += text(42, y, "1. Inputs and coordinates", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 18
paragraph, y = wrap(52, y, "For the YAML-selected frames, x_i are the registered FRONT-projected LiDAR points and c_t are the corresponding NAG-world camera centres. The ellipsoid uses the LiDAR points only for orientation and robust size. Its centre is deliberately the mean camera position, not the point-cloud mean.")
p += paragraph
y -= 7
p += equation(52, y, "c = (1/T) sum_t c_t")
y -= 33
p += text(42, y, "2. PCA orientation", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 18
paragraph, y = wrap(52, y, "PCA is computed from the point cloud around its own point mean x_bar. If Sigma is the covariance, its ordered orthonormal eigenvectors form the columns of R. R maps local principal-axis coordinates to NAG-world coordinates.")
p += paragraph
y -= 7
p += equation(52, y, "Sigma = (1/N) sum_i (x_i - x_bar)(x_i - x_bar)^T ;  Sigma R = R Lambda")
y -= 31
paragraph, y = wrap(52, y, "After fixing the centre at c, all points and cameras are expressed in these PCA coordinates. In the implementation, points are stored as row vectors, so this is written with a right multiplication by R.")
p += paragraph
y -= 7
p += equation(52, y, "u_i = R^T (x_i - c) ;  v_t = R^T (c_t - c)")
y -= 34
p += text(42, y, "3. Robust AR3 source radii", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 18
paragraph, y = wrap(52, y, "For each PCA axis j, q_j is the 99.5th percentile of the absolute point coordinate. This ignores the furthest 0.5 percent of one-dimensional extent measurements, rather than allowing isolated LiDAR outliers to set the surface.")
p += paragraph
y -= 7
p += equation(52, y, "q_j = Quantile_0.995( |u_i,j| )")
y -= 34
paragraph, y = wrap(52, y, "AR3 means no axis may be less than one third of the largest robust extent. The 15 m lower bound prevents a very thin minor axis even when the observed structure is approximately planar.")
p += paragraph
y -= 7
p += equation(52, y, "b_j = max( q_j, max( max_k q_k / 3, 15 m ) )")
p += text(42, 30, "Page 1 of 3", 8, color=(0.35, 0.35, 0.35))
pages.append(p)

p = ""
p += text(42, 744, "Robust coverage, camera constraints, and clipping", 18, bold=True, color=(0.08, 0.20, 0.38))
p += line(42, 722, 570, 722, 1.1, 0.25)
y = 698
p += text(42, y, "4. Robust uniform coverage scale", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 18
paragraph, y = wrap(52, y, "The clamped half-extents b define a preliminary ellipsoid. The 99.5th percentile of normalized point radius supplies one scalar scale s_robust. Multiplying every axis by the same scalar retains the AR3 shape while covering approximately 99.5 percent of selected points.")
p += paragraph
y -= 7
p += equation(52, y, "d_i = || u_i / b ||_2 ;  s_robust = max(1, Quantile_0.995(d_i))")
y -= 27
p += equation(52, y, "r_pre = b * s_robust")
y -= 35
p += text(42, y, "5. Camera-clearance expansion", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 18
paragraph, y = wrap(52, y, "The source-fit code also requires cameras to remain inside the ellipsoid. Let M be the largest camera normalized radius under r_pre. The first term targets a normalized radius of at most 0.4; the second is a conservative lower bound for a 15 m camera-to-surface margin. A uniform scale preserves orientation and aspect ratio.")
p += paragraph
y -= 7
p += equation(52, y, "M = max_t || v_t / r_pre ||_2")
y -= 27
p += equation(52, y, "s_cam = max( 1, M / 0.4, M + 15 m / min_j(r_pre,j) )")
y -= 27
p += equation(52, y, "r_source = r_pre * s_cam")
y -= 35
p += text(42, y, "6. NAG runtime geometry: 2% margin, then sphere cap", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 18
paragraph, y = wrap(52, y, "The JSON stores r_source. At NAG initialization, each radius first receives the configured 2 percent multiplicative margin. The matching sphere run supplies radius R_sphere, and every ellipsoid axis is capped independently by that scalar. This is the definition of the AR3-centred-clipped experiment.")
p += paragraph
y -= 7
p += equation(52, y, "r_final,j = min( 1.02 * r_source,j, R_sphere )")
y -= 35
paragraph, y = wrap(52, y, "The rendered ellipsoid surface is then the following set, where y is a world-coordinate point. R is unchanged by clipping; only the three radii change.")
p += paragraph
y -= 7
p += equation(52, y, "sum_j ( [R^T (y - c)]_j / r_final,j )^2 = 1")
y -= 36
p += fill(42, y - 45, 528, 57, 0.96)
p += text(52, y, "Important consequence", 10, bold=True, color=(0.55, 0.10, 0.08))
paragraph, _ = wrap(52, y - 15, "The sphere cap happens after the fit's camera-clearance constraints. It can therefore reduce an axis enough to weaken the original clearance guarantee. This is the known W191 128-198 pathology and is why a camera-clearance fallback is needed before general automated use.", width=500, size=8.8, leading=11)
p += paragraph
p += text(42, 30, "Page 2 of 3", 8, color=(0.35, 0.35, 0.35))
pages.append(p)

p = ""
p += text(42, 744, "Worked example: W254, frames 86-137", 18, bold=True, color=(0.08, 0.20, 0.38))
p += text(42, 722, "Values taken from frames_086_137_camera_centered_ar3_q995/ellipsoid.json", 9.2, color=(0.28, 0.28, 0.28))
p += line(42, 712, 570, 712, 1.1, 0.25)
y = 688
p += text(42, y, "Stored robust AR3 source fit", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 20
rows = [
    ("Camera-mean centre c", "[37.8550, -1.5221, 14.2546] m"),
    ("Robust q = 0.995 extents", "[74.4615, 18.4492, 3.7521] m"),
    ("After AR3 / 15 m clamp b", "[74.4615, 24.8205, 24.8205] m"),
    ("Robust uniform scale s_robust", "1.173341"),
    ("Camera scale s_cam", "1.000000 (no further expansion needed)"),
    ("Stored source radii r_source", "[87.3688, 29.1229, 29.1229] m"),
]
for number, (label, value) in enumerate(rows):
    if number % 2:
        p += fill(42, y - 14, 528, 20, 0.96)
    p += text(52, y, label, 9, bold=True)
    p += text(285, y, value, 9, mono=True)
    y -= 22
y -= 16
p += text(42, y, "Runtime expansion and clipping", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 20
rows = [
    ("After 2% NAG margin", "[89.1161, 29.7054, 29.7054] m"),
    ("Matching sphere radius R_sphere", "58.6208 m"),
    ("Final clipped AR3 radii r_final", "[58.6208, 29.7054, 29.7054] m"),
]
for number, (label, value) in enumerate(rows):
    if number % 2:
        p += fill(42, y - 14, 528, 20, 0.96)
    p += text(52, y, label, 9, bold=True)
    p += text(285, y, value, 9, mono=True)
    y -= 22
y -= 15
paragraph, y = wrap(52, y, "Thus only the long PCA axis is clipped for this interval. The two minor axes retain their AR3 thickness. The effective final aspect ratio is 58.6208 / 29.7054 = 1.973, even though AR3 constrained the pre-clipping source fit to exactly 3:1.")
p += paragraph
y -= 15
p += text(42, y, "Relation to the red reference sphere", 12, bold=True, color=(0.08, 0.20, 0.38))
y -= 18
paragraph, y = wrap(52, y, "The red sphere is centred at the same c and has radii [58.6208, 58.6208, 58.6208] m. The blue ellipsoid uses the same centre and PCA rotation, but its second and third radii are smaller. The mesh exporter writes these exact effective radii, not the unexpanded JSON radii.")
p += paragraph
y -= 16
p += fill(42, y - 51, 528, 63, 0.95)
p += text(52, y, "Sequence of operations", 10, bold=True, color=(0.08, 0.20, 0.38))
p += text(52, y - 17, "FRONT LiDAR points -> PCA orientation -> camera-mean centre -> q=.995 extents", 8.8, mono=True)
p += text(52, y - 31, "-> AR3 and 15 m clamp -> robust scale -> camera scale -> 2% NAG margin -> sphere cap", 8.8, mono=True)
p += text(42, 30, "Page 3 of 3", 8, color=(0.35, 0.35, 0.35))
pages.append(p)


def build_pdf(streams):
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    page_ids = []
    for stream in streams:
        stream_id = len(objects) + 1
        encoded = stream.encode("latin-1")
        objects.append(f"<< /Length {len(encoded)} >>\nstream\n{stream}endstream")
        page_id = len(objects) + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> /Contents {stream_id} 0 R >>"
        )
        page_ids.append(page_id)
    objects[1] = "<< /Type /Pages /Kids [" + " ".join(f"{page_id} 0 R" for page_id in page_ids) + f"] /Count {len(page_ids)} >>"
    result = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(result.encode("latin-1")))
        result += f"{number} 0 obj\n{obj}\nendobj\n"
    xref = len(result.encode("latin-1"))
    result += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    result += "".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    result += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return result.encode("latin-1")


OUT.write_bytes(build_pdf(pages))
print(OUT)
