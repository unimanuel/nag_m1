"""Create a cohort-separated PDF comparison of completed ellipsoid treatments."""

import argparse
import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/mnt/d/CGProjectData/waymo/diagnostics/texture_sampling_v1/metrics.csv")
DEFAULT_OUTPUT = REPO / "reports/2026-09-18_pathological_vs_nonpathological_comparison.pdf"
W, H = 792, 612
LEFT, RIGHT = 36, 756
NAVY, BLACK, GREY = (0.06, 0.18, 0.34), (0.03, 0.03, 0.03), (0.26, 0.26, 0.26)

PATHOLOGICAL = ("W191_0_40", "W254_0_37", "W254_139_195")
NON_PATHOLOGICAL = ("W191_40_128", "W191_128_198", "W254_37_81", "W254_86_137")
METHODS = ("sphere", "ar3", "raw_v4", "moderate_y", "reprojection_v1", "reprojection_new_radius_scale_v1")
METHOD_NAMES = {
    "sphere": "Sphere",
    "ar3": "Sphere-clipped AR3",
    "raw_v4": "Raw v4 depth proxy",
    "moderate_y": "v4 Moderate-Y",
    "reprojection_v1": "Original reprojection v1",
    "reprojection_new_radius_scale_v1": "Scale-limited reprojection",
}


def esc(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text(x, y, value, size=9, bold=False, color=None):
    color = color or BLACK
    return (
        "BT\n{:.3f} {:.3f} {:.3f} rg\n/{} {} Tf\n{:.1f} {:.1f} Td\n({}) Tj\nET\n".format(
            *color, "F2" if bold else "F1", size, x, y, esc(value)
        )
    )


def line(x1, y1, x2, y2, width=.5, grey=.72):
    return "{:.3f} G\n{:.2f} w\n{:.1f} {:.1f} m\n{:.1f} {:.1f} l\nS\n".format(grey, width, x1, y1, x2, y2)


def fill(x, y, width, height, grey):
    return "{:.3f} g\n{:.1f} {:.1f} {:.1f} {:.1f} re\nf\n".format(grey, x, y, width, height)


def wrap(value, width, size):
    capacity = max(12, int(width / (size * .51)))
    rows, current = [], ""
    for word in value.split():
        candidate = (current + " " + word).strip()
        if current and len(candidate) > capacity:
            rows.append(current)
            current = word
        else:
            current = candidate
    if current:
        rows.append(current)
    return rows


def wrapped(x, y, value, width, size=9, leading=12):
    result = ""
    for row in wrap(value, width, size):
        result += text(x, y, row, size)
        y -= leading
    return result, y


def heading(page, subtitle):
    result = text(LEFT, 568, "Reprojection ellipsoid: pathological versus non-pathological results", 17, True, NAVY)
    result += text(LEFT, 545, subtitle, 9.5, color=GREY)
    result += line(LEFT, 535, RIGHT, 535, 1.1, .25)
    return result


def section(result, y, title):
    return result + text(LEFT, y, title, 12, True, NAVY), y - 17


def table(y, headers, rows, widths, font_size=7.5, row_height=20):
    total = sum(widths)
    assert total <= RIGHT - LEFT
    result, x = fill(LEFT, y - row_height + 2, total, row_height, .82), LEFT + 4
    for header, width in zip(headers, widths):
        result += text(x, y - 12, header, font_size, True)
        x += width
    y -= row_height
    for index, row in enumerate(rows):
        if index % 2:
            result += fill(LEFT, y - row_height + 2, total, row_height, .96)
        x = LEFT + 4
        for item, width in zip(row, widths):
            result += text(x, y - 12, item, font_size)
            x += width
        result += line(LEFT, y - row_height + 2, LEFT + total, y - row_height + 2, .3, .82)
        y -= row_height
    return result, y


def footer(result, page, note):
    result += line(LEFT, 42, RIGHT, 42, .45, .72)
    result += text(LEFT, 28, note, 7.4, color=GREY)
    return result + text(RIGHT - 72, 28, "Page {} of 6".format(page), 7.4, color=GREY)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"sequence", "family", "mode", "MPSNR", "SSIM", "LPIPS"}
    if not rows or required.difference(rows[0]):
        raise ValueError("Input metrics CSV lacks required columns.")
    filtered = []
    for row in rows:
        if row["family"] not in METHODS or row["mode"] not in ("regular", "visuv"):
            continue
        row = dict(row)
        for key in ("MPSNR", "SSIM", "LPIPS"):
            row[key] = float(row[key])
        filtered.append(row)
    for sequence in PATHOLOGICAL + NON_PATHOLOGICAL:
        for mode in ("regular", "visuv"):
            found = {row["family"] for row in filtered if row["sequence"] == sequence and row["mode"] == mode}
            expected = set(METHODS if sequence in PATHOLOGICAL else METHODS[:-1])
            if found != expected:
                raise ValueError("{} {}: expected {}, found {}".format(sequence, mode, sorted(expected), sorted(found)))
    return filtered


def row_for(rows, sequence, family, mode):
    return next(row for row in rows if row["sequence"] == sequence and row["family"] == family and row["mode"] == mode)


def average(rows, sequences, family, mode):
    entries = [row_for(rows, sequence, family, mode) for sequence in sequences]
    return {key: sum(row[key] for row in entries) / len(entries) for key in ("MPSNR", "SSIM", "LPIPS")}


def triplet(values):
    return "{:.3f} / {:.4f} / {:.4f}".format(values["MPSNR"], values["SSIM"], values["LPIPS"])


def delta(values, sphere):
    return "{:+.3f} / {:+.4f} / {:+.4f}".format(
        values["MPSNR"] - sphere["MPSNR"], values["SSIM"] - sphere["SSIM"], values["LPIPS"] - sphere["LPIPS"]
    )


def individual_rows(rows, sequences, mode):
    output = []
    for sequence in sequences:
        sphere = row_for(rows, sequence, "sphere", mode)
        for family in METHODS:
            try:
                values = row_for(rows, sequence, family, mode)
            except StopIteration:
                continue
            output.append((sequence.replace("_", " "), METHOD_NAMES[family], "{:.4f}".format(values["MPSNR"]),
                           "{:.5f}".format(values["SSIM"]), "{:.5f}".format(values["LPIPS"]),
                           "{:+.3f}".format(values["MPSNR"] - sphere["MPSNR"]),
                           "{:+.4f}".format(values["SSIM"] - sphere["SSIM"]),
                           "{:+.4f}".format(values["LPIPS"] - sphere["LPIPS"])))
    return output


def build_pdf(streams):
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
    pages = []
    for stream in streams:
        content = stream.encode("latin-1")
        content_id = len(objects) + 1
        objects.append("<< /Length {} >>\nstream\n{}endstream".format(len(content), stream))
        page_id = len(objects) + 1
        objects.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {} 0 R >>".format(W, H, content_id))
        pages.append(page_id)
    objects[1] = "<< /Type /Pages /Kids [{}] /Count {} >>".format(" ".join("{} 0 R".format(page) for page in pages), len(pages))
    body, offsets = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n", [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(body.encode("latin-1")))
        body += "{} 0 obj\n{}\nendobj\n".format(number, obj)
    xref = len(body.encode("latin-1"))
    body += "xref\n0 {}\n0000000000 65535 f\n".format(len(objects) + 1)
    body += "".join("{:010d} 00000 n\n".format(offset) for offset in offsets[1:])
    body += "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(len(objects) + 1, xref)
    return body.encode("latin-1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = read_rows(args.input)
    pages = []

    p = heading(1, "Verified final metrics; equal-sequence cohort averages and every available individual comparison")
    y = 514
    p, y = section(p, y, "Cohort definition")
    para, y = wrapped(LEFT + 12, y, "Pathological means the original reprojection v1 fit expanded its lateral major axis to 157-171 m: W191 0-40, W254 0-37 and W254 139-195. Non-pathological means the remaining four original-v1 fits: W191 40-128, W191 128-198, W254 37-81 and W254 86-137. This is an outcome-based label for the original optimisation, not a claim that the scenes are intrinsically pathological.", 696, 9.0, 12)
    p += para
    y -= 12
    p, y = section(p, y, "Cohort averages - regular mapping")
    average_rows = []
    for family in METHODS:
        path = average(rows, PATHOLOGICAL, family, "regular")
        non = average(rows, NON_PATHOLOGICAL, family, "regular") if family != METHODS[-1] else None
        average_rows.append((METHOD_NAMES[family], triplet(path), triplet(non) if non else "not trained", delta(path, average(rows, PATHOLOGICAL, "sphere", "regular"))))
    part, y = table(y, ["Treatment", "Pathological P / S / L", "Non-pathological P / S / L", "Pathological vs sphere"], average_rows, [205, 178, 178, 159], 7.4, 20)
    p += part
    y -= 17
    p, y = section(p, y, "Cohort averages - Visible-UV mapping")
    average_rows = []
    for family in METHODS:
        path = average(rows, PATHOLOGICAL, family, "visuv")
        non = average(rows, NON_PATHOLOGICAL, family, "visuv") if family != METHODS[-1] else None
        average_rows.append((METHOD_NAMES[family], triplet(path), triplet(non) if non else "not trained", delta(path, average(rows, PATHOLOGICAL, "sphere", "visuv"))))
    part, y = table(y, ["Treatment", "Pathological P / S / L", "Non-pathological P / S / L", "Pathological vs sphere"], average_rows, [205, 178, 178, 159], 7.4, 20)
    p += part
    y -= 14
    para, y = wrapped(LEFT + 12, y, "P / S / L means PSNR / SSIM / LPIPS. Higher PSNR and SSIM are better; lower LPIPS is better. The scale-limited result exists only for the three pathological intervals, so it has no non-pathological cohort mean.", 696, 8.6, 11)
    p += para
    pages.append(footer(p, 1, "Data: D:/CGProjectData/waymo/diagnostics/texture_sampling_v1/metrics.csv. Results are one training run per treatment."))

    for page_number, cohort_name, sequences, mode in ((2, "Pathological individual comparison", PATHOLOGICAL, "regular"),
                                                        (3, "Pathological individual comparison", PATHOLOGICAL, "visuv"),
                                                        (4, "Non-pathological individual comparison", NON_PATHOLOGICAL, "regular"),
                                                        (5, "Non-pathological individual comparison", NON_PATHOLOGICAL, "visuv")):
        p = heading(page_number, "{} - {} mapping; delta columns are treatment minus same-sequence sphere".format(cohort_name, "regular" if mode == "regular" else "Visible-UV"))
        y = 514
        p, y = section(p, y, cohort_name + " - " + ("regular" if mode == "regular" else "Visible-UV"))
        desc = "All treatments use the saved effective geometry for that completed run. Scale-limited reprojection was trained only for the three original-v1 lateral-expansion intervals." if sequences == PATHOLOGICAL else "Each row compares the original reprojection v1 with completed sphere, AR3, raw-v4 and Moderate-Y treatments on exactly the named interval and texture layout."
        para, y = wrapped(LEFT + 12, y, desc, 696, 8.7, 11)
        p += para
        y -= 8
        part, y = table(y, ["Interval", "Treatment", "PSNR", "SSIM", "LPIPS", "dP", "dS", "dL"], individual_rows(rows, sequences, mode), [90, 180, 85, 85, 85, 65, 65, 65], 7.15, 19)
        p += part
        pages.append(footer(p, page_number, "dP/dS/dL are differences versus the same interval and mapping's sphere. Positive dL means worse LPIPS."))

    p = heading(6, "Interpretation of the cohort split and what it establishes")
    y = 514
    p, y = section(p, y, "Original reprojection v1: the split is real")
    regular_path = average(rows, PATHOLOGICAL, "reprojection_v1", "regular")
    regular_non = average(rows, NON_PATHOLOGICAL, "reprojection_v1", "regular")
    uv_path = average(rows, PATHOLOGICAL, "reprojection_v1", "visuv")
    uv_non = average(rows, NON_PATHOLOGICAL, "reprojection_v1", "visuv")
    first = "On the four non-pathological intervals, original reprojection v1 beats the sphere cohort mean in all three metrics: regular delta {} and Visible-UV delta {}. On the three pathological intervals, it trails sphere by regular delta {} and Visible-UV delta {}. Therefore the overall seven-interval average hides two opposite behaviours rather than a uniformly weak method.".format(delta(regular_non, average(rows, NON_PATHOLOGICAL, "sphere", "regular")), delta(uv_non, average(rows, NON_PATHOLOGICAL, "sphere", "visuv")), delta(regular_path, average(rows, PATHOLOGICAL, "sphere", "regular")), delta(uv_path, average(rows, PATHOLOGICAL, "sphere", "visuv")))
    para, y = wrapped(LEFT + 12, y, first, 696, 9.1, 13)
    p += para
    y -= 11
    p, y = section(p, y, "Effect of source-scale limitation on the three failures")
    effect_rows = []
    for mode in ("regular", "visuv"):
        original, scale = average(rows, PATHOLOGICAL, "reprojection_v1", mode), average(rows, PATHOLOGICAL, "reprojection_new_radius_scale_v1", mode)
        effect_rows.append(("Regular" if mode == "regular" else "Visible-UV", triplet(original), triplet(scale), delta(scale, original)))
    part, y = table(y, ["Mapping", "Original v1 P / S / L", "Scale-limited P / S / L", "Scale minus original"], effect_rows, [115, 200, 200, 205], 8.0, 27)
    p += part
    y -= 17
    p, y = section(p, y, "How to read prior treatments")
    for statement in (
        "Moderate-Y has the strongest pathological cohort mean for SSIM and LPIPS in both mappings, while sphere has the strongest pathological regular PSNR. It remains a strong prior treatment rather than a failure baseline.",
        "The original reprojection objective is the strongest non-pathological cohort treatment on average across the listed methods. That is the result the new unified sampling-floor experiment must preserve.",
        "The scale-limited prior is a meaningful repair: it recovers 0.485 dB regular and 0.755 dB Visible-UV PSNR over original v1 on the pathological cohort. It is still behind the established best treatment on several individual rows, so it should not yet be called a universal solution.",
        "This report does not include the untrained sampling-floor candidates. They were generated after these completed results and require their own training comparison.",
    ):
        para, y = wrapped(LEFT + 12, y, "- " + statement, 690, 8.8, 12)
        p += para
        y -= 3
    pages.append(footer(p, 6, "Cohort averages weight every interval equally. Close differences require replication before a final ranking."))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(build_pdf(pages))
    print(args.output)


if __name__ == "__main__":
    main()
