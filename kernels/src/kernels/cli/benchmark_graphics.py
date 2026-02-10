import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernels.benchmark import TimingResults

try:
    import matplotlib
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

_HF_ORANGE = "#FF9D00"
_HF_GRAY = "#6B7280"
_HF_DARK = "#1A1A2E"
_HF_LIGHT_BG = "#FFFFFF"
_HF_DARK_BG = "#101623"
_HF_LIGHT_TEXT = "#E6EDF3"
_HF_FONT = "DejaVu Sans Mono"


def _fetch_hf_logo_svg() -> str | None:
    try:
        from urllib.request import urlopen

        url = "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"
        with urlopen(url, timeout=5) as response:
            return response.read().decode("utf-8")
    except Exception:
        return None


def _get_colors(dark: bool = False):
    if dark:
        return _HF_DARK_BG, _HF_LIGHT_TEXT, "#30363D", "#484F58"
    return _HF_LIGHT_BG, _HF_DARK, "#EDEAE3", "#D5D1C8"


def _embed_logo_in_svg(svg_path: str, logo_size: int = 24) -> None:
    import re

    logo_svg = _fetch_hf_logo_svg()
    if logo_svg is None:
        return

    with open(svg_path, "r") as f:
        content = f.read()

    viewbox_match = re.search(r'viewBox="([^"]+)"', content)
    if viewbox_match:
        parts = viewbox_match.group(1).split()
        width = float(parts[2])
        height = float(parts[3])
    else:
        width_match = re.search(r'width="([0-9.]+)', content)
        height_match = re.search(r'height="([0-9.]+)', content)
        width = float(width_match.group(1)) if width_match else 800
        height = float(height_match.group(1)) if height_match else 400

    svg_match = re.search(r"<svg[^>]*>(.*)</svg>", logo_svg, re.DOTALL)
    if svg_match is None:
        return
    logo_inner = svg_match.group(1)

    logo_x = 10
    logo_y = height - logo_size - 5
    logo_element = f'<g transform="translate({logo_x},{logo_y}) scale({logo_size/256})">{logo_inner}</g>'
    new_content = content.replace("</svg>", f"{logo_element}</svg>")

    with open(svg_path, "w") as f:
        f.write(new_content)


def _setup_figure(n_workloads: int, group_spacing: float = 1.0, dark: bool = False):
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = _HF_FONT

    bg, text, _, _ = _get_colors(dark)
    fig_height = max(4, n_workloads * group_spacing + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.subplots_adjust(top=0.80)
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    return fig, ax


def _style_axes(
    ax,
    n_workloads: int,
    group_spacing: float,
    max_val: float,
    xlabel: str,
    dark: bool = False,
):
    from matplotlib.patches import Patch

    bg, text, _, _ = _get_colors(dark)
    legend_elements = [
        Patch(facecolor=_HF_ORANGE, edgecolor="white", label="Kernel"),
        Patch(facecolor=_HF_GRAY, edgecolor="white", label="Torch (ref)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.95),
        facecolor=bg,
        edgecolor=_HF_GRAY,
        fontsize=9,
        labelcolor=text,
    )

    ax.set_xlim(0, max_val * 1.5)
    ax.set_ylim(-0.8, n_workloads * group_spacing - 0.2)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color=text, fontsize=10)
    ax.tick_params(colors=_HF_GRAY)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_HF_GRAY)
    ax.spines["bottom"].set_linewidth(0.5)


def _add_header(
    fig, title: str, backend: str, pytorch_version: str, dark: bool = False
):
    _, text, _, _ = _get_colors(dark)
    fig.text(
        0.02,
        0.98,
        title,
        fontsize=14,
        fontweight="bold",
        color=text,
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    subtitle_parts = []
    if pytorch_version:
        subtitle_parts.append(f"PyTorch {pytorch_version}")
    if backend:
        subtitle_parts.append(backend)
    if subtitle_parts:
        fig.text(
            0.98,
            0.98,
            " . ".join(subtitle_parts),
            fontsize=10,
            color=_HF_GRAY,
            ha="right",
            va="top",
            transform=fig.transFigure,
        )


def _format_ops_per_sec(ops: float) -> str:
    if ops >= 1_000_000:
        return f"{ops / 1_000_000:.1f}M ops/s"
    elif ops >= 1_000:
        return f"{ops / 1_000:.1f}k ops/s"
    return f"{ops:.0f} ops/s"


def save_speedup_image(
    results: dict[str, "TimingResults"],
    path: str,
    backend: str = "",
    repo_id: str = "",
    pytorch_version: str = "",
    dark: bool = False,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print(
            "Error: matplotlib required. Install with: pip install 'kernels[benchmark]'",
            file=sys.stderr,
        )
        return

    workloads = [
        (name, results[name])
        for name in sorted(results.keys())
        if results[name].ref_mean_ms is not None and results[name].mean_ms > 0
    ]
    if not workloads:
        print(
            "No reference timings available, skipping image generation.",
            file=sys.stderr,
        )
        return

    _, text, _, _ = _get_colors(dark)
    n_workloads = len(workloads)
    bar_height, group_spacing = 0.20, 1.0
    fig, ax = _setup_figure(n_workloads, group_spacing, dark)

    all_times: list[float] = [t.mean_ms for _, t in workloads]
    all_times += [t.ref_mean_ms for _, t in workloads if t.ref_mean_ms is not None]
    max_time = max(all_times) if all_times else 1.0

    for i, (name, t) in enumerate(workloads):
        base_y = (n_workloads - 1 - i) * group_spacing
        ref_mean = t.ref_mean_ms if t.ref_mean_ms is not None else t.mean_ms
        speedup = ref_mean / t.mean_ms

        y_kern = base_y + bar_height / 2 + 0.05
        ax.barh(
            y_kern,
            t.mean_ms,
            height=bar_height,
            color=_HF_ORANGE,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            t.mean_ms + max_time * 0.02,
            y_kern,
            f"{t.mean_ms:.2f} ms",
            va="center",
            ha="left",
            fontsize=9,
            color=text,
        )

        y_ref = base_y - bar_height / 2 - 0.05
        ax.barh(
            y_ref,
            ref_mean,
            height=bar_height,
            color=_HF_GRAY,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            ref_mean + max_time * 0.02,
            y_ref,
            f"{ref_mean:.2f} ms",
            va="center",
            ha="left",
            fontsize=9,
            color=text,
        )

        ax.text(
            -max_time * 0.02,
            base_y,
            name,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
            color=text,
        )

        speedup_text = (
            f"  {speedup:.2f}x faster"
            if speedup >= 1.0
            else f"  {1/speedup:.2f}x slower"
        )
        speedup_color = _HF_ORANGE if speedup >= 1.0 else _HF_GRAY
        ax.text(
            max(t.mean_ms, ref_mean) + max_time * 0.15,
            base_y,
            speedup_text,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=speedup_color,
        )

    _style_axes(
        ax,
        n_workloads,
        group_spacing,
        max_time,
        "Time (ms)  <-  shorter is better",
        dark,
    )
    _add_header(
        fig,
        f"{repo_id} vs Torch - Latency" if repo_id else "Kernel vs Torch",
        backend,
        pytorch_version,
        dark,
    )

    if "." in path:
        base, ext = path.rsplit(".", 1)
        latency_path = f"{base}_latency.{ext}"
    else:
        latency_path = f"{path}_latency"

    fig.tight_layout()
    fig.savefig(latency_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    if latency_path.endswith(".svg"):
        _embed_logo_in_svg(latency_path)
    print(f"Latency chart saved to: {latency_path}", file=sys.stderr)

    _save_ops_per_sec_image(workloads, path, backend, repo_id, pytorch_version, dark)


def _save_ops_per_sec_image(
    workloads: list[tuple[str, "TimingResults"]],
    base_path: str,
    backend: str = "",
    repo_id: str = "",
    pytorch_version: str = "",
    dark: bool = False,
) -> None:
    if "." in base_path:
        base, ext = base_path.rsplit(".", 1)
        throughput_path = f"{base}_throughput.{ext}"
    else:
        throughput_path = f"{base_path}_throughput"

    _, text, _, _ = _get_colors(dark)
    n_workloads = len(workloads)
    bar_height, group_spacing = 0.20, 1.0
    fig, ax = _setup_figure(n_workloads, group_spacing, dark)

    all_ops: list[float] = []
    for _, t in workloads:
        all_ops.append(1000.0 / t.mean_ms)
        if t.ref_mean_ms is not None:
            all_ops.append(1000.0 / t.ref_mean_ms)
    max_ops = max(all_ops) if all_ops else 1.0

    for i, (name, t) in enumerate(workloads):
        base_y = (n_workloads - 1 - i) * group_spacing
        ref_mean = t.ref_mean_ms if t.ref_mean_ms is not None else t.mean_ms
        kernel_ops, ref_ops = 1000.0 / t.mean_ms, 1000.0 / ref_mean
        speedup = kernel_ops / ref_ops

        y_kern = base_y + bar_height / 2 + 0.05
        ax.barh(
            y_kern,
            kernel_ops,
            height=bar_height,
            color=_HF_ORANGE,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            kernel_ops + max_ops * 0.02,
            y_kern,
            _format_ops_per_sec(kernel_ops),
            va="center",
            ha="left",
            fontsize=9,
            color=text,
        )

        y_ref = base_y - bar_height / 2 - 0.05
        ax.barh(
            y_ref,
            ref_ops,
            height=bar_height,
            color=_HF_GRAY,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            ref_ops + max_ops * 0.02,
            y_ref,
            _format_ops_per_sec(ref_ops),
            va="center",
            ha="left",
            fontsize=9,
            color=text,
        )

        ax.text(
            -max_ops * 0.02,
            base_y,
            name,
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
            color=text,
        )

        speedup_text = (
            f"  {speedup:.2f}x faster"
            if speedup >= 1.0
            else f"  {1/speedup:.2f}x slower"
        )
        speedup_color = _HF_ORANGE if speedup >= 1.0 else _HF_GRAY
        ax.text(
            max(kernel_ops, ref_ops) + max_ops * 0.15,
            base_y,
            speedup_text,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=speedup_color,
        )

    _style_axes(
        ax,
        n_workloads,
        group_spacing,
        max_ops,
        "Operations per second  ->  longer is better",
        dark,
    )
    _add_header(
        fig,
        (
            f"{repo_id} vs Torch - Throughput"
            if repo_id
            else "Kernel vs Torch - Throughput"
        ),
        backend,
        pytorch_version,
        dark,
    )

    fig.tight_layout()
    fig.savefig(throughput_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    if throughput_path.endswith(".svg"):
        _embed_logo_in_svg(throughput_path)
    print(f"Throughput chart saved to: {throughput_path}", file=sys.stderr)


def save_speedup_animation(
    results: dict[str, "TimingResults"],
    path: str,
    backend: str = "",
    repo_id: str = "",
    pytorch_version: str = "",
    dark: bool = False,
) -> None:
    workloads = []
    for name in sorted(results.keys()):
        t = results[name]
        if t.ref_mean_ms is not None and t.mean_ms > 0:
            workloads.append((name, t.ref_mean_ms / t.mean_ms))

    if not workloads:
        print("No reference timings available, skipping animation.", file=sys.stderr)
        return

    if path.endswith(".gif"):
        _save_speedup_gif(workloads, path, backend, repo_id, pytorch_version, dark)
    else:
        _save_speedup_svg(workloads, path, backend, repo_id, pytorch_version, dark)


def _save_speedup_gif(
    workloads: list[tuple[str, float]],
    path: str,
    backend: str,
    repo_id: str,
    pytorch_version: str,
    dark: bool,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print(
            "Error: matplotlib required. Install with: pip install 'kernels[benchmark]'",
            file=sys.stderr,
        )
        return

    import math
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        print(
            "Error: Pillow required for GIF output. Install with: pip install Pillow",
            file=sys.stderr,
        )
        return

    from matplotlib.patches import FancyBboxPatch, Ellipse
    from urllib.request import urlopen

    matplotlib.use("Agg")
    plt.rcParams["font.family"] = _HF_FONT

    hf_logo = None
    try:
        logo_url = "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"
        with urlopen(logo_url, timeout=5) as response:
            logo_data = BytesIO(response.read())
            hf_logo = Image.open(logo_data).convert("RGBA")
            resample = getattr(Image, "Resampling", Image).LANCZOS
            hf_logo = hf_logo.resize((24, 24), resample)
    except Exception:
        pass

    bg, text, track_bg, track_border = _get_colors(dark)
    n_rows = len(workloads)

    svg_width, svg_row_height, svg_padding = 800, 50, 120
    svg_height = n_rows * svg_row_height + svg_padding
    fig_width = 11
    fig_height = fig_width * svg_height / svg_width

    track_x, track_w = 180, 470
    track_start = track_x / svg_width
    track_end = (track_x + track_w) / svg_width
    ball_r = 8 / svg_width

    title = (
        f"{repo_id} vs Torch - Relative Speed"
        if repo_id
        else "Kernel vs Torch - Relative Speed"
    )
    subtitle = " · ".join(
        filter(None, [f"PyTorch {pytorch_version}" if pytorch_version else "", backend])
    )

    ref_dur = 2.0
    fps = 30
    total_frames = int(ref_dur * fps)

    frames = []
    for frame in range(total_frames):
        t = frame / total_frames

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.axis("off")

        title_x = 10 / svg_width
        title_y = 1.0 - 25 / svg_height
        fig.text(
            title_x,
            title_y,
            title,
            fontsize=14,
            fontweight="bold",
            color=text,
            ha="left",
            va="center",
            transform=fig.transFigure,
        )
        if subtitle:
            fig.text(
                1.0 - 10 / svg_width,
                title_y,
                subtitle,
                fontsize=10,
                color=_HF_GRAY,
                ha="right",
                va="center",
                transform=fig.transFigure,
            )

        row_height_norm = svg_row_height / svg_height
        first_row_y = 1.0 - (50 + svg_row_height // 2) / svg_height
        aspect_correction = fig_width / fig_height

        for i, (name, speedup) in enumerate(workloads):
            y = first_row_y - i * row_height_norm

            track_height = 30 / svg_height
            track = FancyBboxPatch(
                (track_start, y - track_height / 2),
                track_end - track_start,
                track_height,
                boxstyle="round,pad=0.01",
                facecolor=track_bg,
                edgecolor=track_border,
            )
            ax.add_patch(track)

            text_offset = 20 / svg_width
            ax.text(
                track_start - text_offset,
                y,
                name,
                ha="right",
                va="center",
                fontsize=10,
                color=text,
            )
            ax.text(
                track_end + text_offset,
                y,
                f"{speedup:.2f}x",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text,
            )

            kernel_period = 1.0 / speedup
            kernel_t = (t % kernel_period) / kernel_period
            kernel_phase = math.sin(kernel_t * math.pi)
            kernel_x = (
                track_start
                + ball_r
                + kernel_phase * (track_end - track_start - 2 * ball_r)
            )

            ref_phase = math.sin(t * math.pi)
            ref_x = (
                track_start
                + ball_r
                + ref_phase * (track_end - track_start - 2 * ball_r)
            )

            ball_offset = 6 / svg_height
            kernel_ball = Ellipse(
                (kernel_x, y + ball_offset),
                ball_r * 2,
                ball_r * 2 * aspect_correction,
                facecolor=_HF_ORANGE,
                edgecolor="white",
                linewidth=1.5,
            )
            ref_ball = Ellipse(
                (ref_x, y - ball_offset),
                ball_r * 2,
                ball_r * 2 * aspect_correction,
                facecolor=_HF_GRAY,
                edgecolor="white",
                linewidth=1.5,
            )
            ax.add_patch(kernel_ball)
            ax.add_patch(ref_ball)

        from matplotlib.patches import Ellipse as MplEllipse

        legend_y = 20 / svg_height
        circle_h = 12 / svg_height
        circle_w = circle_h * fig_height / fig_width
        legend_offset = 20
        orange_x = (svg_width - 150 - legend_offset) / svg_width
        kernel_text_x = (svg_width - 138 - legend_offset) / svg_width
        gray_x = (svg_width - 70 - legend_offset) / svg_width
        ref_text_x = (svg_width - 58 - legend_offset) / svg_width
        fig.patches.append(
            MplEllipse(
                (orange_x, legend_y),
                circle_w,
                circle_h,
                facecolor=_HF_ORANGE,
                edgecolor="white",
                linewidth=1,
                transform=fig.transFigure,
            )
        )
        fig.text(
            kernel_text_x,
            legend_y,
            "Kernel",
            ha="left",
            va="center",
            fontsize=9,
            color=text,
        )
        fig.patches.append(
            MplEllipse(
                (gray_x, legend_y),
                circle_w,
                circle_h,
                facecolor=_HF_GRAY,
                edgecolor="white",
                linewidth=1,
                transform=fig.transFigure,
            )
        )
        fig.text(
            ref_text_x,
            legend_y,
            "Torch (ref)",
            ha="left",
            va="center",
            fontsize=9,
            color=text,
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=100)
        buf.seek(0)
        frame_img = Image.open(buf).convert("RGBA")

        if hf_logo is not None:
            logo_x = 10
            logo_y = frame_img.height - hf_logo.height - 10
            frame_img.paste(hf_logo, (logo_x, logo_y), hf_logo)

        frames.append(frame_img.convert("RGB"))
        plt.close(fig)

    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0
    )
    print(f"Animated GIF saved to: {path}", file=sys.stderr)


def _save_speedup_svg(
    workloads: list[tuple[str, float]],
    path: str,
    backend: str,
    repo_id: str,
    pytorch_version: str,
    dark: bool,
) -> None:
    bg, text, track_bg, track_border = _get_colors(dark)
    n_rows = len(workloads)
    width, row_height, padding = 800, 50, 120
    height = n_rows * row_height + padding
    track_x, track_w, ball_r = 180, 470, 8
    x_min, x_max = track_x + ball_r, track_x + track_w - ball_r

    title = (
        f"{repo_id} vs Torch - Relative Speed"
        if repo_id
        else "Kernel vs Torch - Relative Speed"
    )
    subtitle = " · ".join(
        filter(None, [f"PyTorch {pytorch_version}" if pytorch_version else "", backend])
    )

    ref_dur = 2.0

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:{bg};font-family:{_HF_FONT},monospace">',
        f'<text x="10" y="25" font-size="14" font-weight="bold" fill="{text}">{title}</text>',
    ]
    if subtitle:
        svg_parts.append(
            f'<text x="{width-10}" y="25" font-size="10" fill="{_HF_GRAY}" text-anchor="end">{subtitle}</text>'
        )

    for i, (name, speedup) in enumerate(workloads):
        y = 50 + i * row_height + row_height // 2
        kernel_dur = ref_dur / speedup

        svg_parts.extend(
            [
                f'<rect x="{track_x}" y="{y-15}" width="{track_w}" height="30" rx="4" fill="{track_bg}" stroke="{track_border}"/>',
                f'<text x="{track_x-10}" y="{y+4}" font-size="10" fill="{text}" text-anchor="end">{name}</text>',
                f'<text x="{track_x+track_w+10}" y="{y+4}" font-size="10" font-weight="bold" fill="{text}">{speedup:.2f}x</text>',
                f'<circle cx="{x_min}" cy="{y-6}" r="{ball_r}" fill="{_HF_ORANGE}" stroke="white" stroke-width="1.5">',
                f'  <animate attributeName="cx" values="{x_min};{x_max};{x_min}" dur="{kernel_dur}s" repeatCount="indefinite" calcMode="spline" keySplines="0.5 0 0.5 1;0.5 0 0.5 1"/>',
                f"</circle>",
                f'<circle cx="{x_min}" cy="{y+6}" r="{ball_r}" fill="{_HF_GRAY}" stroke="white" stroke-width="1.5">',
                f'  <animate attributeName="cx" values="{x_min};{x_max};{x_min}" dur="{ref_dur}s" repeatCount="indefinite" calcMode="spline" keySplines="0.5 0 0.5 1;0.5 0 0.5 1"/>',
                f"</circle>",
            ]
        )

    legend_y = height - 20
    svg_parts.extend(
        [
            f'<circle cx="{width-150}" cy="{legend_y}" r="6" fill="{_HF_ORANGE}" stroke="white"/>',
            f'<text x="{width-138}" y="{legend_y+4}" font-size="9" fill="{text}">Kernel</text>',
            f'<circle cx="{width-70}" cy="{legend_y}" r="6" fill="{_HF_GRAY}" stroke="white"/>',
            f'<text x="{width-58}" y="{legend_y+4}" font-size="9" fill="{text}">Torch (ref)</text>',
            "</svg>",
        ]
    )

    svg_path = path.rsplit(".", 1)[0] + ".svg" if "." in path else path + ".svg"
    with open(svg_path, "w") as f:
        f.write("\n".join(svg_parts))
    _embed_logo_in_svg(svg_path)
    print(f"Animated SVG saved to: {svg_path}", file=sys.stderr)
