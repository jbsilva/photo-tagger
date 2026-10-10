"""The full-size viewer's zoom steps and scroll anchoring."""

# Zoom limits for the full-size image viewer, as a multiple of the image's own pixel size. The
# floor keeps a huge photo reachable in one "fit" step; the ceiling stops a click-and-hold from
# asking Qt to scale a 40-megapixel image into gigabytes of pixmap.
MIN_ZOOM = 0.02


MAX_ZOOM = 8.0


# One notch of zoom in or out. 1.25 is small enough to aim with and large enough to feel like
# progress: eight notches cover the whole 1:10 range a fit-to-window view usually spans.
ZOOM_STEP = 1.25


def clamp_zoom(zoom: float) -> float:
    """Hold a zoom factor inside the viewer's usable range."""
    return min(MAX_ZOOM, max(MIN_ZOOM, zoom))


def fit_zoom(image: tuple[int, int], viewport: tuple[int, int]) -> float:
    """
    Zoom factor that makes *image* fit entirely inside *viewport*.

    Never goes above 1.0: fitting a small photo to a big window would upscale it into a blurry mess,
    so a photo smaller than the window is shown at its own size instead. A degenerate size (a failed
    decode, or a window not laid out yet) falls back to 1.0.
    """
    width, height = image
    view_width, view_height = viewport
    if width <= 0 or height <= 0 or view_width <= 0 or view_height <= 0:
        return 1.0
    return clamp_zoom(min(1.0, view_width / width, view_height / height))


def step_zoom(zoom: float, notches: int) -> float:
    """Move *zoom* by *notches* steps in or out (negative zooms out), clamped to the range."""
    return clamp_zoom(zoom * ZOOM_STEP**notches)


def zoom_label(zoom: float) -> str:
    """Render a zoom factor as the percentage shown in the viewer's toolbar."""
    return f"{round(zoom * 100)}%"


def anchored_scroll(value: int, page_step: int, factor: float) -> int:
    """
    Where a scrollbar must land so the viewport keeps showing the same part of the image.

    After the image is rescaled by *factor*, the pixel that was at the center of the viewport has
    moved to ``factor`` times its old offset. Scrolling to that offset, minus half a viewport, puts
    it back under the user's eyes; without this, zooming walks the view toward the top-left corner.
    Negative results are clamped, since a scrollbar cannot go below its minimum.
    """
    return max(0, round(factor * value + (factor - 1) * page_step / 2))
