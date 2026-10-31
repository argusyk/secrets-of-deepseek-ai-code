"""Dynamic tiling of high-resolution images (in the spirit of DeepSeek-VL2).

The vision encoder works on fixed 384x384 inputs. A large image of arbitrary
aspect ratio is resized to one of the candidate resolutions (m*384, n*384)
with m*n <= max_tiles. The choice below first keeps as much of the original
detail as possible (upscaling earns nothing) and then, among equal options,
wastes the least area on padding; the image is cut into m*n local tiles and
one downscaled global thumbnail is added. See Chapter 17 of "Secrets of DeepSeek AI".

Run:
    python dynamic_tiling.py
"""
TILE = 384


def candidates(max_tiles: int = 9):
    return [(m, n) for m in range(1, max_tiles + 1) for n in range(1, max_tiles + 1)
            if m * n <= max_tiles]


def fit(width: int, height: int, m: int, n: int):
    """Scale the image into an (m*TILE) x (n*TILE) canvas keeping the aspect
    ratio. Returns (effective_pixels, padding_pixels, scale)."""
    cw, ch = m * TILE, n * TILE
    s = min(cw / width, ch / height)
    w, h = width * s, height * s
    effective = min(w * h, width * height)    # upscaling adds no real detail
    padding = cw * ch - w * h
    return effective, padding, s


def choose_grid(width: int, height: int, max_tiles: int = 9):
    """Maximize preserved detail, then minimize padding, then use fewer tiles."""
    def key(mn):
        eff, pad, _ = fit(width, height, *mn)
        return (-round(eff), round(pad), mn[0] * mn[1])
    return min(candidates(max_tiles), key=key)


def visual_tokens(m: int, n: int, tokens_per_tile: int = 196) -> int:
    """Local tiles plus one global thumbnail; tokens_per_tile is illustrative."""
    return (m * n + 1) * tokens_per_tile


if __name__ == "__main__":
    for w, h, what in [(384, 384, "icon"), (1920, 1080, "screenshot"),
                       (1240, 1754, "A4 page scan"), (3000, 600, "wide banner"),
                       (600, 4000, "long receipt")]:
        m, n = choose_grid(w, h)
        eff, pad, scale = fit(w, h, m, n)
        share = pad / (m * n * TILE * TILE)
        print(f"{what:13s} {w:5d}x{h:<5d} -> grid {m}x{n} ({m*n} tiles), "
              f"padding {share:5.1%}, scale {min(scale, 1):.2f}, ~{visual_tokens(m, n)} visual tokens")
