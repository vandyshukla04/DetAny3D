"""Tests for the dense-feature layer and the anatomical template. No GPU, no model, no data.

    python -m tools.heading.tests.test_descriptors

These cover the pieces that would fail SILENTLY -- producing a plausible number rather than an
exception. The reversal identity in particular is load-bearing: `AxisTemplate.score_faces`
computes only 2 profiles for 4 hypotheses, deriving the other two by reversing. If that identity
broke, half the scores would be quietly wrong and every downstream table would still look fine.
"""
from __future__ import annotations

import numpy as np

from tools.heading.descriptors import axis_profile, foreground
from tools.heading.template import _centre, _match, opposite_slot


def test_foreground_keeps_the_centre_and_drops_the_corners():
    """The crop is cut from the animal's own 2D box, so the centre is animal and (after
    square-padding) the corners are grass. The mask must exploit exactly that."""
    gh = gw = 14
    D = 16
    g = np.zeros((gh, gw, D))
    g[..., 0] = 1.0                                     # background direction
    cy, cx = gh // 2, gw // 2
    g[cy - 4:cy + 4, cx - 4:cx + 4] = 0
    g[cy - 4:cy + 4, cx - 4:cx + 4, 1] = 1.0            # animal direction

    fg = foreground(g)
    inside = fg[cy - 3:cy + 3, cx - 3:cx + 3].mean()
    corners = np.concatenate([fg[:2, :2].ravel(), fg[:2, -2:].ravel(),
                              fg[-2:, :2].ravel(), fg[-2:, -2:].ravel()]).mean()
    assert inside > 0.95, f"animal centre was masked out ({inside:.2f})"
    assert corners < 0.05, f"grass corners were kept as foreground ({corners:.2f})"


def test_foreground_never_returns_an_empty_mask():
    """A degenerate crop (all patches identical) must fall back to 'everything', not to nothing --
    an empty mask would make every downstream profile silently zero."""
    g = np.zeros((14, 14, 8))
    g[..., 0] = 1.0
    assert foreground(g).sum() > 0


def test_reversing_the_hypothesis_exactly_reverses_the_profile():
    """LOAD-BEARING. score_faces derives the opposite hypothesis' score by reversing the profile
    instead of recomputing it. If this identity fails, 2 of the 4 scores are wrong -- silently."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=(14, 14, 16))
    g /= np.linalg.norm(g, axis=-1, keepdims=True)
    fg = np.ones((14, 14), bool)
    tail, head = np.array([0.15, 0.5]), np.array([0.85, 0.5])

    p_fwd, c_fwd = axis_profile(g, fg, tail, head, bins=5)
    p_rev, c_rev = axis_profile(g, fg, head, tail, bins=5)
    assert np.allclose(p_fwd, p_rev[::-1], atol=1e-6)
    assert np.array_equal(c_fwd, c_rev[::-1])


def test_profile_resolves_a_gradient_along_the_body():
    """If the bins cannot distinguish rump from head, there is no head/tail cue to find."""
    gh = gw = 14
    D = 16
    g = np.zeros((gh, gw, D))
    for c in range(gw):                                 # a left->right anatomical gradient
        g[:, c, min(c * D // gw, D - 1)] = 1.0
    p, cnt = axis_profile(g, np.ones((gh, gw), bool),
                          np.array([0.15, 0.5]), np.array([0.85, 0.5]), bins=5)
    assert cnt.sum() > 0
    adj = [float(p[b] @ p[b + 1]) for b in range(4)]
    assert max(adj) < 0.9, f"profile bins are indistinguishable: {adj}"


def test_profile_is_empty_when_the_animal_is_exactly_end_on():
    """Head and tail project to the same point: the axis has no image extent, so there is no
    profile. It must return zero evidence rather than divide by zero."""
    g = np.zeros((14, 14, 8))
    g[..., 0] = 1.0
    p, cnt = axis_profile(g, np.ones((14, 14), bool),
                          np.array([0.5, 0.5]), np.array([0.5, 0.5]), bins=5)
    assert cnt.sum() == 0 and np.allclose(p, 0)


def test_opposite_slot_uses_face_ids_not_projected_distance():
    """Projected face centres collapse when the animal is end-on -- exactly when we most need the
    pairing to be right. So the opposite face comes from the face IDs, which cannot degenerate."""
    face_ids = np.array([2, 3, 4, 5])                   # 2<->3 and 4<->5 are the opposite pairs
    assert opposite_slot(face_ids, 0) == 1
    assert opposite_slot(face_ids, 1) == 0
    assert opposite_slot(face_ids, 2) == 3
    assert opposite_slot(face_ids, 3) == 2


def _smooth_template(D=32, B=5, seed=0):
    """A realistic anatomical gradient: head -> neck -> torso -> rump varies SMOOTHLY along the
    body. (Random per-bin vectors would be an unrealistically easy target.)"""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=D), rng.normal(size=D)
    T = np.stack([(1 - s) * a + s * b for s in np.linspace(0, 1, B)])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Tc = _centre(T)
    return T, Tc / np.linalg.norm(Tc)


def test_centring_kills_the_dc_component_that_broke_the_4_way():
    """THE 4-WAY FIX. Every patch on an animal looks like "animal", so an UNCENTRED score gave a
    flat, wrong-axis profile ~81% of the true axis' score -- any axis looked good, and the 4-way
    collapsed to 39% while the head/tail cue itself was 83.7%.

    After centring we score the GRADIENT: a flat profile must contribute exactly nothing.
    """
    T, Tc = _smooth_template()
    B = T.shape[0]
    cnt = np.full(B, 20)

    mean_animal = T.mean(0) / np.linalg.norm(T.mean(0))
    flat = np.tile(mean_animal, (B, 1))

    assert abs(_match(flat, cnt, Tc)) < 1e-9, "a flat profile still scores -- the DC is still there"

    # and the bug, reproduced: uncentred, the flat profile nearly matched the truth
    uncentred = lambda P: float((np.einsum("bd,bd->b", P, T) * cnt).sum() / cnt.sum())
    assert uncentred(flat) > 0.75 * uncentred(T), "the historical bug no longer reproduces"


def test_true_axis_beats_reversed_and_wrong_axis():
    """The three ways a hypothesis can be wrong, all of which must lose to the truth."""
    T, Tc = _smooth_template()
    B, D = T.shape
    cnt = np.full(B, 20)
    rng = np.random.default_rng(1)

    s_true = _match(T.copy(), cnt, Tc)
    s_rev = _match(T[::-1].copy(), cnt, Tc)              # head/tail swapped (the 180-deg error)

    a, b = rng.normal(size=D), rng.normal(size=D)
    sym = np.stack([a, b, a, b, a][:B])                  # wrong axis: symmetric flank -> flank
    sym /= np.linalg.norm(sym, axis=1, keepdims=True)
    s_sym = _match(sym, cnt, Tc)

    assert s_true > s_rev, "cannot tell head from tail"
    assert s_true > s_sym, "a wrong-axis profile beats the truth"
    assert s_rev < 0, "the reversed profile should ANTI-correlate with the template"


def test_profile_magnitude_is_kept_because_contrast_is_evidence():
    """The profile is deliberately NOT normalised to unit length. Its magnitude is its CONTRAST
    along the axis -- the true body axis has a strong head->rump gradient while a cross-body
    profile is nearly flat. Normalising would rescale that flat profile up into pure noise, which
    could then beat the truth by luck."""
    T, Tc = _smooth_template()
    B = T.shape[0]
    cnt = np.full(B, 20)

    strong = _match(T.copy(), cnt, Tc)
    weak = _match(0.05 * T, cnt, Tc)                     # same shape, one-twentieth the contrast
    assert strong > weak > 0, "contrast was normalised away; it is evidence and must be kept"


def test_empty_bins_contribute_exactly_nothing():
    """An occluded or off-frame slice of the animal must not vote -- AT ALL.

    The invariant: filling the empty bins with arbitrary garbage must not move the score by one
    bit. That is stronger than "it roughly ignores them", and it is the version that catches the
    real bug -- `_centre` used to subtract the mean over ALL bins including the empty ones (which
    are zero rows), so a missing rump would drag the centre toward the origin, let the DC creep
    back in through the gap, and change the score.
    """
    T, Tc = _smooth_template()
    B, D = T.shape
    rng = np.random.default_rng(2)

    cnt = np.array([10, 0, 10, 0, 10])                  # bins 1 and 3 are occluded
    prof = T.copy()
    prof[1] = 0.0                                       # axis_profile writes a zero row for these
    prof[3] = 0.0
    clean = _match(prof, cnt, Tc)

    noisy = prof.copy()
    noisy[1] = rng.normal(size=D) * 5                   # garbage where there is no evidence
    noisy[3] = rng.normal(size=D) * 5
    assert abs(_match(noisy, cnt, Tc) - clean) < 1e-9, "empty bins leaked into the score"

    assert np.isnan(_match(prof, np.zeros(B, dtype=int), Tc)), "no evidence at all must be NaN"


def test_load_npz_materialises_and_is_not_lazy():
    """THE BUG THAT COST HOURS, and it is silent -- it looks like ordinary array indexing.

    `np.load` on a savez_compressed file returns a LAZY NpzFile: every `z[key][i]` zlib-inflates the
    ENTIRE member before taking one element. CropSet.get() read four fields, so ONE crop cost FOUR
    full decompressions of a 259 MB array.

    MEASURED: z["jpeg"][i] = 4,113 ms per crop -> ~6.6 hours to score the held-out set. It is also
    why bf16, the batch size and every other optimisation changed nothing: the GPU was never the
    bottleneck, and my own benchmark missed it by calling batch() OUTSIDE the timing loop.

    So: load_npz must hand back a plain dict of materialised arrays, never an NpzFile.
    """
    import tempfile
    from pathlib import Path

    from tools.heading.cropset import load_npz

    p = Path(tempfile.mkdtemp()) / "t.npz"
    np.savez_compressed(p, a=np.arange(1000), b=np.zeros((10, 3)))

    d = load_npz(p)
    assert isinstance(d, dict), f"load_npz returned {type(d)} -- a lazy NpzFile is the bug"
    assert not isinstance(d, np.lib.npyio.NpzFile)
    assert all(isinstance(v, np.ndarray) for v in d.values())
    assert d["a"][7] == 7 and d["b"].shape == (10, 3)


def test_cropset_never_holds_a_lazy_npzfile():
    """Guards the same bug at the place it actually bit."""
    import io as _io
    import tempfile
    from pathlib import Path

    from PIL import Image

    from tools.heading.cropset import CropSet

    n = 3
    jp = []
    for _ in range(n):
        b = _io.BytesIO()
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(b, "JPEG")
        jp.append(b.getvalue())
    p = Path(tempfile.mkdtemp()) / "crops.npz"
    np.savez_compressed(
        p, jpeg=np.array(jp, dtype=object),
        y_face=np.zeros(n, np.int8), face_uv=np.zeros((n, 4, 2), np.float32),
        face_ids=np.tile(np.array([2, 3, 4, 5], np.int8), (n, 1)),
        geo_axis=np.zeros(n, np.int8), species=np.array(["zebra"] * n),
        video=np.array(["V"] * n), track=np.array(["t"] * n), frame=np.arange(n, dtype=np.int32),
    )
    cs = CropSet(p)
    assert isinstance(cs.d, dict), "CropSet is holding a LAZY NpzFile -- every get() re-inflates it"


def test_paper_figure_pca_basis_must_be_shared_across_the_track():
    """The bottom strip of the paper figure claims: THE SAME COLOUR IS THE SAME BODY PART, in every
    frame. That claim is only true if the PCA basis is fitted ONCE over the whole track.

    A per-frame PCA re-randomises the colour axes -- and their SIGNS -- every frame, so the head
    comes out a different colour each time and the panel is decorative rather than evidence.
    Measured here: a shared basis is ~4.6x more stable on the same body part.
    """
    from tools.heading.paper_fig import track_pca

    rng = np.random.default_rng(0)
    D = gh = gw = 14, 14, 14
    D, gh, gw = 32, 14, 14
    head = rng.normal(size=D); head /= np.linalg.norm(head)
    rump = rng.normal(size=D); rump /= np.linalg.norm(rump)

    grids, masks = [], []
    for f in range(6):                                  # one animal, drifting across the crop
        g = np.zeros((gh, gw, D))
        shift = f * 1.5
        for c in range(gw):
            t = np.clip((c - shift) / (gw - shift - 1e-6), 0, 1)
            g[:, c] = (1 - t) * rump + t * head + 0.03 * rng.normal(size=D)
        g /= np.linalg.norm(g, axis=-1, keepdims=True)
        grids.append(g)
        masks.append(np.ones((gh, gw), bool))

    def head_patch(g):                                  # follow the BODY PART, not a fixed pixel
        sim = g @ head
        return np.unravel_index(np.argmax(sim), sim.shape)

    shared = track_pca(grids, masks)
    cs, cp = [], []
    for g in grids:
        r, c = head_patch(g)
        cs.append(shared(g)[r, c])
        cp.append(track_pca([g], [np.ones((gh, gw), bool)])(g)[r, c])

    s_shared = np.stack(cs).std(0).mean()
    s_perframe = np.stack(cp).std(0).mean()
    assert s_shared < s_perframe / 2, (
        f"a shared basis ({s_shared:.3f}) is no more stable than a per-frame one "
        f"({s_perframe:.3f}) -- the figure's central claim does not hold")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"running {len(tests)} tests\n")
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
