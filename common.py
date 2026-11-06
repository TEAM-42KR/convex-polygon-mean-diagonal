"""Small shared helpers: Haar frames, PASS/FAIL bookkeeping, check records, tables."""
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKDIR = os.path.join(HERE, "out", "checks")
DATA = os.path.join(HERE, "out")


def haar_frames(rng, B, n):
    """B independent Haar-distributed n x 2 orthonormal frames, returned as the
    complex vectors z_k = Q_k1 + i Q_k2 (shape (B, n)).

    Q = G (G^T G)^(-1/2) with G an n x 2 standard Gaussian matrix (polar part),
    which is exactly Haar on the Stiefel manifold.  Then sum z_k^2 = 0 and
    sum |z_k|^2 = 2, so the edges e_k = z_k^2 / 2 close up with perimeter 1.
    """
    g = rng.standard_normal((B, n, 2))
    a = np.einsum("bi,bi->b", g[:, :, 0], g[:, :, 0])
    c = np.einsum("bi,bi->b", g[:, :, 1], g[:, :, 1])
    b = np.einsum("bi,bi->b", g[:, :, 0], g[:, :, 1])
    # sqrt of the SPD 2x2 matrix [[a,b],[b,c]]: (A + s I)/t, s = sqrt(det), t = sqrt(tr + 2s)
    s = np.sqrt(a * c - b * b)
    t = np.sqrt(a + c + 2 * s)
    ra, rb, rc = (a + s) / t, b / t, (c + s) / t
    det = ra * rc - rb * rb
    ia, ib, ic = rc / det, -rb / det, ra / det           # inverse square root
    q1 = g[:, :, 0] * ia[:, None] + g[:, :, 1] * ib[:, None]
    q2 = g[:, :, 0] * ib[:, None] + g[:, :, 1] * ic[:, None]
    return q1 + 1j * q2


def seg_cross(p1, p2, p3, p4):
    """Proper intersection test of segments [p1,p2] and [p3,p4] (complex arrays).
    Ties have probability zero under the continuous laws used here."""
    def orient(a, b, c):
        return np.sign(((b - a).conjugate() * (c - a)).imag)
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    return (o1 * o2 < 0) & (o3 * o4 < 0)


class Checker:
    def __init__(self, name):
        self.name = name
        self.t0 = time.time()
        self.checks = []
        self.info = {}

    def check(self, label, ok, **fields):
        ok = bool(ok)
        rec = {"check": label, "pass": ok}
        for k, v in fields.items():
            rec[k] = v if isinstance(v, (str, int, bool)) or v is None else float(v)
        self.checks.append(rec)
        extra = "  ".join("%s=%s" % (k, ("%.6g" % v) if isinstance(v, float) else v)
                          for k, v in rec.items() if k not in ("check", "pass"))
        print("%s  %s  %s" % ("PASS" if ok else "FAIL", label, extra), flush=True)
        return ok

    def finish(self):
        os.makedirs(CHECKDIR, exist_ok=True)
        npass = sum(c["pass"] for c in self.checks)
        out = {
            "script": self.name,
            "all_pass": npass == len(self.checks),
            "n_checks": len(self.checks),
            "n_pass": npass,
            "runtime_seconds": round(time.time() - self.t0, 2),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "checks": self.checks,
            "info": self.info,
        }
        try:
            import scipy, mpmath
            out["scipy"] = scipy.__version__
            out["mpmath"] = mpmath.__version__
        except ImportError:
            pass
        path = os.path.join(CHECKDIR, self.name + ".json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print("%s: %d/%d checks PASS, %.1f s, record %s" % (
            self.name, npass, len(self.checks), out["runtime_seconds"],
            os.path.relpath(path, HERE)), flush=True)
        return 0 if out["all_pass"] else 1


def write_table(name, header, rows):
    """Whitespace-separated table with one header line (pgfplots format)."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(v if isinstance(v, str) else
                             ("%d" % v if isinstance(v, (int, np.integer)) else "%.12e" % v)
                             for v in r) + "\n")
    print("wrote", os.path.relpath(path, HERE), flush=True)
    return path
