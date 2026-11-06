#!/usr/bin/env python3
"""Driver: M_n (called E_n here; n = 4..10) from the fixed-dimension representation (Section 5 of the paper).
The values in ../data/fixed_dim_values.json were computed with profile 'prod' on 64 cores; profile 'mini' runs
on one core in about five minutes (entry point ../fixed_dim_mini.py).

Stages (all deterministic):
  tables  : G_m(s) = H_m(x(s)) (m=3..8) and f_k(D) = (pi/k) H_k''(x(D)) sech D (k=3,4,5) on Chebyshev nodes in
            tau, s = D = exp(tau - e^-tau), tau in [-4.53, 5.02] (12 pieces); H from hcorner.py (flint balls).
  outer   : d(n,k), 2 <= k <= n/2, from outer.py with the Chebyshev interpolants; E_n = sum_{k=2}^{n-2} d(n,k)/(n-3).
  holdout : interpolant vs fresh H_all at off-node tau points.
Usage: prod.py tables|outer|holdout --profile prod|test --workers W --dir OUTDIR
"""
import os, sys, json, time, argparse, multiprocessing as mpc
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mpmath as mp

MS = [3, 4, 5, 6, 7, 8]; KS = [3, 4, 5]
PAIRS = [(n, k) for n in range(4, 11) for k in range(2, n // 2 + 1)]
PROFILES = {
  'prod': dict(prec=280, dps=85, nfac=1.0,
               pieces=[(-4.53, -3.85, 72, 'B'), (-3.85, -3.2, 72, 'C'), (-3.2, -2.55, 72, 'C'), (-2.55, -1.9, 72, 'C'),
                       (-1.9, -1.2, 88, 'C'), (-1.2, -0.5, 88, 'C'), (-0.5, 0.2, 88, 'C'), (0.2, 0.95, 88, 'C'),
                       (0.95, 1.75, 88, 'C'), (1.75, 2.65, 96, 'C'), (2.65, 3.75, 100, 'C'), (3.75, 5.02, 104, 'C')],
               grades={'B': dict(hr=0.075, hphi=1/26., hs=1/26., ny=48, na1=64, nats=165, tsmax=4.7),
                       'C': dict(hr=0.055, hphi=1/32., hs=1/32., ny=48, na1=64, nats=165, tsmax=4.7)},
               outer=dict(hr=1/30., hvfac=1, hsig=1/36., hlog=0.047, nats=165, rsplit=2, tsmax=4.8,
                          tlo2=-5.1, thi2=5.9, tlo3=-4.5, thi3=4.5)),
  'test': dict(prec=128, dps=34, nfac=0.5,
               pieces=[(-4.53, -3.85, 72, 'T'), (-3.85, -3.2, 72, 'T'), (-3.2, -2.55, 72, 'T'), (-2.55, -1.9, 72, 'T'),
                       (-1.9, -1.2, 88, 'T'), (-1.2, -0.5, 88, 'T'), (-0.5, 0.2, 88, 'T'), (0.2, 0.95, 88, 'T'),
                       (0.95, 1.75, 88, 'T'), (1.75, 2.65, 96, 'T'), (2.65, 3.75, 100, 'T'), (3.75, 5.02, 104, 'T')],
               grades={'T': dict(hr=0.14, hphi=1/12., hs=1/12., ny=24, na1=32, nats=75, tsmax=4.0)},
               outer=dict(hr=1/12., hvfac=1, hsig=1/12., hlog=0.13, nats=75, rsplit=2, tsmax=4.0,
                          tlo2=-4.3, thi2=5.0, tlo3=-3.6, thi3=3.7)),
}
PROFILES['mini'] = dict(PROFILES['test'], prec=100, dps=26, nfac=0.3,
                        pieces=[(a, b, N, 'M') for (a, b, N, g) in PROFILES['test']['pieces']],
                        grades={'M': dict(hr=0.2, hphi=1/8., hs=1/8., ny=16, na1=20, nats=55, tsmax=3.6)},
                        outer=dict(hr=1/8., hvfac=1, hsig=1/8., hlog=0.18, nats=55, rsplit=2, tsmax=3.6,
                                   tlo2=-4.0, thi2=4.7, tlo3=-3.3, thi3=3.4))

def cheb_nodes(a, b, N):
    """first-kind Chebyshev nodes in tau (mp), barycentric weights."""
    out = []
    for j in range(N):
        th = (2 * j + 1) * mp.pi / (2 * N)
        out.append(((a + b) / 2 + (b - a) / 2 * mp.cos(th), (-1) ** j * mp.sin(th), mp.cos(th)))
    return out

def node_list(P):
    mp.mp.dps = P['dps']
    L = []
    for ip, (a, b, N, g) in enumerate(P['pieces']):
        N = max(8, int(round(N * P['nfac'])))
        for j, (tau, w, xn) in enumerate(cheb_nodes(mp.mpf(a), mp.mpf(b), N)):
            L.append((ip, j, mp.nstr(tau, P['dps'] + 5), g))
    return L

_P = None
def _init(pname):
    global _P
    _P = PROFILES[pname]
    from flint import ctx
    ctx.prec = _P['prec']

def table_task(args):
    ip, j, tau_s, g = args
    from flint import arb
    import hcorner as hc
    t0 = time.time()
    tau = arb(tau_s)
    s = (tau - (-tau).exp()).exp()
    x, eps = hc.x_eps_of_s(s)
    out = hc.H_all(x, eps, MS, _P['grades'][g])
    sech = 1 / s.cosh()
    dg = _P['dps'] + 5
    G = {m: out[m][0].mid().str(dg, radius=False) for m in MS}
    F = {k: (arb.pi() / k * out[k][2] * sech).mid().str(dg, radius=False) for k in KS}
    rad = max(float(out[m][0].rad()) for m in MS), max(float(out[k][2].rad()) for k in KS)
    return dict(ip=ip, j=j, tau=tau_s, G=G, F=F, rad=rad, sec=round(time.time() - t0, 2))

def run_tables(pname, W, odir):
    P = PROFILES[pname]
    L = node_list(P)
    L.sort(key=lambda z: float(z[2]))          # small s (expensive) first
    res = []
    t0 = time.time()
    with mpc.Pool(W, initializer=_init, initargs=(pname,)) as pool:
        for i, r in enumerate(pool.imap_unordered(table_task, L, chunksize=1)):
            res.append(r)
            if i % 50 == 0:
                print('tables', i, len(L), round(time.time() - t0, 1), flush=True)
    json.dump(dict(profile=pname, pieces=P['pieces'], nfac=P['nfac'], nodes=res), open(os.path.join(odir, 'tables.json'), 'w'))
    print('tables done', len(res), round(time.time() - t0, 1), 'max sec', max(r['sec'] for r in res), flush=True)

class Cheb:
    def __init__(self, tabfile, dps):
        mp.mp.dps = dps
        T = json.load(open(tabfile))
        pcs = T['pieces']; nfac = T['nfac']
        self.pieces = []
        byp = {}
        for r in T['nodes']:
            byp.setdefault(r['ip'], {})[r['j']] = r
        for ip, (a, b, N, g) in enumerate(pcs):
            N = max(8, int(round(N * nfac)))
            nodes = cheb_nodes(mp.mpf(a), mp.mpf(b), N)
            G = {m: [mp.mpf(byp[ip][j]['G'][str(m)]) for j in range(N)] for m in MS}
            F = {k: [mp.mpf(byp[ip][j]['F'][str(k)]) for j in range(N)] for k in KS}
            self.pieces.append((mp.mpf(a), mp.mpf(b), [w for (_, w, _) in nodes], [xn for (_, _, xn) in nodes], G, F))
        self.tmin = self.pieces[0][0]; self.tmax = self.pieces[-1][1]
        self.smin = mp.exp(self.tmin - mp.exp(-self.tmin)); self.smax = mp.exp(self.tmax - mp.exp(-self.tmax))
        # log-slope of f at the left end (f = A + B log D + O(D log D))
        t1 = self.tmin + mp.mpf('0.01')
        f0 = self._eval(self.tmin, 'F'); f1 = self._eval(t1, 'F')
        dl = (t1 - mp.exp(-t1)) - (self.tmin - mp.exp(-self.tmin))
        self.fslope = {k: (f1[k] - f0[k]) / dl for k in KS}
        self.f0 = f0; self.G0 = self._eval(self.tmin, 'G'); self.fend = self._eval(self.tmax, 'F')

    @staticmethod
    def tau_of(s):
        L = mp.log(s)
        tau = L if L >= 0 else (-mp.log(-L) - 1 if L < -1 else mp.mpf(0))
        for _ in range(200):
            et = mp.exp(-tau)
            d = (tau - et - L) / (1 + et)
            tau -= d
            if abs(d) < mp.mpf(10) ** (-mp.mp.dps + 3): break
        return tau

    def _eval(self, tau, which):
        for (a, b, ws, xs, G, F) in self.pieces:
            if tau <= b or b == self.tmax:
                break
        x = (2 * tau - a - b) / (b - a)
        tab = G if which == 'G' else F
        c = []
        for (w, xn) in zip(ws, xs):
            dx = x - xn
            if dx == 0:
                j = xs.index(xn)
                return {key: vals[j] for key, vals in tab.items()}
            c.append(w / dx)
        den = mp.fsum(c)
        return {key: mp.fdot(c, vals) / den for key, vals in tab.items()}

    def Gvec(self, s):
        x = mp.atan(1 / mp.sinh(s))
        out = {2: 1 - 2 * x / mp.pi}
        if s >= self.smax:
            out.update({m: 1 - m * x / mp.pi for m in MS})
        elif s <= self.smin:
            out.update(self.G0)
        else:
            out.update(self._eval(self.tau_of(s), 'G'))
        return out

    def fvec(self, D):
        if D >= self.smax:
            return {k: self.fend[k] * mp.cosh(self.smax) / mp.cosh(D) for k in KS}
        if D <= self.smin:
            dl = mp.log(D) - mp.log(self.smin)
            return {k: self.f0[k] + self.fslope[k] * dl for k in KS}
        return self._eval(self.tau_of(D), 'F')

_C = None
def _init_outer(pname, tabfile):
    global _C, _P
    _P = PROFILES[pname]
    _C = Cheb(tabfile, _P['dps'])

def outer_task(args):
    import outer as ou
    i, W = args
    mp.mp.dps = _P['dps']
    t0 = time.time()
    d = ou.d_all(PAIRS, _C, _P['outer'], part=(i, W))
    return dict(i=i, d={'%d,%d' % p: mp.nstr(v, _P['dps']) for p, v in d.items()}, sec=round(time.time() - t0, 1))

def run_outer(pname, W, odir):
    P = PROFILES[pname]
    mp.mp.dps = P['dps']
    tabfile = os.path.join(odir, 'tables.json')
    NP = W * 4
    t0 = time.time()
    tot = {p: mp.mpf(0) for p in PAIRS}
    parts = []
    with mpc.Pool(W, initializer=_init_outer, initargs=(pname, tabfile)) as pool:
        for r in pool.imap_unordered(outer_task, [(i, NP) for i in range(NP)], chunksize=1):
            parts.append(r)
            for key, v in r['d'].items():
                n, k = map(int, key.split(','))
                tot[(n, k)] += mp.mpf(v)
    E = {}
    for n in range(4, 11):
        S = sum((2 if 2 * k < n else 1) * tot[(n, k)] for k in range(2, n // 2 + 1))
        E[n] = S / (n - 3)
    out = dict(profile=pname, d={'%d,%d' % p: mp.nstr(v, P['dps']) for p, v in tot.items()},
               E={str(n): mp.nstr(v, P['dps']) for n, v in E.items()}, wall=round(time.time() - t0, 1),
               part_secs=sorted(r['sec'] for r in parts)[-3:])
    json.dump(out, open(os.path.join(odir, 'outer.json'), 'w'), indent=1)
    print('outer done', out['wall'], flush=True)

def holdout_task(args):
    tau_s, g = args
    r = table_task((0, 0, tau_s, g))
    return r

def run_holdout(pname, W, odir):
    P = PROFILES[pname]
    mp.mp.dps = P['dps']
    C = Cheb(os.path.join(odir, 'tables.json'), P['dps'])
    pts = []
    for ip, (a, b, N, g) in enumerate(P['pieces']):
        for frac in ('0.137', '0.618'):
            pts.append((mp.nstr(mp.mpf(a) + (mp.mpf(b) - mp.mpf(a)) * mp.mpf(frac), P['dps'] + 5), g))
    with mpc.Pool(W, initializer=_init, initargs=(pname,)) as pool:
        res = pool.map(holdout_task, pts, chunksize=1)
    rows = []
    for r in res:
        tau = mp.mpf(r['tau'])
        Gi = C._eval(tau, 'G'); Fi = C._eval(tau, 'F')
        eG = max(abs(Gi[m] - mp.mpf(r['G'][m])) for m in MS)
        eF = max(abs(Fi[k] - mp.mpf(r['F'][k])) / (1 + abs(mp.mpf(r['F'][k]))) for k in KS)
        rows.append((r['tau'][:10], mp.nstr(eG, 3), mp.nstr(eF, 3)))
    json.dump(rows, open(os.path.join(odir, 'holdout.json'), 'w'), indent=1)
    print('holdout', rows, flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage'); ap.add_argument('--profile', default='test')
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--dir', default='.')
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    {'tables': run_tables, 'outer': run_outer, 'holdout': run_holdout}[a.stage](a.profile, a.workers, a.dir)
