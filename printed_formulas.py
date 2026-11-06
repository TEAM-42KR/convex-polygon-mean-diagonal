#!/usr/bin/env python3
"""Check of the closed forms as PRINTED in the paper.

Every formula below was retyped by hand from the LaTeX source of the paper
(section, label and display noted next to it); nothing is read from the paper
files at run time.  The formulas are evaluated at 140 working digits and
compared with independent reference values; the number of agreeing digits
-log10(|printed - reference| / |reference|) is reported for every check.

Constants: pi, log 2, zeta(3), zeta(5), zeta(7), Li4(1/2), Li5(1/2) from
mpmath; the alternating multiple zeta values from PARI/GP (`gp` must be on
PATH), with the convention of the appendix
    zeta(s_1^e_1, ..., s_r^e_r) = sum_{n_1 > ... > n_r >= 1} prod e_j^(n_j) / n_j^(s_j)
              = polylogmult([s_1..s_r], [e_1..e_r])  in GP,
and zeta*(5,-1,1,1) = the 8-term merged sum printed in the appendix (checked
against GP's star value polylogmult(.., .., 1)).

Groups of checks:
 (a) M_8 from display (eq:M8)                        vs 98-digit reference, >= 95 digits
 (b) d_{8,2}, d_{8,3} (Identification id:d8), d_{8,4} (eq:d84);
     (2 d82 + 2 d83 + d84)/5 vs the M_8 reference and vs display (eq:M8)
 (c) M_4..M_7 (Theorem thm:small, and thm:M4, thm:M5), d_{6,2}, d_{6,3},
     d_{7,2}, d_{7,3}; M_6 = (2 d62 + d63)/3, M_7 = (d72 + d73)/2      >= 50 digits
 (d) c_1, c_2, c_3 (introduction and Formal expansion fx:c23)        >= 18 digits
 (e) P(simple pentagon) (Theorem thm:simple5) and the mean diagonal of a
     simple quadrilateral (remark after prop:trichotomy)
plus the printed decimal values (each must agree with the evaluated formula
to all printed digits, truncated or rounded).
"""
import os
import subprocess
import sys

import mpmath as mp
from mpmath import mpf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker

DPS = 140
mp.mp.dps = DPS
F = mp.mpf

# ------------------------------------------------------------------ references
REF = {
    # given to 98 digits
    "M8": "0.27172589367425682824039784172584711046824201640089970039115095267607042085057625635881167115242393",
    "d82": "0.2271363414020088984428194938755787383047365332849458013822232593264469811244625601420382245748234324",
    # 55-digit reference values (same as sorted_sampler.py)
    "d83": "0.2937483582774511231767249981447828013775279273363596966",
    "d84": "0.3168600690123640979629002245885124729766811607618875059",
    "M4": "0.3453051847481618355032977789205408860013184513634833978",
    "M5": "0.3154621014556979611662880552884315465420673744805453633",
    "M6": "0.2958556888751911490912019732607904837860657228457401777",
    "M7": "0.2820120392645604931934154596208657878305703644530818003",
    "d62": "0.2821387234598579905052671919829763000772416858404397297",
    "d63": "0.3232896197058574662630715358164188512037137968563410737",
    "d72": "0.2524038522839323627263822968095553359392710750397520086",
    "d73": "0.3116202262451886236604486224321762397218696538664115919",
    # asymptotic coefficients (c1: the integral of asymptotic_c1.py at 52 digits)
    "c1": "0.5275995824598435623674008898058945119623655443018817",
    "c2": "0.24171353884950172649636",
    "c3": "-0.41644512648653075829",
    # simple_pentagon.py: J quadrature at 50 digits -> 1/2 - 15 J/(4 pi^2)
    "Psimple5": "0.4808734052264137374579030175067118501561",
    "Psimple5_mc": "0.480535",
    "quad_simple": "0.3123310",
}


def ref(key):
    return mpf(REF[key])


def digits(x, y):
    d = abs(x - y)
    if d == 0:
        return float(DPS)
    return float(-mp.log10(d / abs(y)))


# ------------------------------------------------------------------ constants
GP_LINES = [
    "default(realprecision, 150);",
    "print(polylogmult([5,1],[-1,-1]));",                      # zeta(-5,-1)
    "print(polylogmult([5,1,1],[1,1,-1]));",                   # zeta(5,1,-1)
    "print(polylogmult([5,1,1],[-1,-1,1]));",                  # zeta(-5,-1,1)
    "print(zetamult([5,3]));",                                 # zeta(5,3)
    "print(polylogmult([7,1],[-1,-1]));",                      # zeta(-7,-1)
    # zeta*(5,-1,1,1) as the printed 8-term sum
    "print(polylogmult([5,1,1,1],[1,-1,1,1]) + polylogmult([6,1,1],[-1,1,1]) + polylogmult([5,2,1],[1,-1,1])"
    " + polylogmult([5,1,2],[1,-1,1]) + polylogmult([7,1],[-1,1]) + polylogmult([6,2],[-1,1])"
    " + polylogmult([5,3],[1,-1]) + polylogmult([8],[-1]));",
    "print(polylogmult([5,1,1,1],[1,-1,1,1],1));",             # GP star value (t = 1)
    "print(zeta(3)); print(polylog(4,1/2)); print(polylog(5,1/2));",
]
GP_SCRIPT = "\n".join(GP_LINES) + "\n"


def gp_constants():
    out = subprocess.run(["gp", "-q", "-f", "--default", "parisize=256000000"], input=GP_SCRIPT,
                         capture_output=True, text=True, timeout=600, check=True).stdout.split()
    vals = [mpf(s.replace("E", "e")) for s in out]
    if len(vals) != 10:
        raise RuntimeError("unexpected gp output: %r" % out)
    names = ["z5b1b", "z511b", "z5b1b1", "z53", "z7b1b", "zstar_sum", "zstar_gp", "gp_z3", "gp_li4", "gp_li5"]
    return dict(zip(names, vals))


def main():
    ck = Checker("printed_formulas")
    g = gp_constants()
    pi = mp.pi
    L = mp.log(2)
    z3, z5, z7 = mp.zeta(3), mp.zeta(5), mp.zeta(7)
    Li4 = mp.polylog(4, F(1) / 2)
    Li5 = mp.polylog(5, F(1) / 2)
    z5b1b, z511b, z5b1b1, z53, z7b1b = g["z5b1b"], g["z511b"], g["z5b1b1"], g["z53"], g["z7b1b"]
    zstar = g["zstar_sum"]
    R = lambda p, q: F(p) / q

    # --- constants: GP vs mpmath, star value, appendix table (printed 25 digits)
    dc = min(digits(g["gp_z3"], z3), digits(g["gp_li4"], Li4), digits(g["gp_li5"], Li5))
    ck.check("constants: GP vs mpmath (zeta(3), Li4(1/2), Li5(1/2))", dc >= 130, digits=dc, need=130)
    ds = digits(zstar, g["zstar_gp"])
    ck.check("app. A: 8-term merged sum = zeta*(5,-1,1,1) (GP star)", ds >= 130, digits=ds, need=130)
    # printed values of the appendix table, digit groups joined
    table = [("zeta(-5,-1)", z5b1b, "-0.0299016355811494259428679"),
             ("zeta(5,1,-1)", z511b, "-0.0032883569931046385727396"),
             ("zeta(-5,-1,1)", z5b1b1, "-0.0022309760560048714317274"),
             ("zeta(5,3)", z53, "0.0377076729848475440113048"),
             ("zeta(-7,-1)", z7b1b, "-0.0076296078001066733978123"),
             ("zeta*(5,-1,1,1)", zstar, "-1.0082370471594965009190383")]
    for name, val, printed in table:
        nd = len(printed.split(".")[1])
        ok = abs(val - mpf(printed)) <= F(10) ** (-nd)
        ck.check("app. A table value %s printed to %d decimals" % (name, nd), ok,
                 printed=printed, value=mp.nstr(val, nd + 3))

    # ---------------- (b) components, retyped from cases.tex (id:d8) and appendix.tex (eq:d84)
    d82 = (R(937, 32768) * pi ** 2 + R(2275, 576) / pi ** 2
           - (R(110075, 1536) - R(49525, 1536) * z3 + R(55125, 2048) * z3 ** 2) / pi ** 4
           + (R(55125, 128) - R(110425, 512) * z3 - R(49525, 512) * z5 + R(165375, 1024) * z3 * z5
              - R(80325, 2048) * z53) / pi ** 6)
    d83 = (R(10435, 12288) + R(425, 896) * L
           - (R(62545, 4096) + R(14825, 1024) * z3) / pi ** 2
           + (R(64575, 256) + R(93375, 512) * z5) / pi ** 4
           + (-R(55125, 32) + R(218925, 512) * z3 - R(385875, 2048) * z3 ** 2 - R(5625, 32) * L * z3 ** 2
              - R(736875, 2048) * z7 + R(1125, 8) * (L * z5b1b + z511b + z5b1b1)) / pi ** 6)
    d84 = (-R(70161, 139264) * pi ** 2
           - R(1135, 4096) - R(78525, 34816) * L ** 2
           + (R(142975, 6144) - R(48825, 2048) * L + R(525, 512) * L ** 2
              + R(172305, 69632) * L ** 4 + R(424305, 8704) * L * z3 + R(460215, 8704) * Li4) / pi ** 2
           + (-R(91875, 256) + R(363475, 2048) * z3 + R(525, 256) * L ** 4
              + R(1575, 32) * Li4 - R(4725, 8704) * L ** 6 - R(439425, 34816) * L * z5
              - R(6459075, 34816) * z3 ** 2 + R(14175, 4352) * L ** 3 * z3 - R(14175, 1088) * L ** 2 * Li4
              - R(42525, 4352) * z5b1b) / pi ** 4
           + (R(165375, 64) - R(55125, 128) * z3 - R(4801125, 4096) * z5
              - R(1575, 128) * z3 ** 2 + R(4725, 32) * z5b1b + R(4725, 17408) * L ** 8
              - R(7186725, 34816) * L ** 2 * z3 ** 2 - R(127575, 1088) * L * z3 * Li4
              - R(43843275, 69632) * L * z7
              - R(8505, 2176) * L ** 5 * z3 + R(301573125, 139264) * z3 * z5 + R(42525, 272) * Li4 ** 2
              + R(14175, 1088) * L ** 4 * Li4 - R(439425, 17408) * L ** 3 * z5 - R(127575, 1088) * z3 * Li5
              + R(127575, 1088) * L ** 2 * z5b1b + R(42525, 272) * L * z511b + R(42525, 544) * L * z5b1b1
              + R(13395375, 34816) * z53 + R(297675, 544) * z7b1b - R(42525, 544) * zstar) / pi ** 6)

    # ---------------- (a) M8 display (eq:M8), retyped from cases.tex
    K = L * z5b1b + z511b + z5b1b1
    W = (-pi ** 2 * z5b1b + 12 * L ** 2 * z5b1b + 16 * L * z511b + 8 * L * z5b1b1
         + R(2257, 72) * z53 + 56 * z7b1b - 8 * zstar)
    C8 = (-R(175, 128) * z3 - R(1118705, 4096) * z5 - R(79695, 1024) * z3 ** 2 - R(1125, 16) * L * z3 ** 2
          - R(147375, 1024) * z7
          + R(945, 17408) * L ** 8 - R(1437345, 34816) * L ** 2 * z3 ** 2 - R(25515, 1088) * L * z3 * Li4
          - R(8768655, 69632) * L * z7
          - R(1701, 2176) * L ** 5 * z3 + R(69311025, 139264) * z3 * z5 + R(8505, 272) * Li4 ** 2
          + R(2835, 1088) * L ** 4 * Li4
          - R(87885, 17408) * L ** 3 * z5 - R(25515, 1088) * z3 * Li5)
    M8 = (-R(124393, 1392640) * pi ** 2 + R(3493, 12288) + R(85, 448) * L - R(15705, 34816) * L ** 2
          + (R(581, 4608) - R(9765, 2048) * L + R(105, 512) * L ** 2 - R(2965, 512) * z3
             + R(34461, 69632) * L ** 4 + R(84861, 8704) * L * z3 + R(92043, 8704) * Li4) / pi ** 2
          + (R(175, 384) + R(297325, 6144) * z3 + R(105, 256) * L ** 4 + R(315, 32) * Li4 + R(18675, 256) * z5
             - R(945, 8704) * L ** 6 - R(87885, 34816) * L * z5
             - R(1666665, 34816) * z3 ** 2
             + R(2835, 4352) * L ** 3 * z3 - R(2835, 1088) * L ** 2 * Li4) / pi ** 4
          + (C8 + R(945, 32) * z5b1b + R(225, 4) * K + R(8505, 4352) * W) / pi ** 6)

    def cmp(label, x, key_or_val, need, **kw):
        y = ref(key_or_val) if isinstance(key_or_val, str) else key_or_val
        d = digits(x, y)
        ck.check(label, d >= need, digits=round(d, 1), need=need, value=mp.nstr(x, 30), **kw)
        return d

    cmp("(a) M8 display (eq:M8) vs M8 reference", M8, "M8", 95)
    cmp("(a) M8 printed decimal 0.27172...04682 (40 dec.)", M8, mpf("0.2717258936742568282403978417258471104682"), 39.3)
    cmp("(b) d82 (id:d8) vs reference", d82, "d82", 95)
    cmp("(b) d83 (id:d8) vs 55-digit reference", d83, "d83", 50)
    cmp("(b) d84 (eq:d84) vs 55-digit reference", d84, "d84", 50)
    comb8 = (2 * d82 + 2 * d83 + d84) / 5
    cmp("(b) (2 d82 + 2 d83 + d84)/5 vs M8 reference", comb8, "M8", 95)
    cmp("(b) (2 d82 + 2 d83 + d84)/5 vs display (eq:M8)", comb8, M8, 130)
    # componentwise: the value each component must have for the M8 reference to hold
    d83_impl = (5 * ref("M8") - 2 * d82 - d84) / 2
    d84_impl = 5 * ref("M8") - 2 * d82 - 2 * d83
    d82_impl = (5 * ref("M8") - 2 * d83 - d84) / 2
    cmp("(b) d82 vs (5 M8ref - 2 d83 - d84)/2", d82, d82_impl, 95)
    cmp("(b) d83 vs (5 M8ref - 2 d82 - d84)/2", d83, d83_impl, 95)
    cmp("(b) d84 vs 5 M8ref - 2 d82 - 2 d83", d84, d84_impl, 95)
    # the cancellation remark after (eq:M8): rational term at pi^-6
    canc = 2 * (R(55125, 128) - R(55125, 32)) + R(165375, 64)
    ck.check("(b) remark: rational pi^-6 terms cancel (-165375/64 + 165375/64)", canc == 0,
             value=mp.nstr(canc, 5))

    # ---------------- (c) small n, retyped from intro.tex (thm:small) and cases.tex
    M4 = R(3, 128) * pi ** 2 + R(9, 8) / pi ** 2
    M5 = 32 * (2 + L) / (15 * pi ** 2) - R(4, 15)
    M6 = (R(5, 448) * pi ** 2 + R(419, 4608) + 15 * L / 128 + R(95, 192) / pi ** 2 - 105 * z3 / (128 * pi ** 2)
          - 105 * z3 ** 2 / (32 * pi ** 4) + 35 * z3 / (8 * pi ** 4) + 1395 * z5 / (256 * pi ** 4))
    M7 = (-R(22, 105) + (68 - 368 * L - 2160 * L ** 2) / (315 * pi ** 2)
          + (51840 * Li4 + 2160 * L ** 4 + 1728 * L ** 2) / (315 * pi ** 4)
          - (25728 * L + 17424 * z3 - 45360 * L * z3) / (315 * pi ** 4))
    M4b = R(3, 128) * pi ** 2 + R(9, 8) / pi ** 2                    # thm:M4 (cases.tex)
    M5b = 32 * (2 + L) / (15 * pi ** 2) - R(4, 15)                   # thm:M5 (cases.tex)
    d62 = (15 * pi ** 2 / 896 + R(5, 2) / pi ** 2 - R(225, 16) / pi ** 4 + 105 * z3 / (16 * pi ** 4)
           - 315 * z3 ** 2 / (64 * pi ** 4))
    d63 = (R(419, 1536) + 45 * L / 128 - R(225, 64) / pi ** 2 - 315 * z3 / (128 * pi ** 2)
           + R(225, 8) / pi ** 4 + 4185 * z5 / (256 * pi ** 4))
    d72 = R(64, 105) + (2296 + 1568 * L) / (315 * pi ** 2) - (9216 - 128 * L + 4704 * z3) / (105 * pi ** 4)
    d73 = (-R(36, 35) - (240 + 256 * L + 480 * L ** 2) / (35 * pi ** 2)
           + (11520 * Li4 + 480 * L ** 4 + 384 * L ** 2 + 10080 * L * z3 - 5760 * L - 2304 * z3 + 3072)
           / (35 * pi ** 4))
    for lab, x, key in [("M4 (thm:small)", M4, "M4"), ("M4 (thm:M4)", M4b, "M4"),
                        ("M5 (thm:small)", M5, "M5"), ("M5 (thm:M5)", M5b, "M5"),
                        ("M6 (thm:small)", M6, "M6"), ("M7 (thm:small)", M7, "M7"),
                        ("d62 (ssec n=6)", d62, "d62"), ("d63 (ssec n=6)", d63, "d63"),
                        ("d72 (ssec n=7)", d72, "d72"), ("d73 (ssec n=7)", d73, "d73")]:
        cmp("(c) %s vs reference" % lab, x, key, 50)
    cmp("(c) M6 = (2 d62 + d63)/3 (printed forms)", (2 * d62 + d63) / 3, M6, 130)
    cmp("(c) M7 = (d72 + d73)/2 (printed forms)", (d72 + d73) / 2, M7, 130)

    # ---------------- (d) asymptotic coefficients
    c1 = (1 + R(7, 2) * z3) / pi ** 2                                                  # intro.tex
    c2_i = -R(3, 8) - R(7, 16) * z3 + (84 + 93 * z5) / (16 * pi ** 2)                  # intro.tex
    c3_i = (-R(71, 32) + R(63, 4) * z3 - R(2511, 128) * z5
            + (R(123, 8) + R(21, 2) * z3 - R(3069, 16) * z5 + R(51435, 256) * z7) / pi ** 2)
    c2_a = -R(3, 8) - R(7, 16) * z3 + (84 + 93 * z5) / (16 * pi ** 2)                  # fx:c23
    c3_a = (-R(71, 32) + R(63, 4) * z3 - R(2511, 128) * z5
            + (R(123, 8) + R(21, 2) * z3 - R(3069, 16) * z5 + R(51435, 256) * z7) / pi ** 2)
    cmp("(d) c1 = (1 + 7 zeta(3)/2)/pi^2 vs c1 integral", c1, "c1", 50)
    cmp("(d) c2 (intro) vs reference", c2_i, "c2", 18)
    cmp("(d) c3 (intro) vs reference", c3_i, "c3", 18)
    cmp("(d) c2 (fx:c23) vs reference", c2_a, "c2", 18)
    cmp("(d) c3 (fx:c23) vs reference", c3_a, "c3", 18)
    for lab, x, printed in [("c1 0.52760", c1, "0.52760"), ("c2 0.24171", c2_i, "0.24171"),
                            ("c3 -0.41645", c3_i, "-0.41645"),
                            ("c2 0.24171 35388 49501 72649 636", c2_a, "0.24171353884950172649636"),
                            ("c3 -0.41644 51264 86530 75829", c3_a, "-0.41644512648653075829")]:
        nd = len(printed.split(".")[1])
        ok = abs(x - mpf(printed)) <= F(10) ** (-nd)
        ck.check("(d) printed decimal %s" % lab, ok, value=mp.nstr(x, nd + 3), tol="1e-%d" % nd)

    # ---------------- (e) simple pentagon, simple quadrilateral (cases.tex)
    P5 = R(7, 24) + (10 - 10 * L - R(5, 2) * L ** 2) / pi ** 2
    J = (pi ** 2 + 48 * L + 12 * L ** 2 - 48) / 18
    cmp("(e) P(simple pentagon) (thm:simple5) vs 1/2 - 15J/(4pi^2) from quadrature", P5, "Psimple5", 39)
    cmp("(e) thm:simple5 proof: 1/2 - (5/12)(9/pi^2) J = printed probability", R(1, 2) - R(5, 12) * 9 / pi ** 2 * J,
        P5, 130)
    zmc = (float(ref("Psimple5_mc")) - float(P5)) / 0.000499621
    ck.check("(e) P(simple pentagon) vs Monte Carlo of simple_pentagon.py", abs(zmc) < 4, z=zmc, tol="|z|<4")
    ck.check("(e) printed decimal P(simple) 0.48087", abs(P5 - mpf("0.48087")) <= F(10) ** -5,
             value=mp.nstr(P5, 8))
    Q = (R(241, 3840) * pi ** 2 + L ** 2 / 8
         - (24 * Li4 + 21 * z3 * L + L ** 4 + 3 - 6 * L) / (8 * pi ** 2))
    dq = abs(Q - ref("quad_simple"))
    ck.check("(e) simple-quadrilateral mean (remark) vs 0.3123310 (7 digits)", dq < F(10) ** -7,
             value=mp.nstr(Q, 12), abs_diff=float(dq), tol="1e-7")
    ck.check("(e) printed decimal 0.31233", abs(Q - mpf("0.31233")) <= F(10) ** -5, value=mp.nstr(Q, 8))

    ck.info["values_40_digits"] = {k: mp.nstr(v, 40) for k, v in dict(
        M8=M8, d82=d82, d83=d83, d84=d84, M4=M4, M5=M5, M6=M6, M7=M7, d62=d62, d63=d63, d72=d72,
        d73=d73, c1=c1, c2=c2_a, c3=c3_a, P_simple5=P5, quad_simple=Q).items()}
    ck.info["working_digits"] = DPS
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
