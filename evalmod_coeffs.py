#!/usr/bin/env python3
"""Generate the EvalMod polynomial coefficients used by BootParameter.

EvalMod adds -0.25/actual_K to its input, evaluates one polynomial p(x) in the
Chebyshev basis on [-1, 1], then applies num_double_angle steps of
y <- 2*y^2 - c. With actual_K = 2^num_double_angle * initial_K, the result is
(1/2pi) * sin(2pi * actual_K * t) when

    p(x) ~= (0.5/pi)^(1/2^num_double_angle) * cos(2pi * initial_K * x).

p only has to be accurate where its input lands: x = (I + m - 0.25)/actual_K,
where I is the rounding integer of the bootstrap mod-up and
|m| <= 2^-log_message_ratio. The range of I grows with the Hamming weight of
the bootstrap secret.

Methods:

  discrete    Han-Ki interpolation (Han and Ki, "Better Bootstrapping for
              Approximate Homomorphic Encryption", ePrint 2019/688), as in
              Lattigo's CosDiscrete (utils/cosine/cosine_approx.go). Nodes sit
              within 2^-log_message_ratio of every integer |I| < actual_K, so
              it needs at least 2*actual_K - 1 coefficients.
  continuous  Even least-squares fit with Chebyshev weight on |x| <= D. This is
              Lattigo's CosContinuous restricted to the inputs that occur, and
              needs no extra degree when actual_K grows.

The presets in src/extension/BootParameter.cpp are reproduced by:

  python3 evalmod_coeffs.py discrete --num-double-angle 3 --degree 30
  python3 evalmod_coeffs.py continuous --num-double-angle 4 --degree 31 \\
      --domain 0.85

`python3 evalmod_coeffs.py check` regenerates both and compares them with the
values in BootParameter.cpp.
"""

import argparse
import decimal
import pathlib
import re
import sys
from decimal import Decimal as Dec

import numpy as np
from numpy.polynomial import chebyshev as cheb

INITIAL_K = 2  # BootParameter::initial_K_
LOG_MESSAGE_RATIO = 5  # BootParameter default

PRESETS = {
    3: dict(method="discrete", degree=30),
    4: dict(method="continuous", degree=31, domain=0.85),
}


def actual_k(num_double_angle):
    return (1 << num_double_angle) * INITIAL_K


def scale_factor(num_double_angle):
    return (0.5 / np.pi) ** (1.0 / 2 ** num_double_angle)


def target(x, num_double_angle):
    return scale_factor(num_double_angle) * np.cos(2 * np.pi * INITIAL_K * x)


def han_ki_degrees(degree, K, dev):
    """Number of nodes per integer interval (Lattigo genDegrees)."""
    log2 = np.log2
    log2_two_pi = log2(2 * np.pi)
    err = 1.0 / dev
    totdeg = 2 * K - 1
    deg = [1] * K
    base = -sum(log2(i) for i in range(1, 2 * K))
    base += (2 * K - 1) * log2_two_pi + log2(err)
    bdd = [base + sum(log2(j + err) for j in range(1, K - i))
           + sum(log2(j + err) for j in range(1, K + i)) for i in range(K)]
    for _ in range(200):
        if totdeg >= degree + 1:
            break
        maxi = int(np.argmax(bdd))
        if maxi != 0:
            if totdeg + 2 > degree + 1:
                break
            for i in range(K):
                bdd[i] += 2 * log2_two_pi - log2(totdeg + 1) - log2(totdeg + 2)
                if i != maxi:
                    bdd[i] += log2(abs(i - maxi) + err) + log2(i + maxi + err)
                else:
                    bdd[i] += log2(err) - 1.0 + log2(2.0 * i + err)
            totdeg += 2
        else:
            bdd[0] += log2(err) - 1.0 + log2_two_pi - log2(totdeg + 1)
            for i in range(1, K):
                bdd[i] += log2_two_pi + log2(i + err) - log2(totdeg + 1)
            totdeg += 1
        deg[maxi] += 1
    return deg, totdeg


def fit_discrete(num_double_angle, degree, log_message_ratio):
    """Han-Ki interpolation, matching Lattigo's CosDiscrete output.

    The nodes cluster around each integer, so the interpolation system is
    ill-conditioned. It is solved in 80-digit decimal arithmetic (Lattigo uses
    256-bit floats) and rounded to float64 at the end.
    """
    K = actual_k(num_double_angle)
    deg, totdeg = han_ki_degrees(degree, K, float(1 << log_message_ratio))
    if totdeg - 1 > degree:
        sys.exit(f"discrete: actual_K = {K} needs degree >= {totdeg - 1}, "
                 f"got {degree}; use the continuous method")
    with decimal.localcontext() as ctx:
        ctx.prec = 80
        pi = _dec_pi()
        nodes = [Dec(0)] if deg[0] % 2 else []
        for i in range(K - 1, -1, -1):
            for j in range(deg[i] // 2 if i == 0 else deg[i]):
                d = _dec_cos(pi * j / deg[i], pi) / (1 << log_message_ratio)
                nodes += [i + d, -(i + d)]
        x = [(n - Dec("0.25")) / K for n in nodes]
        scale = Dec(scale_factor(num_double_angle))
        y = [scale * _dec_cos(2 * pi * INITIAL_K * xi, pi) for xi in x]
        rows = []
        for xi in x:
            row = [Dec(1), xi]
            while len(row) < totdeg:
                row.append(2 * xi * row[-1] - row[-2])
            rows.append(row[:totdeg])
        coeffs = np.array([float(c) for c in _dec_solve(rows, y)])
    coeffs[1::2] = 0.0  # EvalMod evaluates an even polynomial
    return coeffs


def _dec_pi():
    """pi by the Gauss-Legendre iteration."""
    a, b, t, p = Dec(1), Dec(1) / Dec(2).sqrt(), Dec(1) / 4, Dec(1)
    for _ in range(10):
        a, b, t, p = ((a + b) / 2, (a * b).sqrt(),
                      t - p * ((a - b) / 2) ** 2, 2 * p)
    return (a + b) ** 2 / (4 * t)


def _dec_cos(x, pi):
    x %= 2 * pi
    eps = Dec(10) ** -(decimal.getcontext().prec + 2)
    term, total, n = Dec(1), Dec(1), 0
    while abs(term) > eps:
        n += 2
        term = -term * x * x / (n * (n - 1))
        total += term
    return total


def _dec_solve(a, b):
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    a = [row[:] + [bi] for row, bi in zip(a, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            for k in range(c, n + 1):
                a[r][k] -= f * a[c][k]
    x = [Dec(0)] * n
    for r in range(n - 1, -1, -1):
        s = sum(a[r][k] * x[k] for k in range(r + 1, n))
        x[r] = (a[r][n] - s) / a[r][r]
    return x


def fit_continuous(num_double_angle, degree, domain, samples=20001):
    """Even least-squares fit with Chebyshev weight on |x| <= domain."""
    x = np.linspace(-domain, domain, samples)
    w = (1.0 - (x / domain) ** 2 + 1e-9) ** -0.25  # rows: sqrt of the weight
    basis = cheb.chebvander(x / domain, degree)[:, ::2]
    c, *_ = np.linalg.lstsq(basis * w[:, None],
                            target(x, num_double_angle) * w, rcond=None)
    # Re-express sum c_j T_2j(x/D) in the T_n(x) basis on [-1, 1].
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    values = cheb.chebvander(nodes / domain, degree)[:, ::2] @ c
    coeffs = cheb.chebfit(nodes, values, degree)
    coeffs[1::2] = 0.0
    return coeffs


def chain_error(coeffs, num_double_angle, log_message_ratio, max_rounding_int):
    """Max |EvalMod(t) - sin(2pi actual_K t)/2pi| over the reachable inputs."""
    K = actual_k(num_double_angle)
    m = np.linspace(-1.0, 1.0, 257) * 2.0 ** -log_message_ratio
    rounding = np.arange(-max_rounding_int, max_rounding_int + 1)
    t = ((rounding[:, None] + m[None, :]) / K).ravel()
    y = cheb.chebval(t - 0.25 / K, coeffs)
    c = scale_factor(num_double_angle)
    for _ in range(num_double_angle):
        c *= c
        y = 2 * y * y - c
    return np.max(np.abs(y - np.sin(2 * np.pi * K * t) / (2 * np.pi)))


def generate(method, num_double_angle, degree, log_message_ratio, domain):
    if method == "discrete":
        return fit_discrete(num_double_angle, degree, log_message_ratio)
    return fit_continuous(num_double_angle, degree, domain)


def shipped_coefficients(path):
    """Parse SelectModCoefficients() in BootParameter.cpp."""
    src = pathlib.Path(path).read_text()
    out = {}
    pattern = r"num_double_angle == (\d+)\)\s*\{\s*return\s*\{([^}]*)\}"
    for k, body in re.findall(pattern, src):
        values = body.replace("\n", " ").split(",")
        out[int(k)] = np.array([float(v) for v in values if v.strip()])
    return out


def check(path):
    """Compare regenerated presets with BootParameter.cpp by their values.

    A fit on |x| <= D < 1 does not pin its Chebyshev coefficients on [-1, 1]
    down to float precision (many coefficient vectors agree on the fit
    domain), so the test is the polynomial's value on the domain it serves.
    """
    shipped = shipped_coefficients(path)
    ok = True
    for k, preset in PRESETS.items():
        ref = shipped.get(k)
        if ref is None:
            print(f"num_double_angle = {k}: not in {path}")
            ok = False
            continue
        got = generate(preset["method"], k, preset["degree"],
                       LOG_MESSAGE_RATIO, preset.get("domain"))
        D = preset.get("domain", 1.0)
        xs = np.linspace(-D, D, 100001)
        value_diff = np.max(np.abs(cheb.chebval(xs, got)
                                   - cheb.chebval(xs, ref)))
        nz = ref != 0
        coeff_diff = np.max(np.abs(got - ref)[nz] / np.abs(ref[nz]))
        good = len(got) == len(ref) and value_diff < 1e-13
        ok &= good
        print(f"num_double_angle = {k} ({preset['method']}): "
              f"{len(got)} coefficients, max relative coefficient difference "
              f"{coeff_diff:.1e}, max |p - p_shipped| on |x| <= {D} "
              f"{value_diff:.1e}: {'OK' if good else 'MISMATCH'}")
    return ok


def main():
    p = argparse.ArgumentParser(
        description="Generate EvalMod coefficients for BootParameter.")
    p.add_argument("method", choices=["discrete", "continuous", "check"])
    p.add_argument("--num-double-angle", type=int)
    p.add_argument("--degree", type=int)
    p.add_argument("--log-message-ratio", type=int, default=LOG_MESSAGE_RATIO)
    p.add_argument("--max-rounding-int", type=int,
                   help="largest |I| to support (default actual_K - 1)")
    p.add_argument("--domain", type=float,
                   help="continuous: fit on |x| <= DOMAIN (default: the "
                        "smallest domain that covers --max-rounding-int)")
    p.add_argument("--boot-parameter-cpp",
                   default=pathlib.Path(__file__).parent
                   / "src/extension/BootParameter.cpp")
    a = p.parse_args()

    if a.method == "check":
        sys.exit(0 if check(a.boot_parameter_cpp) else 1)
    if a.num_double_angle is None or a.degree is None:
        p.error("--num-double-angle and --degree are required")

    K = actual_k(a.num_double_angle)
    imax = K - 1 if a.max_rounding_int is None else a.max_rounding_int
    domain = a.domain
    if domain is None:
        domain = (imax + 0.25 + 2.0 ** -a.log_message_ratio) / K
    coeffs = generate(a.method, a.num_double_angle, a.degree,
                      a.log_message_ratio, domain)
    print("{")
    for i in range(0, len(coeffs), 2):
        pair = ", ".join(f"{v:.20g}" for v in coeffs[i:i + 2])
        print(f"    {pair}{',' if i + 2 < len(coeffs) else ''}")
    print("}")
    err = chain_error(coeffs, a.num_double_angle, a.log_message_ratio, imax)
    print(f"// EvalMod chain error {err:.2e} for |I| <= {imax} (float64)")


if __name__ == "__main__":
    main()
