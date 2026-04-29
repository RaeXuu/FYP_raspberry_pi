"""
测试 iOS lfilter 实现是否正确 —— 与 scipy lfilter/filtfilt 对比
"""
import numpy as np
from scipy.signal import butter, filtfilt as scipy_filtfilt, lfilter as scipy_lfilter

b = [0.016987710409093026, 0.0, -0.08493855204546513, 0.0,
     0.16987710409093026, 0.0, -0.16987710409093026, 0.0,
     0.08493855204546513, 0.0, -0.016987710409093026]
a = [1.0, -5.88635597176117, 15.590871175483045, -24.902215524177628,
     27.031963965883904, -21.023701701970058, 11.819133833668117,
     -4.705870557530761, 1.2734414445826752, -0.21324429325710365,
     0.015979779753429305]

M = len(b) - 1
N = len(a) - 1


def odd_extend(x, padlen):
    first, last = x[0], x[-1]
    left = [2 * first - x[i + 1] for i in range(padlen)][::-1]
    right = [2 * last - x[-2 - i] for i in range(padlen)]
    return np.array(left + list(x) + right, dtype=np.float64)


def ios_lfilter(x):
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros(len(x))
    xi = np.zeros(max(M, N))
    yi = np.zeros(N)
    for n in range(len(x)):
        yn = b[0] * x[n]
        if n > 0:
            mb = min(M, n)
            for i in range(1, mb + 1):
                yn += b[i] * xi[i - 1]
            nb = min(N, n)
            for i in range(1, nb + 1):
                yn -= a[i] * yi[i - 1]
        if M > 0:
            for i in range(M - 1, 0, -1):
                xi[i] = xi[i - 1]
            xi[0] = x[n]
        if N > 0:
            for i in range(N - 1, 0, -1):
                yi[i] = yi[i - 1]
            yi[0] = yn
        y[n] = yn
    return y


def ios_filtfilt(x):
    padlen = 33
    padded = odd_extend(x, padlen)
    fwd = ios_lfilter(padded)
    rev = fwd[::-1].copy()
    bwd = ios_lfilter(rev)
    result = bwd[::-1].copy()
    return result[padlen:-padlen]


# ── Test 1: lfilter 对比 ──
np.random.seed(42)
test_x = np.random.randn(1000).astype(np.float64)

ios_y = ios_lfilter(test_x)
scipy_y = scipy_lfilter(b, a, test_x)

print("=== Test 1: lfilter ===")
print(f"  iOS vs scipy lfilter match: {np.allclose(ios_y, scipy_y, atol=1e-6)}")
print(f"  Max diff: {np.max(np.abs(ios_y - scipy_y)):.2e}")

# ── Test 2: filtfilt 对比 ──
ios_ff = ios_filtfilt(test_x)
scipy_ff = scipy_filtfilt(b, a, test_x)

print("\n=== Test 2: filtfilt ===")
print(f"  iOS vs scipy filtfilt match: {np.allclose(ios_ff, scipy_ff, atol=1e-6)}")
print(f"  Max diff: {np.max(np.abs(ios_ff - scipy_ff)):.2e}")

# ── Test 3: toy input 对比 ──
toy = [0.079518, 0.194913, 0.208568, 0.214726, 0.157965,
       0.064793, 0.050067, 0.035877, 0.037216, 0.025971]
toy_x = np.array(toy + [0.0] * 39990, dtype=np.float64)

ios_toy = ios_filtfilt(toy_x)
scipy_toy = scipy_filtfilt(b, a, toy_x)

print("\n=== Test 3: toy input ===")
print(f"  iOS:    {[f'{v:.6f}' for v in ios_toy[:10]]}")
print(f"  scipy:  {[f'{v:.6f}' for v in scipy_toy[:10]]}")
print(f"  Match: {np.allclose(ios_toy, scipy_toy, atol=1e-6)}")

#! Result:

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/test_lfilter.py
# === Test 1: lfilter ===
#   iOS vs scipy lfilter match: True
#   Max diff: 6.34e-10

# === Test 2: filtfilt ===
#   iOS vs scipy filtfilt match: False
#   Max diff: 1.59e-01

# === Test 3: toy input ===
#   iOS:    ['-0.010407', '0.091252', '0.143254', '0.128215', '0.069160', '0.008248', '-0.025287', '-0.031598', '-0.029898', '-0.035356']
#   scipy:  ['-0.000000', '0.102254', '0.154734', '0.140062', '0.081267', '0.020518', '-0.012950', '-0.019283', '-0.017690', '-0.023332']
#   Match: False