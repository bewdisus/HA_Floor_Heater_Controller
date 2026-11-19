#!/usr/bin/env python3
"""Идентификация тепловой модели пола по выгрузкам истории Home Assistant.

Использование:
    python thermal_model.py <floor_temperature.csv> <pump_running.csv>

Модель первого порядка:
    dT/dt = u * Q/C - (T - T_amb) / tau,   u = 1 при работающем насосе

Параметры (tau, T_amb, Q/C) подбираются по измеренным фазам нагрева и остывания:
    остывание:  t = tau * ln((T_hi - T_amb) / (T_lo - T_amb))
    нагрев:     t = integral dT / (Q/C - (T - T_amb)/tau)

Печатает параметры, проверку и прогноз цикла для разных ширин гистерезиса.
"""

import sys
import numpy as np
import pandas as pd


def load_t(p):
    d = pd.read_csv(p)
    d["ts"] = pd.to_datetime(d["last_changed"], utc=True)
    d["v"] = pd.to_numeric(d["state"], errors="coerce")
    d = d.dropna(subset=["v"]).sort_values("ts").drop_duplicates("ts")
    s = d.set_index("ts")["v"]
    s.index = s.index.astype("datetime64[ns, UTC]")
    return s


def load_p(p):
    d = pd.read_csv(p)
    d["ts"] = pd.to_datetime(d["last_changed"], utc=True)
    d = d[d["state"].isin(["on", "off"])].sort_values("ts").drop_duplicates("ts")
    s = d.set_index("ts")["state"]
    s.index = s.index.astype("datetime64[ns, UTC]")
    return s


def phases(t, p, min_len=5):
    tr = p[p != p.shift()].iloc[1:]
    out = []
    for i in range(len(tr) - 1):
        a, st, b = tr.index[i], tr.iloc[i], tr.index[i + 1]
        dur = (b - a).total_seconds() / 60.0
        if dur < min_len:
            continue
        t0, t1 = t.asof(a), t.asof(b)
        if np.isnan(t0) or np.isnan(t1) or abs(t1 - t0) < 0.3:
            continue
        out.append((st, t0, t1, dur))
    return out


def fit_off(ph):
    off = [x for x in ph if x[0] == "off"]
    best = None
    for ta in np.arange(10.0, 26.0, 0.05):
        a = np.array([np.log((x[1] - ta) / (x[2] - ta)) for x in off])
        d = np.array([x[3] for x in off])
        tau = float(np.dot(a, d) / np.dot(a, a))
        err = float(np.sum((tau * a - d) ** 2))
        if best is None or err < best[1]:
            best = (ta, err, tau)
    return best  # ta, err, tau


def fit_q(ph, ta, tau):
    on = [x for x in ph if x[0] == "on"]
    def pred(q):
        out = []
        for _, t0, t1, _ in on:
            Ts = np.linspace(min(t0, t1), max(t0, t1), 400)
            out.append(abs(np.trapezoid(1.0 / (q - (Ts - ta) / tau), Ts)))
        return np.array(out)
    d = np.array([x[3] for x in on])
    lo, hi = 0.02, 0.6
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.sum(pred(mid)) > np.sum(d):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cycle(target, h, ta, tau, q):
    a, b = target - h / 2, target + h / 2
    if b >= ta + q * tau:
        return float("nan"), float("nan")
    Ts = np.linspace(a, b, 4001)
    th = float(np.trapezoid(1.0 / (q - (Ts - ta) / tau), Ts))
    tc = float(tau * np.log((b - ta) / (a - ta)))
    return th, tc


def main(tp, pp):
    t, p = load_t(tp), load_p(pp)
    ph = phases(t, p)
    off = [x for x in ph if x[0] == "off"]
    on = [x for x in ph if x[0] == "on"]
    ta, err, tau = fit_off(ph)
    q = fit_q(ph, ta, tau)

    print("фаз: нагрев %d, простой %d" % (len(on), len(off)))
    print("\n=== ПАРАМЕТРЫ МОДЕЛИ ===")
    print("tau    = %.0f мин (%.1f ч)" % (tau, tau / 60))
    print("T_amb  = %.2f °C" % ta)
    print("Q/C    = %.4f °C/мин (%.2f °C/ч)" % (q, q * 60))
    print("T_eq   = %.1f °C (равновесие при непрерывной работе насоса)" % (ta + q * tau))
    print("наклон при 30 °C: ON %+.4f, OFF %+.4f °C/мин"
          % (q - (30 - ta) / tau, -(30 - ta) / tau))

    print("\n=== ПРОВЕРКА ===")
    res = []
    for st, t0, t1, dur in ph:
        a, b = min(t0, t1), max(t0, t1)
        if st == "off":
            pr = tau * np.log((b - ta) / (a - ta))
        else:
            Ts = np.linspace(a, b, 400)
            pr = abs(np.trapezoid(1.0 / (q - (Ts - ta) / tau), Ts))
        res.append(pr - dur)
    res = np.array(res)
    print("ошибка по фазам: медиана %+.1f мин, p90 |ошибка| %.1f мин" %
          (np.median(res), np.percentile(np.abs(res), 90)))

    print("\n=== ПРОГНОЗ ЦИКЛА (target 30 °C) ===")
    print(" H, °C | нагрев, мин | простой, мин | период, мин | пусков/сутки")
    for h in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
        th, tc = cycle(30.0, h, ta, tau, q)
        print(" %5.2f | %11.1f | %12.1f | %11.1f | %13.0f"
              % (h, th, tc, th + tc, 1440 / (th + tc)))

    print("\nпредел регулирования: цель выше %.1f °C недостижима (T_eq - H/2)"
          % (ta + q * tau))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
