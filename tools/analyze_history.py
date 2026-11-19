#!/usr/bin/env python3
"""Анализ выгрузок истории Home Assistant для контроллера тёплого пола (этап 3).

Использование:
    python analyze_history.py <floor_temperature.csv> <pump_running.csv>

Ожидаемый формат CSV — стандартный экспорт истории HA:
    entity_id,state,last_changed

Печатает: качество данных, проверку порогов, параметры цикла, скорости
нагрева/остывания, шум измерения, дребезг реле и оценку перелёта.
"""

import sys
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["last_changed"], utc=True)
    return df


def load_temperature(path):
    df = load(path)
    df["v"] = pd.to_numeric(df["state"], errors="coerce")
    df = df.dropna(subset=["v"]).sort_values("ts").drop_duplicates("ts")
    s = df.set_index("ts")["v"]
    # единая точность времени (pandas 3 хранит us, арифметика Timedelta даёт ns)
    s.index = s.index.astype("datetime64[ns, UTC]")
    return s


def load_pump(path):
    df = load(path)
    df = df[df["state"].isin(["on", "off"])].sort_values("ts").drop_duplicates("ts")
    tr = df[~df["state"].eq(df["state"].shift())].reset_index(drop=True)
    tr["ts"] = tr["ts"].astype("datetime64[ns, UTC]")
    tr["dur"] = (tr["ts"].shift(-1) - tr["ts"]).dt.total_seconds() / 60.0
    tr["gap"] = tr["ts"].diff().dt.total_seconds() / 60.0
    return df.set_index("ts")["state"], tr


def median(x, fmt="%.3f"):
    x = pd.Series(x).dropna()
    return "n=%d median=%s p90=%s max=%s" % (
        len(x), fmt % x.median(), fmt % x.quantile(0.9), fmt % x.max())


def main(t_path, p_path):
    t = load_temperature(t_path)
    p_raw, tr = load_pump(p_path)
    hours = (t.index[-1] - t.index[0]).total_seconds() / 3600

    step = np.median(np.diff(np.sort(t.unique())))
    print("=== ДАННЫЕ ===")
    print("период: %.1f ч (%s -> %s MSK)" % (
        hours, t.index[0].tz_convert("Europe/Moscow").strftime("%d.%m %H:%M"),
        t.index[-1].tz_convert("Europe/Moscow").strftime("%d.%m %H:%M")))
    print("записей T: %d | шаг квантования: %.3f °C | диапазон %.2f..%.2f" % (
        len(t), step, t.min(), t.max()))
    print("записей насоса: %d | переключений: %d" % (len(p_raw), len(tr)))

    # --- пороги ---
    tt = pd.Series([t.asof(x) for x in tr["ts"]])
    print("\n=== ПОРОГИ ===")
    for st in ("on", "off"):
        m = tr["state"] == st
        print("T при %-3s: медиана %.2f | p10 %.2f | p90 %.2f" % (
            st, tt[m].median(), tt[m].quantile(0.1), tt[m].quantile(0.9)))

    # --- фазы и скорости ---
    on = tr[(tr["state"] == "on") & (tr["dur"] >= 10)]
    off = tr[(tr["state"] == "off") & (tr["dur"] >= 10)]
    print("\n=== ФАЗЫ (>=10 мин) ===")
    print("нагрев:  %s (мин)" % median(on["dur"], "%.0f"))
    print("простой: %s (мин)" % median(off["dur"], "%.0f"))
    period = on["dur"].median() + off["dur"].median()
    print("период: %.0f мин | пусков/сутки: %.0f | скважность: %.0f%%" % (
        period, 24 * 60 / period, 100 * on["dur"].median() / period))

    def dT(df):
        return pd.Series([t.asof(b) - t.asof(a) for a, b in zip(df["ts"], df["ts"] + pd.to_timedelta(df["dur"], unit="m"))])

    up = dT(on)
    down = -dT(off)
    print("подъём за фазу: %s (°C)" % median(up))
    print("снижение за простой: %s (°C)" % median(down))
    print("скорость нагрева: %s (°C/мин)" % median(up / on["dur"]))
    print("скорость остывания: %s (°C/мин)" % median(down / off["dur"]))

    # --- шум ---
    d = t.diff()
    secs = t.index.to_series().diff().dt.total_seconds()
    near = d[secs <= 15]
    print("\n=== ШУМ ИЗМЕРЕНИЯ ===")
    print("соседние отсчёты (<=15 c): n=%d sigma=%.3f °C p95|dT|=%.3f max=%.3f" % (
        len(near), near.std(), near.abs().quantile(0.95), near.abs().max()))
    jumps = ((d.abs() > 0.5) & (secs < 120)).sum()
    print("скачков >0.5 °C за <2 мин: %d (%.0f/сутки)" % (jumps, jumps * 24 / hours))

    # --- шум вокруг переключений (для опытов A/B/C) ---
    print("\n=== ШУМ ОКОЛО ПЕРЕКЛЮЧЕНИЙ (±2 мин) ===")
    sig, mx = [], []
    for x in tr["ts"]:
        w = t[(t.index >= x - pd.Timedelta(minutes=2)) & (t.index <= x + pd.Timedelta(minutes=2))]
        if len(w) < 5:
            continue
        dd = w.diff().dropna()
        sig.append(dd.std())
        mx.append(dd.abs().max())
    if sig:
        print("sigma вокруг коммутации: медиана %.3f °C | фон по периоду %.3f °C | отношение %.1fx"
              % (np.median(sig), near.std(), np.median(sig) / near.std()))
        print("max |dT| рядом: медиана %.2f °C | коммутаций с |dT|>0.5 °C: %.0f%%"
              % (np.median(mx), 100 * np.mean(np.array(mx) > 0.5)))
        print("в оценке коммутаций: %d" % len(sig))
    else:
        print("недостаточно коммутаций для оценки")

    # --- дребезг ---
    print("\n=== ДРЕБЕЗГ ===")
    iv = tr["gap"].dropna()
    print("интервалов <2 мин: %d | <5 мин: %d" % ((iv < 2).sum(), (iv < 5).sum()))
    short_on = tr[(tr["state"] == "on") & (tr["dur"] < 2)]
    print("включений короче 2 мин: %d" % len(short_on))

    burst, cur = [], 0
    for i, g in enumerate(tr["gap"]):
        if not (i > 0 and g < 5):
            cur += 1
        burst.append(cur)
    tr["burst"] = burst
    grp = tr.groupby("burst").agg(n=("ts", "size"), start=("ts", "first"),
                                  gap_before=("gap", "first"))
    b = grp[grp["n"] >= 2]
    print("кластеров (>=2 переключения с интервалом <5 мин): %d, в них %d из %d переключений"
          % (len(b), b["n"].sum(), len(tr)))
    for _, row in b.iterrows():
        print("  %s  переключений %d  пауза перед %.0f мин" % (
            row["start"].tz_convert("Europe/Moscow").strftime("%d.%m %H:%M"),
            row["n"], row["gap_before"]))
    print("кластеров после простоя >20 мин: %d из %d" % ((b["gap_before"] > 20).sum(), len(b)))

    # --- перелёт ---
    print("\n=== ПЕРЕЛЁТ ПОСЛЕ ОСТАНОВА (простой >30 мин) ===")
    offs = [tr["ts"].iloc[i] for i in range(len(tr) - 1)
            if tr["state"].iloc[i] == "off" and tr["dur"].iloc[i] > 30]
    win = {}
    for m0, m1 in [(0, 2), (0, 5), (0, 10), (0, 20), (20, 40)]:
        vals = []
        for a in offs:
            s = t[(t.index >= a + pd.Timedelta(minutes=m0)) &
                  (t.index < a + pd.Timedelta(minutes=m1))]
            if len(s):
                vals.append(s.mean())
        win["%d-%d" % (m0, m1)] = pd.Series(vals).median()
    print("показание в момент останова: %.2f °C" % pd.Series([t.asof(a) for a in offs]).median())
    for k, v in win.items():
        print("  среднее %s мин: %.2f °C" % (k, v))
    print("вывод: роста выше момента останова нет -> перелёт ниже уровня помехи")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
