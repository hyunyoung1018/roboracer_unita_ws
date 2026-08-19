#!/usr/bin/env python3
"""정지 장애물의 s/d 지터를 실측한다. 20초 모으고 요약만 출력.

    python3 measure_jitter.py [초]

/tracking/classification_debug 는 구독자가 없으면 발행을 건너뛰므로,
이 스크립트가 붙는 것만으로 켜진다. 끝나면 다시 꺼진다.
"""
import json, sys, math
from collections import defaultdict
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0


def std(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


class Probe(Node):
    def __init__(self):
        super().__init__('jitter_probe')
        self.rows = defaultdict(list)
        self.create_subscription(
            String, '/tracking/classification_debug', self.cb, 10)
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

    def cb(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        t = float(d.get('stamp', 0.0))
        for o in d.get('obstacles', ()):
            self.rows[int(o['tracker_id'])].append(
                (t, float(o['s_center']), float(o['d_center']),
                 float(o.get('vs', 0.0)), bool(o.get('is_static', False)),
                 str(o.get('stable_class', '?')),
                 bool(o.get('selected_opponent', False))))

    def done(self):
        return self.get_clock().now().nanoseconds * 1e-9 - self.t0 > DUR


def main():
    rclpy.init()
    n = Probe()
    while rclpy.ok() and not n.done():
        rclpy.spin_once(n, timeout_sec=0.1)

    print(f"\n{DUR:.0f} 초 수집, 트랙 {len(n.rows)} 개\n")
    hdr = (f"{'id':>5} {'n':>4} {'지속s':>6} {'s 표준편차':>10} {'s 최대-최소':>11} "
           f"{'프레임간 Δs':>12} {'d 표준편차':>10} {'|vs| 최대':>9} "
           f"{'non-static':>11} {'상대선택':>8}")
    print(hdr)
    print('-' * len(hdr))
    for tid, r in sorted(n.rows.items(), key=lambda kv: -len(kv[1])):
        if len(r) < 10:
            continue
        ts = [x[0] for x in r]
        s = [x[1] for x in r]
        dd = [x[2] for x in r]
        vs = [abs(x[3]) for x in r]
        ns = sum(1 for x in r if not x[4])
        sel = sum(1 for x in r if x[6])
        ds = [abs(s[i + 1] - s[i]) for i in range(len(s) - 1)]
        print(f"{tid:5d} {len(r):4d} {ts[-1]-ts[0]:6.1f} {std(s):10.4f} "
              f"{max(s)-min(s):11.4f} {std(ds):12.4f} {std(dd):10.4f} "
              f"{max(vs):9.3f} {ns*100.0/len(r):10.0f}% {sel*100.0/len(r):7.0f}%")
    print("\n창 길이별 '순 변위' — 정지 장애물이면 작아야 한다")
    print(f"{'id':>5} {'1.0s':>8} {'1.5s':>8} {'2.0s':>8}   (양끝 5개 중앙값 차이, m)")
    for tid, r in sorted(n.rows.items(), key=lambda kv: -len(kv[1])):
        if len(r) < 45:
            continue
        s = [x[1] for x in r]
        out = []
        for w in (20, 30, 40):
            best = 0.0
            for i in range(len(s) - w):
                win = s[i:i + w]
                a = sorted(win[:5])[2]
                b = sorted(win[-5:])[2]
                best = max(best, abs(b - a))
            out.append(best)
        print(f"{tid:5d} {out[0]:8.4f} {out[1]:8.4f} {out[2]:8.4f}")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
