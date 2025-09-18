import numpy as np
import hashlib, hmac

def decode(payload):
    ans = ""
    for i in range(0, len(payload), 2):
        b = int(payload[i:i+2], 16)
        ans += chr(b) if 0x20 <= b < 0x7F else payload[i:i+2]
    return ans

def AE2(s, window_size):
    n, i, buf = len(s), 0, []
    out = []
    while i < n:
        mv, mp= int(s[i:i+2], 16), i
        buf.append(s[i:i+2])
        i += 2
        while i < n:
            v = int(s[i:i+2], 16)
            if v <= mv:
                if i == mp + 2*window_size:
                    buf.append(s[i:i+2])
                    out.append("".join(buf))
                    buf = []
                    i += 2
                    break
            else:
                mv, mp = v, i
            buf.append(s[i:i+2])
            i += 2
    if buf:
        out.append("".join(buf))
    return out

def contents2count(hex_list, vec_size, win_size, key=None):
    V = []
    for s in hex_list:
        v = np.zeros(vec_size, dtype=np.float64)
        for ch in AE2(s, win_size):
            data = ch.encode()
            if key is None:
                h = hashlib.sha256(data).digest()
            else:
                k = key if isinstance(key, bytes) else str(key).encode()
                h = hmac.new(k, data, hashlib.sha256).digest()
            idx = int.from_bytes(h[:8], "big") % vec_size
            v[idx] += 1.0
        V.append(v)
    return np.vstack(V) if V else np.zeros((0, vec_size), dtype=np.float64)
