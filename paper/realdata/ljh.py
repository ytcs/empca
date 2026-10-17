"""Minimal reader for NIST LJH pulse files (versions 2.1 and 2.2)."""
import numpy as np


def read_ljh(path):
    raw = open(path, "rb").read()
    end = raw.index(b"#End of Header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("latin-1")
    info = {}
    for line in header.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    n = int(info["Total Samples"])
    version = info.get("Save File Format Version", "2.1.0")
    if version.startswith("2.2"):
        dt = np.dtype([("subframecount", "<i8"), ("posix_usec", "<i8"), ("data", "<u2", n)])
    else:
        dt = np.dtype([("junk", "<u1", 6), ("data", "<u2", n)])
    nrec = (len(raw) - end) // dt.itemsize
    rec = np.frombuffer(raw, dtype=dt, count=nrec, offset=end)
    return rec["data"].astype(float), info
