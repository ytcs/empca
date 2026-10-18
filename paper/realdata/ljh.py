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


def write_ljh(path, records, timebase, presamples, channel=1, offset=1000.0, scale=2000.0):
    """Write records (n, N) as an LJH 2.2 file: uint16 = offset + scale * value."""
    records = np.atleast_2d(records)
    n, N = records.shape
    header = (
        "#LJH Memorial File Format\n"
        "Save File Format Version: 2.2.0\n"
        "Software Version: empca simulation\n"
        "Digitized Word Size in Bytes: 2\n"
        f"Channel: {channel}\n"
        "Number of columns: 1\nNumber of rows: 1\n"
        "Column number (from 0-0 inclusive): 0\nRow number (from 0-0 inclusive): 0\n"
        "#End of description\n"
        "Number of Digitizers: 1\nNumber of Active Channels: 1\n"
        "Timestamp offset (s): 1700000000.000000\n"
        "Digitizer: 1\nBits: 16\n"
        f"Timebase: {timebase:e}\n"
        "Number of samples per point: 1\n"
        f"Presamples: {presamples}\n"
        f"Total Samples: {N}\n"
        "#End of Header\n"
    )
    dt = np.dtype([("subframecount", "<i8"), ("posix_usec", "<i8"), ("data", "<u2", N)])
    rec = np.zeros(n, dtype=dt)
    rec["subframecount"] = np.arange(n) * 100000
    rec["posix_usec"] = 1700000000000000 + np.arange(n) * 10000
    rec["data"] = np.clip(np.round(offset + scale * records), 0, 65535).astype(np.uint16)
    with open(path, "wb") as f:
        f.write(header.encode("latin-1"))
        f.write(rec.tobytes())
