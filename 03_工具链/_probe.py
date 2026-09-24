import socket, json
def send(line, t=10):
    s = socket.create_connection(("127.0.0.1", 8792), timeout=t); s.settimeout(t)
    s.sendall((line + "\n").encode()); buf = b""
    while b"\n" not in buf:
        c = s.recv(65536)
        if not c: break
        buf += c
    s.close(); return buf.decode("utf-8", "replace").strip()
for probe in ["HELP", "TOKEN", "STATUS", "RECONNECT"]:
    try:
        print("%-10s -> %s" % (probe, send(probe)[:300]))
    except Exception as e:
        print("%-10s -> EXC %s" % (probe, e))
