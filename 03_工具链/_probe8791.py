import socket
def send(port, line, t=6):
    s = socket.create_connection(("127.0.0.1", port), timeout=t); s.settimeout(t)
    s.sendall((line + "\n").encode()); buf = b""
    try:
        while b"\n" not in buf:
            c = s.recv(65536)
            if not c: break
            buf += c
    except Exception as e:
        buf += ("<timeout %s>" % e).encode()
    s.close(); return buf.decode("utf-8", "replace").strip()
for probe in ["PING", "HELP", "STATUS", "MINT", "GET", "TOKEN"]:
    try:
        print("%-8s -> %s" % (probe, send(8791, probe)[:240]))
    except Exception as e:
        print("%-8s -> EXC %s" % (probe, e))
