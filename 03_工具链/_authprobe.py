import paramiko, socket
# 只查"跳板允许哪些认证方式"：auth_none 不需要 token，因此**不消耗**它
try:
    t = paramiko.Transport(("113.47.8.48", 2234))
    t.start_client(timeout=20)
    try:
        t.auth_none("probe")
        print("AUTH_NONE_OK（意外：无需认证）")
    except paramiko.BadAuthenticationType as e:
        print("ALLOWED_METHODS =", e.allowed_types)
    except Exception as e:
        print("AUTH_PROBE_OTHER %s: %s" % (type(e).__name__, e))
    t.close()
except Exception as e:
    print("TRANSPORT_FAIL %s: %s" % (type(e).__name__, e))
