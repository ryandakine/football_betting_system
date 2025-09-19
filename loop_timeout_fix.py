
# Add to all loop files:
import signal
def timeout_handler(signum, frame): raise TimeoutError("Loop killed")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute timeout
