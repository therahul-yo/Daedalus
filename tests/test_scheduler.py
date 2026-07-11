import threading
import time

from daedalus.scheduler import FifoLock


def test_fifo_lock_serves_waiters_in_arrival_order():
    lock = FifoLock()
    order = []
    gates = [threading.Event() for _ in range(3)]

    def worker(number):
        gates[number].wait()
        with lock:
            order.append(number)

    threads = [threading.Thread(target=worker, args=(number,)) for number in range(3)]
    with lock:
        for number, thread in enumerate(threads):
            thread.start()
            gates[number].set()
            deadline = time.monotonic() + 1
            while lock.queued < number + 1 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert lock.queued == number + 1
    for thread in threads:
        thread.join()
    assert order == [0, 1, 2]
