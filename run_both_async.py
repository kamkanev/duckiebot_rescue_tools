#!/usr/bin/env python3
"""
Run `duckie_teleop_gui.py` and `testing/test_stop_lanefollow.py` concurrently as async subprocesses.
Usage: python3 run_both_async.py [BOT_NAME]

Example: python3 run_both_async.py entebot208
"""

import asyncio
import os
import sys

PY = sys.executable or "python3"

async def stream_process(cmd, name):
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    print(f"Started {name} (pid {proc.pid})")
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(f"[{name}] {line.decode().rstrip()}")
    except asyncio.CancelledError:
        proc.terminate()
        raise
    rc = await proc.wait()
    print(f"{name} exited with {rc}")
    return rc

async def main():
    bot = sys.argv[1] if len(sys.argv) > 1 else None
    root = os.getcwd()
    teleop_cmd = [PY, os.path.join(root, "duckie_teleop_gui.py")]
    test_cmd = [PY, os.path.join(root, "testing", "test_stop_lanefollow.py")]
    if bot:
        teleop_cmd.append(bot)
        test_cmd.append(bot)

    t1 = asyncio.create_task(stream_process(teleop_cmd, "teleop"))
    t2 = asyncio.create_task(stream_process(test_cmd, "test_stop"))

    done, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
    # if one exits, cancel the other
    for p in pending:
        p.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Runner terminated by user")
