import asyncio
import websockets
import json
import sys

BOT = sys.argv[1] if len(sys.argv) > 1 else "entebot208"
WS = f"ws://{BOT}.local:9001/ros_api"

async def main():
    async with websockets.connect(WS) as ws:
        print("Requesting topic list...")
        await ws.send(json.dumps({
            "op": "call_service",
            "service": "/rosapi/topics",
            "args": {}
        }))
        
        resp = await ws.recv()
        print(resp)

asyncio.run(main())
