__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import asyncio
import json
import select
import sys

has_termios = False
termios = None
tty = None

try:
    import termios
    import tty
    has_termios = True
except ImportError:
    pass

try:
    import websockets
except ImportError:
    print("Error: The 'websockets' library is required to run the agent.")
    print("Please install it: pip install websockets")
    sys.exit(1)

async def receive_loop(websocket):
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "setup":
                print(f"\n[Handshake Complete] Assigned Space Invaders Player ID: {data.get('player_id')}")
                print("Controls: A to move LEFT, D to move RIGHT, SPACEBAR to shoot, Q to quit.")
                print("="*60)
            elif data.get("type") in ("state", "update"):
                score = data.get("score", 0)
                lives = data.get("lives", 0)
                high_score = data.get("high_score", 0)
                player_x = data.get("player_x", 0.0)
                game_over = data.get("game_over", False)
                aliens = data.get("aliens", [])
                lasers = data.get("lasers", [])

                # Clean screen redraw
                sys.stdout.write("\033[H\033[2J")
                sys.stdout.flush()

                print("="*18 + " SPACE INVADERS TERMINAL HUD " + "="*18)
                print(f"HIGH SCORE: {high_score:<10} | SCORE: {score:<10} | LIVES: {lives:<5}")
                print("-" * 65)
                print(f"Spaceship X Position : {player_x:<8.1f}")
                print(f"Active Aliens        : {len(aliens):<5} / 10 | Lasers on Screen: {len(lasers)}")
                if game_over:
                    print("="*18 + " 💥 GAME OVER! 💥 " + "="*18)
                else:
                    print("="*65)
                print("\n[ACTIVE INPUT] Focus this terminal. Press A/D to move, SPACE to shoot. Press Q to quit.")
    except websockets.exceptions.ConnectionClosed:
        print("\nDisconnected from Space Invaders Server.")

async def send_loop(websocket):
    fd = sys.stdin.fileno() if (has_termios and termios is not None) else None
    old_settings = None
    if fd is not None and termios is not None and tty is not None:
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

    try:
        while True:
            key = ""
            if has_termios and fd is not None:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = sys.stdin.read(1)
            else:
                line = sys.stdin.readline().strip().lower()
                if line == "a":
                    key = "a"
                elif line == "d":
                    key = "d"
                elif line == " ":
                    key = " "
                elif line == "q":
                    key = "q"

            if key:
                if key.lower() == "q":
                    break
                if key.lower() == "a":
                    await websocket.send(json.dumps({"action": "move", "direction": "WEST"}))
                elif key.lower() == "d":
                    await websocket.send(json.dumps({"action": "move", "direction": "EAST"}))
                elif key == " ":
                    await websocket.send(json.dumps({"action": "shoot"}))
            await asyncio.sleep(0.02)
    finally:
        if fd is not None and old_settings is not None and termios is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nExiting Manual Agent...")

async def main():
    url = "ws://localhost:8765/ws"
    print(f"Connecting to Space Invaders Server on {url}...")
    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps({"client": "agent", "name": "Terminal Space Invaders"}))
            await asyncio.gather(
                receive_loop(websocket),
                send_loop(websocket)
            )
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSpace Invaders driver exited.")
