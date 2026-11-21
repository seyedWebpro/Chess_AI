"""
dom_bridge_server.py: Asynchronous WebSocket Bridge Server for browser DOM updates.
Receives live board and piece mutations from the userscript in Chess.com/Lichess
and bridges them to Python with sub-millisecond dispatch.
"""

from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass
import asyncio
import json
import time
import threading
import chess
from loguru import logger
import websockets
from websockets.asyncio.server import Server, serve

from online.dom_parsers import (
    parse_chess_com_dom_pieces,
    parse_lichess_dom_pieces,
    extract_chess_com_highlights,
)


@dataclass
class DOMGameState:
    """Represents a validated snapshot of the chess game from the browser DOM."""
    fen: str
    pieces: Dict[str, str]
    site: str
    orientation: str
    highlights: List[str]
    last_move: Optional[str]
    timestamp: float
    latency_ms: float = 0.0


class DOMBridgeServer:
    """
    Async WebSocket server bridging browser DOM events to Python.
    Default port: 8765.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_update_callback: Optional[Callable[[DOMGameState], None]] = None,
    ):
        self.host = host
        self.port = port
        self.on_update_callback = on_update_callback
        self.latest_state: Optional[DOMGameState] = None
        self._connected_clients: Set[Any] = set()
        self._server: Optional[Server] = None
        self._is_running = False
        self._board_history: List[DOMGameState] = []

    @property
    def client_count(self) -> int:
        """Returns number of actively connected browser tabs."""
        return len(self._connected_clients)

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def _handle_client(self, websocket: Any):
        """Handles incoming WebSocket connection from browser userscript."""
        self._connected_clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Browser DOM Bridge connected from: {client_addr} (Total clients: {self.client_count})")

        try:
            async for message in websocket:
                t_recv = time.time()
                try:
                    data = json.loads(message)
                except Exception as e:
                    logger.warning(f"Invalid JSON from DOM bridge: {e}")
                    continue

                msg_type = data.get("type")
                if msg_type == "dom_update":
                    payload = data.get("payload", {})
                    state = self._process_dom_payload(payload, t_recv)
                    if state:
                        self.latest_state = state
                        self._board_history.append(state)
                        if self.on_update_callback:
                            try:
                                self.on_update_callback(state)
                            except Exception as cb_err:
                                logger.error(f"Error in DOM update callback: {cb_err}")

                elif msg_type == "pong":
                    t_sent = data.get("timestamp", t_recv * 1000) / 1000.0
                    rtt_ms = (t_recv - t_sent) * 1000.0
                    if self.latest_state:
                        self.latest_state.latency_ms = rtt_ms

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Browser DOM Bridge connection closed: {client_addr}")
        finally:
            self._connected_clients.discard(websocket)
            logger.info(f"Client disconnected. Remaining clients: {self.client_count}")

    def _process_dom_payload(self, payload: Dict[str, Any], recv_time: float) -> Optional[DOMGameState]:
        """Parses browser DOM payload into a structured DOMGameState."""
        site = payload.get("site", "chess.com")
        orientation = payload.get("orientation", "white")
        t_browser = payload.get("timestamp", recv_time)
        latency_ms = max(0.0, (recv_time - t_browser) * 1000.0)

        pieces_dict: Dict[str, str] = {}
        highlights: List[str] = []
        fen: str = ""

        if site == "chess.com":
            raw_pieces = payload.get("pieces", [])
            raw_highlights = payload.get("highlights", [])
            fen, pieces_dict = parse_chess_com_dom_pieces(raw_pieces)
            highlights = extract_chess_com_highlights(raw_highlights)

        elif site == "lichess":
            raw_pieces = payload.get("pieces", [])
            fen, pieces_dict = parse_lichess_dom_pieces(raw_pieces)

        # Detect last move if 2 highlight squares exist
        last_move = None
        if len(highlights) == 2:
            # e.g. e2 and e4
            # If we have previous state, check which square lost a piece and which gained/changed
            if self.latest_state and self.latest_state.pieces:
                sq1, sq2 = highlights[0], highlights[1]
                prev = self.latest_state.pieces
                curr = pieces_dict
                if sq1 in prev and sq1 not in curr:
                    last_move = f"{sq1}{sq2}"
                elif sq2 in prev and sq2 not in curr:
                    last_move = f"{sq2}{sq1}"
                else:
                    last_move = f"{sq1}{sq2}"
            else:
                last_move = f"{highlights[0]}{highlights[1]}"

        return DOMGameState(
            fen=fen,
            pieces=pieces_dict,
            site=site,
            orientation=orientation,
            highlights=highlights,
            last_move=last_move,
            timestamp=recv_time,
            latency_ms=latency_ms,
        )

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcasts a JSON message to all connected browser tabs."""
        if not self._connected_clients:
            return
        payload = json.dumps(message)
        tasks = [client.send(payload) for client in list(self._connected_clients)]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self):
        """Starts WebSocket server."""
        self._server = await serve(self._handle_client, self.host, self.port)
        self._is_running = True
        logger.success(f"DOM Bridge WebSocket server listening on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stops WebSocket server and closes all active client connections."""
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._connected_clients.clear()
        logger.info("DOM Bridge WebSocket server stopped.")


class ThreadedDOMBridgeServer:
    """
    Synchronous wrapper managing DOMBridgeServer inside a background daemon thread.
    Enables non-blocking integration with synchronous game loops and tests.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_update_callback: Optional[Callable[[DOMGameState], None]] = None,
    ):
        self.server = DOMBridgeServer(host, port, on_update_callback)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()

    @property
    def client_count(self) -> int:
        return self.server.client_count

    @property
    def latest_state(self) -> Optional[DOMGameState]:
        return self.server.latest_state

    def broadcast(self, message: Dict[str, Any]):
        """Thread-safe synchronous broadcast to all connected browser tabs."""
        if not self._loop or not self.server.is_running:
            return
        asyncio.run_coroutine_threadsafe(self.server.broadcast(message), self._loop)

    def start(self, timeout: float = 3.0):
        """Starts server in background thread and waits until listening."""
        if self._thread and self._thread.is_alive():
            return

        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=timeout):
            raise TimeoutError("DOM Bridge server failed to start within timeout.")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _starter():
            await self.server.start()
            self._ready_event.set()

        self._loop.run_until_complete(_starter())
        self._loop.run_forever()

    def stop(self, timeout: float = 3.0):
        """Gracefully terminates background server and loop."""
        if not self._loop or not self.server.is_running:
            return

        async def _stopper():
            await self.server.stop()

        future = asyncio.run_coroutine_threadsafe(_stopper(), self._loop)
        try:
            future.result(timeout=timeout)
        except Exception:
            pass

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._loop = None
