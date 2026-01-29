"""
Mouse Server - Runs on host (Windows)
Lightweight socket server that receives coordinates from Docker
and controls the mouse using pyautogui.
"""

import socket
import json
import threading

import pyautogui


# Disable pyautogui fail-safe (move to corner to abort)
# Enable this for safety: pyautogui.FAILSAFE = True
pyautogui.FAILSAFE = True

# Disable pyautogui pause between actions for responsiveness
pyautogui.PAUSE = 0


class MouseServer:
    def __init__(self, host='0.0.0.0', port=5555):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False

        # Get screen resolution
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")

    def handle_client(self, client_socket, address):
        """Handle a single client connection."""
        print(f"Client connected from {address}")

        # Send screen resolution to client
        info = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height
        }
        client_socket.send(json.dumps(info).encode('utf-8'))

        buffer = ""

        try:
            while self.running:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break

                buffer += data

                # Process complete messages (newline-delimited JSON)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            command = json.loads(line)
                            self.process_command(command)
                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON: {e}")

        except Exception as e:
            print(f"Client error: {e}")

        finally:
            client_socket.close()
            print(f"Client {address} disconnected")

    def process_command(self, command):
        """Process a mouse command."""
        action = command.get('action')

        if action == 'move':
            x = command.get('x', 0)
            y = command.get('y', 0)

            # Clamp to screen bounds
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))

            pyautogui.moveTo(x, y, _pause=False)

        elif action == 'click':
            button = command.get('button', 'left')
            pyautogui.click(button=button, _pause=False)

        elif action == 'double_click':
            pyautogui.doubleClick(_pause=False)

        elif action == 'right_click':
            pyautogui.rightClick(_pause=False)

        elif action == 'scroll':
            amount = command.get('amount', 0)
            pyautogui.scroll(amount, _pause=False)

        elif action == 'drag':
            x = command.get('x', 0)
            y = command.get('y', 0)
            pyautogui.drag(x, y, _pause=False)

    def start(self):
        """Start the mouse server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True

            print(f"Mouse server listening on {self.host}:{self.port}")
            print("Waiting for hand tracker connection...")
            print("Press Ctrl+C to stop")
            print("")
            print("Safety note: Move mouse to corner of screen to abort (failsafe)")

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        print(f"Accept error: {e}")

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            self.stop()

    def stop(self):
        """Stop the mouse server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Mouse server stopped")


if __name__ == '__main__':
    server = MouseServer()
    server.start()
