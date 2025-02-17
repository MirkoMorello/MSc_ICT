import socket
import struct
import numpy as np
import base64
import json
import threading
import time

def handle_client(conn, addr):
    print(f"Connected by {addr}")
    next_state = "WAKEWORD"
    try:
        while True:
            raw_msglen = recvall(conn, 4)
            if not raw_msglen:
                break
            msglen = struct.unpack('!I', raw_msglen)[0]
            audio_data_bytes = recvall(conn, msglen)
            if not audio_data_bytes:
                break
            print(f"Received audio data: {len(audio_data_bytes)} bytes")
            audio_data_np = np.frombuffer(audio_data_bytes, dtype=np.float32)
            print(f"Audio data as numpy array: {audio_data_np.shape}")
            if next_state == "WAKEWORD":
                if np.random.rand() < 0.5:
                    print("Wake word detected (simulated)")
                    next_state = "VAD"
                    response_audio = generate_beep(440, 0.1, 16000)
                else:
                    print("Wake word NOT detected (simulated)")
                    response_audio = None
            elif next_state == "VAD":
                avg_amplitude = np.mean(np.abs(audio_data_np))
                if avg_amplitude > 0.05:
                    print("Speech detected (simulated)")
                    if np.random.rand() < 0.8:
                        response_audio = generate_sine_wave(1000, 1.0, 16000)
                        next_state = "WAKEWORD"
                    else:
                        print("Simulated end of turn / error.")
                        response_audio = generate_sine_wave(500, 0.3, 16000)
                        next_state = "WAKEWORD"
                else:
                    print("Silence detected (simulated)")
                    response_audio = None
            else:
                print("Invalid State!")
                return
            if response_audio is not None:
                audio_base64 = base64.b64encode(response_audio.tobytes()).decode()
            else:
                audio_base64 = ""
            response = {
                "audio": audio_base64,
                "next_state": next_state,
            }
            response_json = json.dumps(response).encode()
            conn.sendall(struct.pack("!I", len(response_json)))
            conn.sendall(response_json)
    except (ConnectionResetError, BrokenPipeError) as e:
        print(f"Client disconnected: {e}")
    except struct.error as e:
        print(f"Struct error: {e}")
    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def generate_sine_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

def generate_beep(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

def start_server(host='0.0.0.0', port=8080):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Fake server listening on {host}:{port}")
    try:
        while True:
            conn, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()
