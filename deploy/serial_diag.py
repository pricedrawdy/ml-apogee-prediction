"""
serial_diag.py — Raw serial port diagnostic.

Usage:  python deploy/serial_diag.py /dev/ttyACM0
        python deploy/serial_diag.py /dev/ttyUSB0

Prints every byte received for 30 seconds.
Run this WITHOUT flight_demo.py running.
"""
import sys, time
import serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

print(f"Opening {port} @ {baud} baud ...")
print("(Arduino should reset now — watch for countdown lines)")
print("=" * 50)

ser = serial.Serial(port, baud, timeout=1.0)
time.sleep(0.5)
ser.reset_input_buffer()

deadline = time.time() + 60.0
lines_seen = 0

while time.time() < deadline:
    raw = ser.readline()
    if raw:
        line = raw.decode("utf-8", errors="replace").strip()
        print(repr(line))
        lines_seen += 1
        if lines_seen >= 5 and not any(c.isdigit() for c in line):
            continue  # skip pure comment lines in count
    else:
        print("[timeout - no data for 1s]")

ser.close()
print(f"\nDone. {lines_seen} lines received.")
