from muselsl import stream, list_muses

import signal
import subprocess
import time

MAC_ADDRESS = None


def get_default_muse(avail_muses):
    print("No Muse device configured.")

    if not avail_muses:
        print("Make sure your Muse is on and in Pairing Mode.")
        exit()

    for i, muse in enumerate(avail_muses):
        name = muse["name"]
        addr = muse["address"]
        print(F"[{i}]: {name} ({addr})")
    
    selected_muse = -1
    while not(0 <= selected_muse <= len(avail_muses) - 1):
        try:
            selected_muse = int(input(f"Select your Muse from the above list (0 - {len(avail_muses)-1}): "))
        except ValueError:
            pass

    mac_addr = avail_muses[selected_muse]["address"]

    with open("mac-addr.txt", "w") as outfile:
        outfile.write(mac_addr)

    return mac_addr

def start_stream(muses):
    global MAC_ADDRESS

    if MAC_ADDRESS is None:
        try:
            with open("mac-addr.txt", "r") as infile:
                MAC_ADDRESS = infile.readline()
                print(f"Attempting to connect to previously configured device at {MAC_ADDRESS}")
        except FileNotFoundError:
            MAC_ADDRESS = get_default_muse(muses)

    muse_conn = None
    for m in avail_muses:
        if m["address"] == MAC_ADDRESS:
            muse_conn = m

    if muse_conn is None:
        print(f"Muse with MAC {MAC_ADDRESS} not found.")
        MAC_ADDRESS = get_default_muse(muses)
        start_stream(muses)

    p = subprocess.Popen(f"exec muselsl stream --address {MAC_ADDRESS}", stdout=subprocess.PIPE, shell=True)

    print(f"Stream started with PID {p.pid}")

    return p

if __name__ == "__main__":
    try:
        avail_muses = list_muses()

        if not avail_muses:
            exit(1)

        stream_process = start_stream(avail_muses)

        while True:
            pass
    finally:
        stream_process.send_signal(signal.SIGINT)

