import json
from functools import partial
import sys
import urllib

from socket import error as SocketError
import websocket

sys.path.insert(1, "../")
from util.common import IFACE, IP, OS, WIFI


def check_connection():
    try:
        urllib.request.urlopen(IP, timeout=0.1)
        return True
    except urllib.request.URLError:
        return False


def connect_to_wifi():
    if OS == "Darwin":
        import objc
        from objc import CWInterface

        objc.loadBundle(
            "CoreWLAN",
            bundle_path="/System/Library/Frameworks/CoreWLAN.framework",
            module_globals=globals(),
        )

        iface = CWInterface.interface()
        networks, error = iface.scanForNetworksWithName_error_(WIFI["network"], None)
        network = networks.anyObject()
        success, error = iface.associateToNetwork_password_error_(
            network, WIFI["password"], None
        )
        if error:
            print(f"[DEBUG] Couldn't connect to wifi '{WIFI['network']}'")
            return False
        else:
            print(f"[DEBUG] Connected to wifi '{WIFI['network']}'")
            return True

    elif OS == "Linux":
        import os

        cmd = "sudo iwlist wlp2s0 scan | grep -ioE 'ssid:\"(.*{}.*)'"
        result = list(os.popen(cmd.format(WIFI["SERVER"])))

        if "Device or resource busy" in result:
            return False
        else:
            ssid_list = [item.lstrip("SSID:").strip('"\n') for item in result]

        for name in ssid_list:
            cmd = (
                f"nmcli d wifi connect {name} "
                f"password {WIFI['password']} "
                f"iface {IFACE}"
            )
            if os.system(cmd) != 0:
                print(f"[DEBUG] Couldn't connect to wifi '{WIFI['network']}'")
                return False
            else:
                print(f"[DEBUG] Connected to wifi '{WIFI['network']}'")
                return True

    else:
        print(f"[DEBUG] OS '{OS}' not supported for auto wifi connection")
        return False


class WebSocket:
    def __init__(self, ip, id, facenet):
        self.ip = ip
        self.id = id
        self.facenet = facenet

    def on_message(self, ws, message):
        message = json.loads(message)
        print("######### SENDING #########")
        print(message)
        print("###########################\n")

    def on_error(self, ws, error):
        print("######### ERROR #########")
        print(error)
        print("###########################\n")

    def on_close(self, ws, close_status_code, close_msg):
        print("######### CLOSING #########")
        print(close_status_code, "::", close_msg)
        print("###########################\n")

    def connect(self, **facenet_kwargs):
        def on_open(ws, **kwargs):
            ws.send(json.dumps({"id": str(self.id)}))
            self.facenet.real_time_recognize(**kwargs, socket=ws)

        websocket.enableTrace(True)
        ws = websocket.WebSocketApp(
            f"ws://{self.ip}/v1/nano",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        ws.on_open = partial(on_open, **facenet_kwargs)
        ws.run_forever()

    def connect_v2(self, **facenet_kwargs):
        def on_open(ws, **kwargs):
            ws.send(json.dumps({"id": str(self.id)}))
            self.facenet.real_time_recognize(**kwargs, socket=ws)

        def new_send(self, *args, **kwargs):
            self.send(*args, **kwargs)
            if not check_connection():
                raise SocketError

        websocket.enableTrace(True)
        ws = websocket.WebSocketApp(
            f"ws://{self.ip}/v1/nano",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        ws.send = new_send  # monkey-patch
        ws.on_open = partial(on_open, **facenet_kwargs)
        ws.run_forever()

    @classmethod
    def run(cls, facenet, id=1, **kwargs):
        def _run():
            ws = cls(IP, id, facenet)
            ws.connect_v2(**kwargs)

        i = 0
        while True:
            i += 1
            print(f"------ RESETTING ({i}) ------\n\n\n")
            try:
                _run()
            except SocketError:
                continue
