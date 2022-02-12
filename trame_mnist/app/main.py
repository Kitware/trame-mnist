# import os
# os.environ["WSLINK_HEARBEAT"] = "0"
from trame import setup_dev
from . import controller, ui


def start_server():
    setup_dev(ui)
    ui.layout.start()


def start_desktop():
    ui.layout.start_desktop_window()


def main():
    controller.on_start()
    start_server()


if __name__ == "__main__":
    main()
