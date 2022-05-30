from trame.app import get_server, jupyter
from trame_mnist.app import engine, ui


def show(server=None, **kwargs):
    """Run and display the trame application in jupyter's event loop
    The kwargs are forwarded to IPython.display.IFrame()
    """
    if server is None:
        server = get_server()

    if isinstance(server, str):
        server = get_server(server)

    # Disable logging
    import logging

    engine_logger = logging.getLogger("trame_mnist.app.engine")
    engine_logger.setLevel(logging.WARNING)

    # Initilize app
    engine.initialize(server)
    ui.initialize(server)

    # Show as cell result
    jupyter.show(server, **kwargs)
