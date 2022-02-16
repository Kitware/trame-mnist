================
Trame MNIST
================

Example application using **trame** for exploring MNIST dataset in the context of AI training and XAI thanks to **XAITK**.

* Free software: BSD License
* `XAITK Saliency with MNIST <https://github.com/XAITK/xaitk-saliency/blob/master/examples/MNIST_scikit_saliency.ipynb>`_
* `XAI Discovery Platform | MNIST Sample Data <http://obereed.net:3838/mnist/>`_

Installing
----------

For the Python layer it is recommended to use `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to properly install the various ML packages.

conda setup on macOS
^^^^^^^^^^^^^^^^^^^^^

Go to `conda documentation <https://docs.conda.io/en/latest/miniconda.html>`_ for your OS

.. code-block:: console

    brew install miniforge
    conda init zsh

venv setup for AI
^^^^^^^^^^^^^^^^^^

.. code-block:: console

    # Needed in order to get py3.9 with lzma
    # PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.9.9

    conda create --name trame-mnist python=3.9
    conda activate trame-mnist
    conda install "pytorch==1.9.1" "torchvision==0.10.1" -c pytorch
    conda install scipy "scikit-learn==0.24.2" "scikit-image==0.18.3" -c conda-forge

    # For development when inside repo
    pip install -e .

    # For testing (no need to clone repo)
    pip install trame-mnist



Running the application
------------------------

.. code-block:: console

    conda activate trame-mnist
    trame-mnist

If **cuda** is available, the application will use your GPU, but you can also force the usage of your cpu by adding to your command line the following argument: **--cpu**

|image_1| |image_2| |image_3|

.. |image_1| image:: https://github.com/Kitware/trame-mnist/raw/master/gallery/trame-mnist-02.jpg
  :width: 32%
.. |image_2| image:: https://github.com/Kitware/trame-mnist/raw/master/gallery/trame-mnist-03.jpg
  :width: 32%
.. |image_3| image:: https://github.com/Kitware/trame-mnist/raw/master/gallery/trame-mnist-04.jpg
  :width: 32%

License
--------

**trame-mnist** is distributed under the OSI-approved BSD 3-clause License.