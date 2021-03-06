.. _`quick-start`:

Quick Start
====================================================================

.. contents:: Table of Contents

.. note::

    If you're a *Model Developer* just looking to contribute models to a running instance of Rafiki, refer to :ref:`quickstart-model-developers`.

.. note::

    If you're an *Application Developer* just looking to train and deploy models on a running instance of Rafiki, refer to :ref:`quickstart-app-developers`.

.. note::

    If you're an *Application User* just looking to make predictions to deployed models on a running instance of Rafiki, refer to :ref:`quickstart-app-users`.


This guide assumes you have deployed your an empty instance of Rafiki and you want to do a *full* train-inference flow, 
including preparation of dataset and adding of models to Rafiki. Below, the sequence of examples submit the 
`Fashion MNIST dataset <https://github.com/zalandoresearch/fashion-mnist>`_ for training and inference. 
Alternatively, after installing Rafiki Client's dependencies, you can run `./examples/scripts/client-quickstart.py <https://github.com/nginyc/rafiki/blob/master/examples/scripts/client-quickstart.py>`_.

.. note::

    If you haven't set up Rafiki on your local machine, refer to :ref:`quick-setup` before continuing.


Installing the Client
--------------------------------------------------------------------

.. include:: ./client-installation.include.rst


Initializing the Client
--------------------------------------------------------------------

Example:

    .. code-block:: python

        from rafiki.client import Client
        client = Client(admin_host='localhost', admin_port=3000)
        client.login(email='superadmin@rafiki', password='rafiki')


.. seealso:: :meth:`rafiki.client.Client.login`

Creating models
--------------------------------------------------------------------

.. include:: ./client-create-models.include.rst


Listing models by task
--------------------------------------------------------------------

.. include:: ./client-list-models.include.rst


Creating a train job
--------------------------------------------------------------------

.. include:: ./client-create-train-job.include.rst


Listing train jobs of an app
--------------------------------------------------------------------

.. include:: ./client-list-train-jobs.include.rst


Creating an inference job with the latest train job for an app
--------------------------------------------------------------------

.. include:: ./client-create-inference-job.include.rst


Listing inference jobs of an app
--------------------------------------------------------------------

.. include:: ./client-list-inference-jobs.include.rst


Making predictions
--------------------------------------------------------------------

.. include:: ./making-predictions.include.rst


Stopping a running inference job
--------------------------------------------------------------------

.. include:: ./client-stop-inference-job.include.rst