
To create a model, you will need to submit a model class that extends :class:`rafiki.model.BaseModel` in a single Python file,
where the model's implementation conforms to a specific task (see :ref:`tasks`). 

Refer to the parameters of :meth:`rafiki.client.Client.create_model` for configuring how your model runs on Rafiki,
and refer to :ref:`creating-models` to understand more about how to write & test models for Rafiki.

Examples:

    .. code-block:: python

        client.create_model(
            name='TfFeedForward',
            task='IMAGE_CLASSIFICATION',
            model_file_path='examples/models/image_classification/TfFeedForward.py',
            model_class='TfFeedForward',
            dependencies={ 'tensorflow': '1.12.0' }
        )

        client.create_model(
            name='SkDt',
            task='IMAGE_CLASSIFICATION',
            model_file_path='examples/models/image_classification/SkDt.py',
            model_class='SkDt',
            dependencies={ 'scikit-learn': '0.20.0' }
        )

.. seealso:: :meth:`rafiki.client.Client.create_model`
