import base64
import pickle

from sklearn import svm

from rafiki.constants import TaskType
from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class


class SkSvr(BaseModel):
    def get_knob_config(self):
        return {
            'knobs': {
                'max_iter': {
                    'type': 'int',
                    'range': [100, 100000]
                },
                'kernel': {
                    'type': 'string',
                    'values': ['linear', 'poly', 'rbf', 'sigmoid']
                },
                'gamma': {
                    'type': 'string',
                    'values': ['auto', 'scale']
                },
                'C': {
                    'type': 'float_exp',
                    'range': [1e-2, 1e1]
                }
            }
        }

    def init(self, knobs):
        self._max_iter = knobs.get('max_iter')
        self._kernel = knobs.get('kernel')
        self._gamma = knobs.get('gamma')
        self._C = knobs.get('C')
        self._reg = svm.SVR(
            max_iter=self._max_iter,
            kernel=self._kernel,
            gamma=self._gamma,
            C=self._C
        )

    def train(self, dataset_uri):
        dataset = self.utils.load_dataset_of_table(dataset_uri)
        x = dataset.get_x()
        y = dataset.get_y()
        self._reg.fit(x, y)

    def evaluate(self, dataset_uri):
        """
        In the train() and evaluate(), dataset has been loaded for more than once. This can be optimized in the future?
        :param dataset_uri:
        :return:
        """
        dataset = self.utils.load_dataset_of_table(dataset_uri)
        x = dataset.get_x()
        y = dataset.get_y()
        return self._reg.score(x, y)

    def predict(self, queries):
        return self._reg.predict(queries)

    def dump_parameters(self):
        params = {}
        reg_bytes = pickle.dumps(self._reg)
        reg_base64 = base64.b64encode(reg_bytes).decode('utf-8')
        params['reg_base64'] = reg_base64
        return params

    def load_parameters(self, params):
        reg_base64 = params.get('reg_base64', None)
        if reg_base64 is None:
            raise InvalidModelParamsException()

        reg_bytes = base64.b64decode(reg_base64.encode('utf-8'))
        self._reg = pickle.loads(reg_bytes)

    def destroy(self):
        pass


if __name__ == '__main__':
    # model = SkLasso()
    # model._reg = Lasso(0.1)
    # model.train('data/home_rentals_train.zip')
    # print(model.evaluate('data/home_rentals_test.zip'))
    # queries = [[3358, 2, 1, 743, 10, 3230, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #            [3359, 1, 1, 533, 10, 1903, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #            [3360, 3, 2, 1186, 62, 4437, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    #            ]
    # print(model.predict(queries))
    test_model_class(
        model_file_path=__file__,
        model_class='SkSvr',
        task=TaskType.TABLE_REGRESSION,
        dependencies={},
        train_dataset_uri='data/home_rentals_train.zip',
        test_dataset_uri='data/home_rentals_test.zip',
        queries=[
            [3358, 2, 1, 743, 10, 3230, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [3359, 1, 1, 533, 10, 1903, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3360, 3, 2, 1186, 62, 4437, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        ]
    )
