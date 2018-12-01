import base64
import pickle

from sklearn.linear_model import Lasso

from rafiki.constants import ModelDependency, TaskType
from rafiki.model import BaseModel, InvalidModelParamsException, \
    test_model_class


class SkLasso(BaseModel):
    def get_knob_config(self):
        return {
            'knobs': {
                'alpha': {
                    'type': 'float_exp',
                    'range': [1e-3, 1e5]
                }
            }
        }

    def init(self, knobs):
        self._alpha = knobs.get('alpha')
        self._reg = Lasso(self._alpha)

    def train(self, dataset_uri):
        table = self.utils.load_dataset_of_table(dataset_uri).get_table()
        x = table.iloc[:, 0:table.shape[1] - 1]
        y = table.iloc[:, table.shape[1] - 1]
        self._reg.fit(x, y)

    def evaluate(self, dataset_uri):
        """
        In the train() and evaluate(), dataset has been loaded for more than
        once. This can be optimized in the future?
        :param dataset_uri:
        :return:
        """
        table = self.utils.load_dataset_of_table(dataset_uri).get_table()
        x = table.iloc[:, 0:table.shape[1]-1]
        y = table.iloc[:, table.shape[1]-1]
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
    test_model_class(
        model_file_path=__file__,
        model_class='SkLasso',
        task=TaskType.TABLE_REGRESSION,
        dependencies={
        },
        train_dataset_uri='data/home_rentals_train.zip',
        test_dataset_uri='data/home_rentals_test.zip',
        queries=[
            [2, 1, 605, 'good', 10, 3120, 'west_welmwood', 3120],
            [2, 1, 517, 'poor', 17, 2916, 'northwest', 2916],
            [1, 1, 515, 'great', 0, 2558, 'east_elmwood', 2558]
        ]
    )