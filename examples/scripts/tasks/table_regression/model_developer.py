from examples.scripts.client_quickstart import RAFIKI_HOST, ADMIN_PORT, USER_PASSWORD, MODEL_DEVELOPER_EMAIL, \
    create_model
from rafiki.client import Client
from rafiki.constants import TaskType, ModelDependency

if __name__ == '__main__':
    app = 'home_rentals_regression'
    task = TaskType.TABLE_REGRESSION

    client = Client(admin_host=RAFIKI_HOST, admin_port=ADMIN_PORT)

    print('Logging in as model developer...')
    client.login(email=MODEL_DEVELOPER_EMAIL, password=USER_PASSWORD)

    print('Adding models to Rafiki...')
    create_model(client, 'SkLasso', task, 'examples/models/table_regression/SkLasso.py', 'SkLasso', dependencies={
        ModelDependency.SCIKIT_LEARN: '0.20.0'})
