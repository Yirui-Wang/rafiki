import pprint

from examples.scripts.client_quickstart import RAFIKI_HOST, ADMIN_PORT, APP_DEVELOPER_EMAIL, USER_PASSWORD, \
    create_train_job, wait_until_train_job_has_completed, wait_until_inference_job_is_running
from rafiki.client import Client
from rafiki.constants import TaskType

if __name__ == '__main__':
    app = 'home_rentals_regression'
    task = TaskType.TABLE_REGRESSION
    client = Client(admin_host=RAFIKI_HOST, admin_port=ADMIN_PORT)
    train_dataset_uri = 'https://raw.githubusercontent.com/Yirui-Wang/rafiki/master/data/home_rentals_train.zip'
    test_dataset_uri = 'https://raw.githubusercontent.com/Yirui-Wang/rafiki/master/data/home_rentals_test.zip'

    print('Logging in as app developer...')
    client.login(email=APP_DEVELOPER_EMAIL, password=USER_PASSWORD)

    print('Creating train job for app "{}" on Rafiki...'.format(app))
    (train_job, train_job_web_url) = create_train_job(client, app, task, train_dataset_uri, test_dataset_uri)
    pprint.pprint(train_job)

    print('Waiting for train job to complete...')
    print('You can view the status of the train job at {}'.format(train_job_web_url))
    print('Login as an app developer with email "{}" and password "{}"'.format(APP_DEVELOPER_EMAIL, USER_PASSWORD))
    print('This might take a few minutes')
    result = wait_until_train_job_has_completed(client, app)
    if not result:
        raise Exception('Train job has errored or stopped')
    print('Train job has been completed!')

    print('Listing best trials of latest train job for app "{}"...'.format(app))
    pprint.pprint(client.get_best_trials_of_train_job(app))

    print('Creating inference job for app "{}" on Rafiki...'.format(app))
    pprint.pprint(client.create_inference_job(app))

    print('Waiting for inference job to be running...')
    predictor_host = wait_until_inference_job_is_running(client, app)
    if not predictor_host:
        raise Exception('Inference job has errored or stopped')
    print('Inference job is running!')
