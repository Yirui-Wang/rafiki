import pprint
import time
import requests
import traceback
import os

from rafiki.client import Client
from rafiki.constants import TaskType, UserType, BudgetType, TrainJobStatus, InferenceJobStatus, ModelDependency

RAFIKI_HOST = 'localhost'
ADMIN_PORT = 3000
ADMIN_WEB_PORT = 3001
SUPERADMIN_EMAIL = 'superadmin@rafiki'
MODEL_DEVELOPER_EMAIL = 'model_developer@rafiki'
APP_DEVELOPER_EMAIL = 'app_developer@rafiki'
USER_PASSWORD = 'rafiki'
ENABLE_GPU = int(os.environ.get('ENABLE_GPU', 0))

def create_user(client, email, password, user_type):
    try:
        return client.create_user(email, password, user_type)
    except:
        # print(traceback.format_exc())
        print('Failed to create user "{}" - maybe it already exists?'.format(email))

def create_model(client, name, task, model_file_path, model_class, dependencies):
    try:
        return client.create_model(name, task, model_file_path, model_class, dependencies=dependencies)
    except:
        # print(traceback.format_exc())
        print('Failed to create model "{}" - maybe it already exists?'.format(name))

def create_train_job(client, app, task, train_dataset_uri, test_dataset_uri, enable_gpu=0):
    budget = {
        BudgetType.MODEL_TRIAL_COUNT: 2,
        BudgetType.ENABLE_GPU: enable_gpu
    }

    train_job = client.create_train_job(app, task, train_dataset_uri, test_dataset_uri, budget=budget)

    app = train_job.get('app')
    app_version = train_job.get('app_version')
    train_job_web_url = 'http://{}:{}/train-jobs/{}/{}'.format(RAFIKI_HOST, ADMIN_WEB_PORT, app, app_version)
    return (train_job, train_job_web_url)

def wait_until_train_job_has_completed(client, app):
    while True:
        time.sleep(10)
        try:
            train_job = client.get_train_job(app)
            status = train_job.get('status')
            if status == TrainJobStatus.COMPLETED:
                # Train job completed!
                return True
            elif status != TrainJobStatus.RUNNING:
                # Train job has either errored or been stopped
                return False
            else:
                continue
        except:
            pass

# Returns `predictor_host` of inference job
def wait_until_inference_job_is_running(client, app):
    while True:
        time.sleep(10)
        try:
            inference_job = client.get_running_inference_job(app)
            status = inference_job.get('status')
            if status  == InferenceJobStatus.RUNNING:
                return inference_job.get('predictor_host')
            elif status in [InferenceJobStatus.ERRORED, InferenceJobStatus.STOPPED]:
                # Inference job has either errored or been stopped
                return False
            else:
                continue

        except:
            pass

def make_predictions(client, predictor_host, queries):
    predictions = []

    for query in queries:
        res = requests.post(
            url='http://{}/predict'.format(predictor_host),
            json={ 'query': query }
        )

        if res.status_code != 200:
            raise Exception(res.text)

        predictions.append(res.json()['prediction'])

    return predictions
    
if __name__ == '__main__':
    app = 'fashion_mnist_app'
    task = TaskType.IMAGE_CLASSIFICATION
    train_dataset_uri = 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_train.zip?raw=true'
    test_dataset_uri = 'https://github.com/nginyc/rafiki-datasets/blob/master/fashion_mnist/fashion_mnist_for_image_classification_test.zip?raw=true'
    queries = [
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], 
        [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], 
        [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], 
        [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], 
        [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], 
        [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], 
        [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], 
        [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], 
        [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], 
        [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ]

    client = Client(admin_host=RAFIKI_HOST, admin_port=ADMIN_PORT)
    client.login(email=SUPERADMIN_EMAIL, password=USER_PASSWORD)

    print('Creating model developer in Rafiki...')
    create_user(client, MODEL_DEVELOPER_EMAIL, USER_PASSWORD, UserType.MODEL_DEVELOPER)

    print('Creating app developer in Rafiki...')
    create_user(client, APP_DEVELOPER_EMAIL, USER_PASSWORD, UserType.APP_DEVELOPER)

    print('Logging in as model developer...')
    client.login(email=MODEL_DEVELOPER_EMAIL, password=USER_PASSWORD)

    print('Adding models to Rafiki...') 
    create_model(client, 'TfFeedForward', task, 'examples/models/image_classification/TfFeedForward.py', \
                'TfFeedForward', dependencies={ ModelDependency.TENSORFLOW: '1.12.0' })
    create_model(client, 'SkDt', task, 'examples/models/image_classification/SkDt.py', \
                'SkDt', dependencies={ ModelDependency.SCIKIT_LEARN: '0.20.0' })

    print('Logging in as app developer...')
    client.login(email=APP_DEVELOPER_EMAIL, password=USER_PASSWORD)

    print('Creating train job for app "{}" on Rafiki...'.format(app)) 
    (train_job, train_job_web_url) = create_train_job(client, app, task, train_dataset_uri, \
                                                    test_dataset_uri, enable_gpu=ENABLE_GPU)
    pprint.pprint(train_job)

    print('Waiting for train job to complete...')
    print('You can view the status of the train job at {}'.format(train_job_web_url))
    print('Login as an app developer with email "{}" and password "{}"'.format(APP_DEVELOPER_EMAIL, USER_PASSWORD)) 
    print('This might take a few minutes')
    result = wait_until_train_job_has_completed(client, app)
    if not result: raise Exception('Train job has errored or stopped')
    print('Train job has been completed!')

    print('Listing best trials of latest train job for app "{}"...'.format(app))
    pprint.pprint(client.get_best_trials_of_train_job(app))

    print('Creating inference job for app "{}" on Rafiki...'.format(app))
    pprint.pprint(client.create_inference_job(app))

    print('Waiting for inference job to be running...')
    predictor_host = wait_until_inference_job_is_running(client, app)
    if not predictor_host: raise Exception('Inference job has errored or stopped')
    print('Inference job is running!')

    print('Making predictions for queries:')
    print(queries)
    predictions = make_predictions(client, predictor_host, queries)
    print('Predictions are:')
    print(predictions)

    print('Stopping inference job...')
    pprint.pprint(client.stop_inference_job(app))

