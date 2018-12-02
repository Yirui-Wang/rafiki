from examples.scripts.client_quickstart import make_predictions, RAFIKI_HOST, ADMIN_PORT
from rafiki.client import Client

if __name__ == '__main__':
    client = Client(admin_host=RAFIKI_HOST, admin_port=ADMIN_PORT)

    print('Making predictions for queries:')
    print(queries)
    predictions = make_predictions(client, predictor_host, queries)
    print('Predictions are:')
    print(predictions)

    print('Stopping inference job...')
    pprint.pprint(client.stop_inference_job(app))
