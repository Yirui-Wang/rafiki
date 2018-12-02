from examples.scripts.client_quickstart import RAFIKI_HOST, ADMIN_PORT, USER_PASSWORD, create_user, \
    MODEL_DEVELOPER_EMAIL, APP_DEVELOPER_EMAIL
from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import UserType

if __name__ == '__main__':
    client = Client(admin_host=RAFIKI_HOST, admin_port=ADMIN_PORT)
    client.login(email=SUPERADMIN_EMAIL, password=USER_PASSWORD)

    print('Creating model developer in Rafiki...')
    create_user(client, MODEL_DEVELOPER_EMAIL, USER_PASSWORD, UserType.MODEL_DEVELOPER)

    print('Creating app developer in Rafiki...')
    create_user(client, APP_DEVELOPER_EMAIL, USER_PASSWORD, UserType.APP_DEVELOPER)
