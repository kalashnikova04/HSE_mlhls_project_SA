import os
from datetime import datetime
from elasticsearch import Elasticsearch

host = os.getenv('elastic_host')
port = os.getenv('elastic_port')
username = os.getenv('elastic_username')
password = os.getenv('elastic_password')

es = Elasticsearch(
    [f'http://{host}:{port}'],
    http_auth=(username, password)
)

index_name = 'service_logs'

