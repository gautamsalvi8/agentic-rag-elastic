import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

password = os.getenv("ELASTIC_PASSWORD")

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", password)
)

print(es.info())
