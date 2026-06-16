import os
import pymysql
import chromadb
from dotenv import load_dotenv

load_dotenv()

def get_chroma():
    """ChromaDB 서버 연결"""
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"),
        port=int(os.getenv("CHROMA_PORT"))
    )
    return client.get_collection("topik_collection")

def get_mysql():
    """MySQL 서버 연결"""
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE", "nova_class"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn
