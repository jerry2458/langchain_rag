import pymysql
import pandas as pd
import os


DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PW"),
    "database": os.getenv("DB"),
    "port": 3306,
    "cursorclass": pymysql.cursors.DictCursor
}

def get_connection():
    return pymysql.connect(**DB_CONFIG)

def get_data_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM traffic_info_2")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_data(start_date=None, end_date=None, limit=500):
    conn = get_connection()

    if start_date and end_date:
        query = """
            SELECT * FROM traffic_info_2
            WHERE DATE(collDate) BETWEEN ? AND ?
            ORDER BY collDate DESC
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    else:
        query = f"""
            SELECT * FROM traffic_info_2
            ORDER BY collDate DESC
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df
