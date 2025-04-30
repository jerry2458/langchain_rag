from sqlalchemy import create_engine, text
import pandas as pd
import os

# 1. DB 접속 설정
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "jerry",
    "password": "Sweet0326!!",
    "database": "test",
    "port": 3306
}

# 2. SQLAlchemy 엔진 생성
engine = create_engine(
    "mysql+pymysql://root:Sweet0326!!@127.0.0.1:3306/test",
    pool_pre_ping=True
)

# 3. 총 행 수 조회 함수
def get_data_count():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM traffic_info_2"))
        count = result.scalar()
    return count

# 4. 데이터 조회 함수
def get_data(start_date=None, end_date=None, limit=500):
    if start_date and end_date:
        query = text("""
            SELECT * FROM traffic_info_2
            WHERE DATE(collDate) BETWEEN :start_date AND :end_date
            ORDER BY collDate DESC
        """)
        params = {"start_date": start_date, "end_date": end_date}
    else:
        query = text(f"""
            SELECT * FROM traffic_info_2
            ORDER BY collDate DESC
            LIMIT {limit}
        """)
        params = {}

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    return df
