import json
import psycopg2
from app import config

SCHEMA = f'"{config.DB_SCHEMA}"'


def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        sslmode=config.DB_SSLMODE,
    )


# ── key_api_config ──────────────────────────────────────────────────────────

def get_api_config(key: str, api_name: str) -> dict | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT api_config
            FROM {SCHEMA}.key_api_config
            WHERE key = %s AND api_name = %s
              AND LOWER(api_config_status) = 'active'
            """,
            (key, api_name),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        raw = row[0]
        return raw if isinstance(raw, dict) else json.loads(raw)
    finally:
        conn.close()


# ── classifier_config ───────────────────────────────────────────────────────

def get_classifier_config(key: str, classifier_config_model: str) -> dict | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT classifier_config_json
            FROM {SCHEMA}.classifier_config
            WHERE key = %s AND classifier_config_model = %s AND LOWER(status) = 'active'
            """,
            (key, classifier_config_model),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        raw = row[0]
        return raw if isinstance(raw, dict) else json.loads(raw)
    finally:
        conn.close()


# ── classifier_config_details ───────────────────────────────────────────────

def get_classifier_details(key: str, classifier_config_model: str) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT classifier, vector_data
            FROM {SCHEMA}.classifier_config_details
            WHERE key = %s AND classifier_config_model = %s
              AND vector_data IS NOT NULL AND LOWER(status) = 'active'
            """,
            (key, classifier_config_model),
        )
        rows = cur.fetchall()
        cur.close()
        return [{"classifier": r[0], "vector_data": r[1]} for r in rows]
    finally:
        conn.close()


def classifier_detail_exists(
    key: str, classifier_config_model: str, classifier: str
) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT 1 FROM {SCHEMA}.classifier_config_details
            WHERE key = %s AND classifier_config_model = %s AND classifier = %s
            """,
            (key, classifier_config_model, classifier),
        )
        exists = cur.fetchone() is not None
        cur.close()
        return exists
    finally:
        conn.close()


def insert_classifier_detail(
    key: str,
    classifier_config_model: str,
    classifier: str,
    text_data: str,
    vector_data: str,
    created_by: str,
):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.classifier_config_details
                (key, classifier_config_model, classifier, text_data, vector_data,
                 created_on, created_by, status)
            VALUES (%s, %s, %s, %s, %s, DATE_TRUNC('second', NOW()), %s, 'Active')
            """,
            (key, classifier_config_model, classifier, text_data, vector_data, created_by),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def update_classifier_detail(
    key: str,
    classifier_config_model: str,
    classifier: str,
    text_data: str,
    vector_data: str,
    modified_by: str,
):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE {SCHEMA}.classifier_config_details
            SET text_data = %s, vector_data = %s, modified_on = DATE_TRUNC('second', NOW()), modified_by = %s
            WHERE key = %s AND classifier_config_model = %s AND classifier = %s
            """,
            (text_data, vector_data, modified_by, key, classifier_config_model, classifier),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()
