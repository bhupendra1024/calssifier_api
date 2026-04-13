from __future__ import annotations

import datetime
import json
from zoneinfo import ZoneInfo

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

def get_api_config(key: str, request_model: str) -> dict | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT api_config
            FROM {SCHEMA}.key_api_config
            WHERE key = %s AND request_model = %s
              AND LOWER(api_config_status) = 'active'
            """,
            (key, request_model),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        raw = row[0]
        return raw if isinstance(raw, dict) else json.loads(raw)
    finally:
        conn.close()

def get_key_master(key: str) -> str | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT org_name
            FROM {SCHEMA}.key_master
            WHERE key = %s AND LOWER(key_status) = 'active'
            """,
            (key,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        return row[0]
    finally:
        conn.close()

# ── classifier_config ───────────────────────────────────────────────────────

def get_classifier_config(org_name: str, classifier_config_model: str) -> dict | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT classifier_config_json
            FROM {SCHEMA}.classifier_config
            WHERE org_name = %s AND classifier_config_model = %s AND mode = 'EMBED' AND LOWER(status) = 'active'
            """,
            (org_name, classifier_config_model),
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

def get_classifier_details(org_name: str, classifier_config_model: str) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT classifier, vector_data
            FROM {SCHEMA}.classifier_config_details
            WHERE org_name = %s AND classifier_config_model = %s
              AND vector_data IS NOT NULL AND LOWER(status) = 'active'
            """,
            (org_name, classifier_config_model),
        )
        rows = cur.fetchall()
        cur.close()
        return [{"classifier": r[0], "vector_data": r[1]} for r in rows]
    finally:
        conn.close()


def classifier_detail_exists(
    org_name: str, classifier_config_model: str, classifier: str
) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT 1 FROM {SCHEMA}.classifier_config_details
            WHERE org_name = %s AND classifier_config_model = %s AND classifier = %s
            """,
            (org_name, classifier_config_model, classifier),
        )
        exists = cur.fetchone() is not None
        cur.close()
        return exists
    finally:
        conn.close()


def insert_classifier_detail(
    org_name: str,
    classifier_config_model: str,
    classifier: str,
    text_data: str,
    vector_data: str,
    created_by: str,
):
    conn = get_connection()
    try:
        timestamp = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.classifier_config_details
                (org_name, classifier_config_model, classifier, text_data, vector_data,
                 created_on, created_by, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'Active')
            """,
            (org_name, classifier_config_model, classifier, text_data, vector_data, timestamp, created_by),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def update_classifier_detail(
    org_name: str,
    classifier_config_model: str,
    classifier: str,
    text_data: str,
    vector_data: str,
    modified_by: str,
):
    conn = get_connection()
    try:
        timestamp = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE {SCHEMA}.classifier_config_details
            SET text_data = %s, vector_data = %s, modified_on = %s, modified_by = %s
            WHERE org_name = %s AND classifier_config_model = %s AND classifier = %s
            """,
            (text_data, vector_data, timestamp, modified_by, org_name, classifier_config_model, classifier),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()
