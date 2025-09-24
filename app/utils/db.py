import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List
import numpy as np
import torch
import json

DB_PATH = Path(__file__).parent.parent / 'data' / 'app.db'

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )          
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            chunk_index INTEGER,
            chunk_text TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def save_interaction(document: str, question: str, answer: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO history (document, question, answer, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (document, question, answer, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def search_answer_by_filename(document: str, question: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT answer
        FROM history
        WHERE document = ? and question = ?
        ORDER BY id DESC
        LIMIT 1
    """, (document, question)) 
    rows = cursor.fetchall()
    conn.commit()
    conn.close()
    return rows
     
def list_interactions_by_filename(filename, limit=20) -> List[any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT document, question, answer, timestamp 
        FROM history
        WHERE document = ?
        ORDER BY id DESC
        LIMIT ?    
    """, (filename, limit,))
    rows = cursor.fetchall()
    conn.commit()
    conn.close()
    return rows

def add_embeddings(filename, chunks, embeddings):
    conn = get_connection()
    cursor = conn.cursor()
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            emb_json = json.dumps(emb.tolist())
            cursor.execute("""
                INSERT INTO documents (filename, chunk_index, chunk_text, embedding)
                VALUES (?, ?, ?, ?)
            """, (filename, i, chunk, emb_json))
    conn.commit()
    conn.close()

def get_embeddings(filename):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_text, embedding FROM documents WHERE filename = ? ORDER BY chunk_index", (filename,))
    rows = cursor.fetchall()
    
    if not rows:
        return None, None
    
    chunks = [row[0] for row in rows]
    embeddings = torch.tensor([json.loads(row[1]) for row in rows])  # volta como tensor 2D

    conn.commit()
    conn.close()

    return (chunks, embeddings)
