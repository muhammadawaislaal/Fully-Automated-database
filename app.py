import os
import re
import json
import sqlite3
import pandas as pd
import streamlit as st
import tempfile
from sqlalchemy import create_engine, text, inspect
from datetime import datetime
from openai import OpenAI
import altair as alt
import graphviz

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="🤖 AI Database Agent", layout="wide")

# ----------------- OpenAI Client -----------------
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    st.error("❌ No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets or environment.")
    st.stop()

# ----------------- Paths -----------------
TEMP_DIR = tempfile.gettempdir()
DB_PATH = os.path.join(TEMP_DIR, "ai_db.sqlite")
STATE_PATH = os.path.join(TEMP_DIR, "ai_db_state.json")

# ----------------- Memory Saver -----------------
def save_state():
    state = {
        "schema_text": st.session_state.get("schema_text", ""),
        "last_sql": st.session_state.get("last_sql", ""),
        "history": st.session_state.get("history", [])
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        for k, v in state.items():
            st.session_state[k] = v

# ----------------- Session State Init -----------------
if "engine" not in st.session_state:
    st.session_state.engine = None
if "schema_text" not in st.session_state:
    st.session_state.schema_text = "No database created yet."
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Load saved memory
load_state()

# ----------------- Helpers -----------------
FORBIDDEN = {"DROP", "ALTER DATABASE", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA"}

def make_engine():
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

def get_schema_text(engine) -> str:
    insp = inspect(engine)
    tables = insp.get_table_names()
    lines = []
    for t in tables:
        cols = [f"{c['name']} ({c['type']})" for c in insp.get_columns(t)]
        lines.append(f"📊 {t}: {', '.join(cols)}")
    return "\n\n".join(lines) if lines else "No tables found."

def extract_sql(output: str) -> str:
    if not output:
        return ""
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", output, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return output.strip()

def validate_sql(sql: str, allow_write: bool = True) -> bool:
    if not sql:
        return False
    sql_upper = sql.upper()
    for bad in FORBIDDEN:
        if bad in sql_upper:
            return False
    if not allow_write:
        if sql_upper.startswith(("INSERT", "UPDATE", "DELETE", "ALTER")):
            return False
    return True

def ask_model_to_sql(prompt: str, schema: str, mode: str = "create") -> str:
    system_msg = f"You are an expert SQL assistant. Target dialect: SQLite."
    user_msg = f"""
Database schema:
{schema}

User request: {prompt}

Mode: {mode}
Return only valid SQL without explanation.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=0,
        max_tokens=800
    )
    raw = resp.choices[0].message.content
    return extract_sql(raw)

def run_sql(engine, sql: str):
    try:
        df = pd.read_sql_query(text(sql), engine)
        return df
    except Exception as e:
        return str(e)

def visualize(df: pd.DataFrame):
    if df.empty:
        st.info("No data to visualize.")
        return
    if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
        chart = alt.Chart(df).mark_bar().encode(
            x=df.columns[0],
            y=df.columns[1]
        )
        st.altair_chart(chart, use_container_width=True)
    elif "date" in df.columns[0].lower():
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=df.columns[0],
            y=df.columns[1]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

def render_er_diagram(engine):
    insp = inspect(engine)
    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", fontsize="10")

    for table in insp.get_table_names():
        cols = [f"{c['name']} : {c['type']}" for c in insp.get_columns(table)]
        label = f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
        label += f"<TR><TD BGCOLOR='lightblue'><B>{table}</B></TD></TR>"
        for col in cols:
            label += f"<TR><TD ALIGN='LEFT'>{col}</TD></TR>"
        label += "</TABLE>>"
        dot.node(table, label=label, shape="plaintext")

    for table in insp.get_table_names():
        for fk in insp.get_foreign_keys(table):
            if fk["referred_table"]:
                dot.edge(table, fk["referred_table"], label="FK")

    st.graphviz_chart(dot, use_container_width=True)

# ----------------- Tabs -----------------
st.title("🤖 AI Database Agent with Memory")
tabs = st.tabs(["📂 Create Database", "✏️ Update Database", "📊 Explore & Analyze", "🗂️ History"])

# ----------------- Tab 1: Create -----------------
with tabs[0]:
    st.subheader("Create a New Database from English Description")
    user_desc = st.text_area("📝 Describe the database you want to create", 
                             placeholder="Example: A software company with employees, projects, clients, and finances.")
    if st.button("🚀 Generate & Create Database"):
        if not user_desc.strip():
            st.error("Please provide a description.")
        else:
            with st.spinner("Generating SQL schema..."):
                sql_schema = ask_model_to_sql(user_desc, "", mode="create")
                if validate_sql(sql_schema):
                    try:
                        if os.path.exists(DB_PATH):
                            os.remove(DB_PATH)
                        engine = make_engine()
                        with engine.begin() as conn:
                            for stmt in sql_schema.split(";"):
                                if stmt.strip():
                                    conn.execute(text(stmt))
                        st.session_state.engine = engine
                        st.session_state.schema_text = get_schema_text(engine)
                        st.session_state.history.append({"action": "create", "sql": sql_schema})
                        save_state()
                        st.success("✅ Database created successfully!")
                        st.code(st.session_state.schema_text)
                        st.subheader("📊 ER Diagram")
                        render_er_diagram(engine)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.code(sql_schema, language="sql")
                else:
                    st.error("Invalid or unsafe SQL generated.")
                    st.code(sql_schema, language="sql")

# ----------------- Tab 2: Update -----------------
with tabs[1]:
    st.subheader("Update Your Database")
    if not st.session_state.engine:
        st.warning("⚠️ Create a database first in the previous tab.")
    else:
        update_req = st.text_area("✏️ Describe the update you want", 
                                  placeholder="Example: Add a salary column to employees table.")
        if st.button("⚡ Apply Update"):
            if not update_req.strip():
                st.error("Please provide an update request.")
            else:
                with st.spinner("Generating update SQL..."):
                    sql_update = ask_model_to_sql(update_req, st.session_state.schema_text, mode="update")
                    if validate_sql(sql_update, allow_write=True):
                        try:
                            with st.session_state.engine.begin() as conn:
                                for stmt in sql_update.split(";"):
                                    if stmt.strip():
                                        conn.execute(text(stmt))
                            st.session_state.schema_text = get_schema_text(st.session_state.engine)
                            st.session_state.history.append({"action": "update", "sql": sql_update})
                            save_state()
                            st.success("✅ Database updated successfully!")
                            st.code(sql_update, language="sql")
                            st.code(st.session_state.schema_text)
                            st.subheader("📊 Updated ER Diagram")
                            render_er_diagram(st.session_state.engine)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.code(sql_update, language="sql")
                    else:
                        st.error("Invalid SQL generated.")
                        st.code(sql_update, language="sql")

# ----------------- Tab 3: Explore -----------------
with tabs[2]:
    st.subheader("Explore and Analyze Your Database")
    if not st.session_state.engine:
        st.warning("⚠️ Create a database first.")
    else:
        read_only = st.checkbox("🔒 Read-only mode", value=True)
        query_req = st.text_area("💬 Ask in plain English", 
                                 placeholder="Example: Show me total profits per year.")
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("✨ Generate SQL"):
                if not query_req.strip():
                    st.error("Please provide a query request.")
                else:
                    with st.spinner("Generating SQL..."):
                        sql_query = ask_model_to_sql(query_req, st.session_state.schema_text, mode="select")
                        if validate_sql(sql_query, allow_write=not read_only):
                            st.session_state.last_sql = sql_query
                            st.session_state.history.append({"action": "query_gen", "sql": sql_query})
                            save_state()
                            st.success("✅ SQL generated!")
                            st.code(sql_query, language="sql")
                        else:
                            st.error("Invalid SQL generated.")
                            st.code(sql_query, language="sql")
        with col2:
            if st.button("▶️ Run SQL"):
                if not st.session_state.last_sql:
                    st.error("No SQL query available. Generate one first.")
                else:
                    sql_to_run = st.session_state.last_sql
                    with st.spinner("Running query..."):
                        result = run_sql(st.session_state.engine, sql_to_run)
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result, use_container_width=True)
                            visualize(result)
                            csv = result.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download CSV", data=csv, file_name="results.csv", mime="text/csv")
                            st.session_state.history.append({"action": "query_run", "sql": sql_to_run})
                            save_state()
                        else:
                            st.error(f"Execution error: {result}")

# ----------------- Tab 4: History -----------------
with tabs[3]:
    st.subheader("🗂️ Saved History")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history[::-1], 1):
            st.markdown(f"**{i}. {item['action'].upper()}**")
            st.code(item["sql"], language="sql")
    else:
        st.info("No history saved yet.")
