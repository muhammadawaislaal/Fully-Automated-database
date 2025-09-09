import os
import re
import streamlit as st
import pandas as pd
import sqlite3
# Remove matplotlib and seaborn imports
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, String, Integer, Float, DateTime
from openai import OpenAI
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import tempfile
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------- Config & OpenAI client -----------------
st.set_page_config(page_title="AI Database Assistant", layout="wide", page_icon="🤖")

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    st.error("No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets or environment.")
    st.stop()

# ----------------- Utility / safety -----------------
FORBIDDEN = {"DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA", "CREATE", "DELETE"}

def fix_group_order_aliases(sql: str) -> str:
    """
    Remove invalid 'AS alias' inside GROUP BY and ORDER BY clauses produced by the model.
    """
    if not sql:
        return sql
    
    def _clean_clause(match):
        clause = match.group(0)
        cleaned = re.sub(r"\s+AS\s+[`\"']?([A-Za-z0-9_]+)[`\"']?", r"", clause, flags=re.IGNORECASE)
        cleaned = re.sub(r",\s*([A-Za-z0-9_.]+)\s*$", r", \1", cleaned)
        return cleaned

    sql = re.sub(r"GROUP\s+BY\s+[^\n;]+", lambda m: _clean_clause(m), sql, flags=re.IGNORECASE)
    sql = re.sub(r"ORDER\s+BY\s+[^\n;]+", lambda m: _clean_clause(m), sql, flags=re.IGNORECASE)
    sql = re.sub(r"\b([A-Za-z0-9_.]+)\s+AS\s+\1\b", r"\1", sql, flags=re.IGNORECASE)

    return sql

# ----------------- DB helpers -----------------
def init_sqlite_demo(path: str = "demo_db.sqlite"):
    """Create a small demo SQLite DB if missing."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Drop tables if they exist to start fresh
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS customers")
    cursor.execute("DROP TABLE IF EXISTS employees")
    cursor.execute("DROP TABLE IF EXISTS departments")
    
    # Create tables
    demo_sql = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    signup_date TEXT,
    country TEXT,
    tier TEXT
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price REAL,
    cost REAL,
    stock_quantity INTEGER
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    qty INTEGER,
    order_date TEXT,
    status TEXT,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers (id),
    FOREIGN KEY (product_id) REFERENCES products (id)
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department_id INTEGER,
    hire_date TEXT,
    salary REAL
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT,
    budget REAL
);

INSERT INTO customers (name, email, signup_date, country, tier) VALUES
  ('Alice Smith', 'alice@example.com', '2024-01-15', 'USA', 'Gold'),
  ('Bob Johnson', 'bob@example.com', '2024-02-20', 'UK', 'Silver'),
  ('Carol Williams', 'carol@example.com', '2024-03-10', 'Canada', 'Gold'),
  ('David Brown', 'david@example.com', '2024-04-05', 'Australia', 'Bronze'),
  ('Eva Davis', 'eva@example.com', '2024-05-12', 'USA', 'Silver'),
  ('Frank Wilson', 'frank@example.com', '2024-06-18', 'Germany', 'Gold'),
  ('Grace Lee', 'grace@example.com', '2024-07-22', 'Japan', 'Platinum'),
  ('Henry Taylor', 'henry@example.com', '2024-08-30', 'France', 'Silver');

INSERT INTO products (name, category, price, cost, stock_quantity) VALUES
  ('Widget A', 'Widgets', 9.99, 5.00, 100),
  ('Widget B', 'Widgets', 19.99, 10.00, 75),
  ('Gadget X', 'Gadgets', 29.50, 15.00, 50),
  ('Gadget Y', 'Gadgets', 39.99, 20.00, 40),
  ('Tool Z', 'Tools', 15.00, 7.50, 120),
  ('Software Pro', 'Software', 99.99, 20.00, 200),
  ('Service Basic', 'Services', 49.99, 10.00, NULL),
  ('Service Premium', 'Services', 149.99, 30.00, NULL);

INSERT INTO departments (name, budget) VALUES
  ('Sales', 100000),
  ('Marketing', 75000),
  ('Engineering', 150000),
  ('Support', 50000);

INSERT INTO employees (name, department_id, hire_date, salary) VALUES
  ('John Manager', 1, '2023-01-15', 75000),
  ('Jane Analyst', 2, '2023-03-10', 65000),
  ('Mike Developer', 3, '2023-05-20', 85000),
  ('Sarah Support', 4, '2023-07-05', 55000),
  ('Tom Sales', 1, '2023-09-12', 60000),
  ('Lisa Marketer', 2, '2023-11-30', 70000);

INSERT INTO orders (customer_id, product_id, qty, order_date, status, total_amount) VALUES
  (1, 1, 3, '2024-06-01', 'Completed', 29.97),
  (1, 3, 1, '2024-06-15', 'Completed', 29.50),
  (2, 2, 2, '2024-06-20', 'Completed', 39.98),
  (2, 4, 1, '2024-07-05', 'Completed', 39.99),
  (3, 1, 1, '2024-07-10', 'Completed', 9.99),
  (3, 5, 2, '2024-07-15', 'Completed', 30.00),
  (4, 3, 1, '2024-08-01', 'Completed', 29.50),
  (4, 2, 3, '2024-08-10', 'Completed', 59.97),
  (5, 4, 2, '2024-08-20', 'Completed', 79.98),
  (5, 1, 1, '2024-09-01', 'Completed', 9.99),
  (1, 5, 1, '2024-09-05', 'Completed', 15.00),
  (2, 3, 2, '2024-09-10', 'Completed', 59.00),
  (6, 6, 1, '2024-09-15', 'Completed', 99.99),
  (7, 7, 1, '2024-09-20', 'Completed', 49.99),
  (8, 8, 1, '2024-09-25', 'Pending', 149.99),
  (6, 2, 2, '2024-10-01', 'Processing', 39.98),
  (7, 4, 1, '2024-10-05', 'Processing', 39.99);
"""
    
    try:
        cursor.executescript(demo_sql)
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error creating demo database: {e}")
        return False
    finally:
        conn.close()

def make_engine(db_type: str, sqlite_path: str = None, url: str = None):
    if db_type == "SQLite (local demo)":
        if not sqlite_path:
            sqlite_path = "demo_db.sqlite"
        return create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
    else:
        if not url:
            raise ValueError("No connection URL provided.")
        return create_engine(url)

def get_schema_text(engine) -> str:
    try:
        insp = inspect(engine)
        tables = insp.get_table_names()
        lines = []
        for t in tables:
            try:
                cols = [f"{c['name']} ({c['type']})" for c in insp.get_columns(t)]
                lines.append(f"📊 {t}: {', '.join(cols)}")
            except Exception:
                cols = []
                lines.append(f"📊 {t}: Could not get columns")
        return "\n\n".join(lines) if lines else "No tables found."
    except Exception as e:
        return f"Could not introspect schema: {e}"

def get_schema_dict(engine) -> Dict[str, list]:
    try:
        insp = inspect(engine)
        schema = {}
        for t in insp.get_table_names():
            try:
                schema[t] = [c["name"] for c in insp.get_columns(t)]
            except Exception:
                schema[t] = []
        return schema
    except Exception:
        return {}

# ----------------- SQL extraction / validation -----------------
def extract_sql(model_output: str) -> str:
    if not model_output:
        return ""
    # Remove code fences
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", model_output, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # Find SQL keywords
    m2 = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|EXPLAIN)\b", model_output, flags=re.IGNORECASE)
    if m2:
        start = m2.start(1)
        candidate = model_output[start:].strip()
        candidate = re.sub(r"```+\s*$", "", candidate).strip()
        return candidate
    
    return model_output.strip()

def validate_sql(clean_sql: str, read_only: bool) -> Tuple[bool,str,str]:
    if not clean_sql:
        return False, "Empty query.", ""
    
    sql = clean_sql.strip()
    sql_upper = sql.upper()
    
    if sql_upper.startswith("ERROR"):
        return False, sql, sql
    
    for bad in FORBIDDEN:
        if re.search(rf"\b{bad}\b", sql_upper):
            return False, f"Forbidden keyword detected: {bad}", sql
    
    m = re.search(r"^\s*([A-Z]+)", sql_upper)
    if not m:
        return False, "Invalid SQL syntax (couldn't detect first keyword).", sql
    
    kw = m.group(1)
    read_allowed = {"SELECT", "WITH", "EXPLAIN"}
    full_allowed = read_allowed | {"INSERT", "UPDATE", "DELETE"}
    
    if read_only:
        if kw in read_allowed:
            return True, "OK", sql
        else:
            return False, f"Only read queries allowed in Read-only mode (found: {kw}).", sql
    else:
        if kw in full_allowed:
            return True, "OK", sql
        else:
            return False, f"Query type not supported (found: {kw}).", sql

# ----------------- Pivot helpers -----------------
def infer_date_and_entity(schema: Dict[str, list]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # First check for orders table (most common)
    if "orders" in schema:
        cols = schema["orders"]
        date_col = next((c for c in cols if "date" in c.lower()), "order_date")
        
        # Check for customer references
        if "customer_id" in cols:
            return "orders", date_col, "customer_id"
        elif "product_id" in cols:
            return "orders", date_col, "product_id"
    
    # Check for other tables with date columns
    for table, cols in schema.items():
        date_col = next((c for c in cols if "date" in c.lower()), None)
        if date_col:
            # Look for entity columns
            entity_col = next((c for c in cols if any(keyword in c.lower() for keyword in 
                                                     ["name", "id", "product", "customer", "client", "user"])), None)
            return table, date_col, entity_col
    
    return None, None, None

def build_year_pivot_sql(table: str, date_col: str, entity_col: str, year: str, schema: Dict[str, list]) -> str:
    # Handle entity column mapping
    if entity_col and entity_col.endswith("_id"):
        base_table = entity_col.replace("_id", "")
        if base_table in schema:
            join_clause = f"LEFT JOIN {base_table} ON {table}.{entity_col} = {base_table}.id"
            select_entity = f"{base_table}.name AS {base_table}_name"
            group_by_entity = f"{base_table}.name"
        else:
            join_clause = ""
            select_entity = f"{table}.{entity_col}"
            group_by_entity = f"{table}.{entity_col}"
    else:
        join_clause = ""
        select_entity = f"{table}.{entity_col}" if entity_col else "1"
        group_by_entity = select_entity

    # Build monthly cases
    month_cases = []
    for m in range(1, 13):
        month_str = f"{m:02d}"
        month_name = datetime(2024, m, 1).strftime("%b")
        month_cases.append(f"SUM(CASE WHEN strftime('%m', {table}.{date_col}) = '{month_str}' THEN COALESCE({table}.qty, 1) ELSE 0 END) AS \"{month_name}\"")

    cases_sql = ",\n    ".join(month_cases)
    
    sql = f"""SELECT
    {select_entity},
    {cases_sql}
FROM {table}
{join_clause}
WHERE strftime('%Y', {table}.{date_col}) = '{year}'
GROUP BY {group_by_entity}
ORDER BY {group_by_entity};"""
    
    return sql

# ----------------- Model prompt / SQL generation -----------------
def ask_model_to_sql_llm(nl_request: str, schema_text: str, read_only: bool, prefer_full_columns: bool = True) -> Tuple[bool, str, str]:
    """
    Use the LLM to generate SQL. Returns (ok, raw_output, cleaned_sql).
    """
    if not nl_request.strip():
        return False, "Empty natural language request.", ""

    system_msg = """You are an expert SQL assistant. Respond with a single SQL statement only (no explanation). 
Default dialect: SQLite. IMPORTANT RULES:
1. Never use 'AS' inside GROUP BY or ORDER BY clauses
2. Use explicit table.column references when needed
3. When user says 'clients', use 'customers' table
4. When user says 'client_id', use 'customer_id' column
5. Use proper date formatting for SQLite: strftime() functions"""

    user_msg = f"""Database schema:
{schema_text}

User request: {nl_request}

Generate a clean SQL query that answers the user's request. 
Return only the SQL statement without any explanations or code fences."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content
        cleaned = extract_sql(raw)
        return True, raw, cleaned
    except Exception as e:
        return False, f"OpenAI error: {e}", ""

# ----------------- Database Creation Functions -----------------
def generate_database_schema_from_description(description: str) -> Tuple[bool, str, str]:
    """
    Generate a database schema from a natural language description.
    Returns (success, schema_sql, error_message)
    """
    system_msg = """You are an expert database architect. Based on the user's description of their business or data needs,
generate a complete SQLite database schema with appropriate tables, columns, and relationships.

Include:
1. All necessary tables with appropriate columns and data types
2. Primary keys and foreign key relationships
3. Sample data insertion statements for at least 5 rows per table
4. Comments explaining the purpose of each table

Return only the SQL code without any explanations or markdown formatting."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": description}
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        sql_code = resp.choices[0].message.content
        # Clean up the SQL code
        sql_code = re.sub(r"```(?:sql)?\s*", "", sql_code)
        sql_code = re.sub(r"```\s*$", "", sql_code)
        return True, sql_code, ""
    except Exception as e:
        return False, "", f"Error generating schema: {e}"

def execute_schema_creation(sql_code: str, db_path: str = "custom_db.sqlite") -> bool:
    """
    Execute the SQL schema creation code.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute the SQL code
        cursor.executescript(sql_code)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error creating database: {e}")
        return False

# ----------------- Data Visualization Functions -----------------
def create_visualization(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, 
                        group_by: str = None, title: str = None):
    """
    Create various types of visualizations from DataFrame.
    """
    try:
        if chart_type == "Bar Chart":
            if group_by:
                fig = px.bar(df, x=x_col, y=y_col, color=group_by, title=title or f"{y_col} by {x_col}")
            else:
                fig = px.bar(df, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}")
                
        elif chart_type == "Line Chart":
            if group_by:
                fig = px.line(df, x=x_col, y=y_col, color=group_by, title=title or f"{y_col} over {x_col}")
            else:
                fig = px.line(df, x=x_col, y=y_col, title=title or f"{y_col} over {x_col}")
                
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=title or f"Distribution of {y_col} by {x_col}")
            
        elif chart_type == "Scatter Plot":
            if group_by:
                fig = px.scatter(df, x=x_col, y=y_col, color=group_by, title=title or f"{y_col} vs {x_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, title=title or f"{y_col} vs {x_col}")
                
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, title=title or f"Distribution of {x_col}")
            
        elif chart_type == "Heatmap":
            # For heatmap, we need a pivot table
            if group_by and y_col:
                pivot_df = df.pivot_table(values=y_col, index=x_col, columns=group_by, aggfunc='sum')
                fig = px.imshow(pivot_df, title=title or f"Heatmap of {y_col} by {x_col} and {group_by}")
            else:
                st.warning("Heatmap requires both x column and group by column")
                return None
                
        else:
            st.warning("Unsupported chart type")
            return None
            
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def generate_insights_from_data(df: pd.DataFrame, table_name: str = None) -> str:
    """
    Use AI to generate insights from the data.
    """
    # Create a summary of the data
    summary = f"""
    Table: {table_name or 'Unknown'}
    Shape: {df.shape[0]} rows, {df.shape[1]} columns
    Columns: {', '.join(df.columns)}
    Sample data: {df.head(3).to_string()}
    """
    
    system_msg = """You are a data analyst. Based on the provided dataset summary, generate 3-5 key insights about the data.
Focus on trends, patterns, anomalies, and business implications. Keep each insight concise and actionable."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": summary}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {e}"

# ----------------- UI Setup -----------------
def main():
    st.sidebar.header("🔧 Database Configuration")
    db_type = st.sidebar.selectbox("Database Type", ["SQLite (local demo)", "SQLite (custom)", "Postgres / MySQL (custom URL)"])

    # Initialize session state
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "schema_text" not in st.session_state:
        st.session_state.schema_text = "No DB connected."
    if "schema_dict" not in st.session_state:
        st.session_state.schema_dict = {}
    if "current_db_path" not in st.session_state:
        st.session_state.current_db_path = "demo_db.sqlite"
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Create Database"

    # Database connection setup
    if db_type == "SQLite (local demo)":
        sqlite_path = st.sidebar.text_input("SQLite Database Path", value="demo_db.sqlite")
        
        if st.sidebar.button("🚀 Initialize Demo Database", help="Create sample database with demo data"):
            with st.spinner("Creating demo database..."):
                success = init_sqlite_demo(sqlite_path)
                if success:
                    st.session_state.current_db_path = sqlite_path
                    st.sidebar.success("✅ Demo database created successfully!")
                    st.session_state.db_initialized = True
                else:
                    st.sidebar.error("❌ Failed to create demo database")

        if st.sidebar.button("🔗 Connect to Database"):
            try:
                engine = make_engine(db_type, sqlite_path=sqlite_path)
                st.session_state.engine = engine
                st.session_state.schema_text = get_schema_text(engine)
                st.session_state.schema_dict = get_schema_dict(engine)
                st.session_state.current_db_path = sqlite_path
                st.sidebar.success("✅ Connected to database!")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {e}")

    elif db_type == "SQLite (custom)":
        custom_db_path = st.sidebar.text_input("Custom Database Path", value="custom_db.sqlite")
        st.session_state.current_db_path = custom_db_path
        
        if st.sidebar.button("🔗 Connect to Custom Database"):
            try:
                engine = make_engine("SQLite (local demo)", sqlite_path=custom_db_path)
                st.session_state.engine = engine
                st.session_state.schema_text = get_schema_text(engine)
                st.session_state.schema_dict = get_schema_dict(engine)
                st.sidebar.success("✅ Connected to custom database!")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {e}")

    else:  # External database
        db_url = st.sidebar.text_input("Database URL", 
                                      placeholder="postgresql://user:pass@host:port/dbname or mysql+pymysql://...")
        if st.sidebar.button("🔗 Connect to External Database") and db_url:
            try:
                engine = make_engine(db_type, url=db_url)
                st.session_state.engine = engine
                st.session_state.schema_text = get_schema_text(engine)
                st.session_state.schema_dict = get_schema_dict(engine)
                st.sidebar.success("✅ Connected to external database!")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {e}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🗃️ Create Database", "✏️ Manage Database", "📊 Explore Database"])

    # Tab 1: Create Database
    with tab1:
        st.header("🗃️ Create a New Database")
        st.write("Describe your database requirements in plain English, and we'll generate a complete database schema for you.")
        
        db_description = st.text_area(
            "Describe your database needs:",
            height=150,
            placeholder="e.g., 'I need a database for my software company with tables for employees, projects, clients, and invoices. Employees have names, roles, and salaries. Projects have names, deadlines, and budgets. Clients have company names and contact information. Invoices have amounts, dates, and status.'"
        )
        
        db_name = st.text_input("Database Name", value="my_database.sqlite")
        
        if st.button("Generate Database", type="primary"):
            if not db_description:
                st.error("Please describe your database requirements.")
            else:
                with st.spinner("Generating database schema..."):
                    success, schema_sql, error = generate_database_schema_from_description(db_description)
                    
                    if success:
                        st.success("Database schema generated successfully!")
                        with st.expander("View Generated Schema"):
                            st.code(schema_sql, language="sql")
                        
                        # Execute the schema creation
                        with st.spinner("Creating database..."):
                            if execute_schema_creation(schema_sql, db_name):
                                st.success(f"Database '{db_name}' created successfully!")
                                
                                # Connect to the new database
                                try:
                                    engine = make_engine("SQLite (local demo)", sqlite_path=db_name)
                                    st.session_state.engine = engine
                                    st.session_state.schema_text = get_schema_text(engine)
                                    st.session_state.schema_dict = get_schema_dict(engine)
                                    st.session_state.current_db_path = db_name
                                    st.success("✅ Connected to your new database!")
                                except Exception as e:
                                    st.error(f"Connected but encountered error: {e}")
                            else:
                                st.error("Failed to create database.")
                    else:
                        st.error(f"Failed to generate schema: {error}")

    # Tab 2: Manage Database
    with tab2:
        st.header("✏️ Manage Your Database")
        
        if not st.session_state.engine:
            st.warning("Please connect to a database first.")
        else:
            st.write("Use natural language to modify your database.")
            
            # Display schema
            with st.expander("📊 Current Database Schema", expanded=True):
                st.code(st.session_state.schema_text)
            
            # Read-only mode toggle
            read_only = st.checkbox("🔒 Read-only Mode", value=False, 
                                   help="Prevents INSERT/UPDATE/DELETE operations for safety")
            if not read_only:
                st.warning("⚠️ Full access mode enabled. Use with caution!")
            
            # Management query input
            management_query = st.text_area(
                "💬 What would you like to change?",
                height=100,
                placeholder="e.g., 'Add a new customer named John Doe with email john@example.com', 'Update product prices to increase by 10%', 'Delete all orders from last month'"
            )
            
            if st.button("Execute Change", type="primary"):
                if not management_query:
                    st.error("Please enter a management query.")
                else:
                    with st.spinner("Generating and executing change..."):
                        ok, raw, cleaned = ask_model_to_sql_llm(
                            management_query, 
                            st.session_state.schema_text, 
                            read_only
                        )
                        
                        if ok:
                            # Validate the SQL
                            is_valid, message, norm_sql = validate_sql(cleaned, read_only)
                            
                            if is_valid:
                                try:
                                    # Execute the SQL
                                    with st.session_state.engine.connect() as conn:
                                        result = conn.execute(text(norm_sql))
                                        conn.commit()
                                    
                                    st.success("✅ Change executed successfully!")
                                    
                                    # Refresh schema if structure changed
                                    if any(keyword in norm_sql.upper() for keyword in ["CREATE", "ALTER", "DROP"]):
                                        st.session_state.schema_text = get_schema_text(st.session_state.engine)
                                        st.session_state.schema_dict = get_schema_dict(st.session_state.engine)
                                        st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Execution failed: {e}")
                                    st.code(norm_sql, language="sql")
                            else:
                                st.error(f"❌ Invalid SQL: {message}")
                                st.code(cleaned, language="sql")
                        else:
                            st.error(f"❌ Failed to generate SQL: {raw}")

    # Tab 3: Explore Database
    with tab3:
        st.header("📊 Explore Your Database")
        
        if not st.session_state.engine:
            st.warning("Please connect to a database first.")
        else:
            # Display schema
            with st.expander("📊 Database Schema", expanded=True):
                st.code(st.session_state.schema_text)
            
            # Exploration query input
            exploration_query = st.text_area(
                "💬 What would you like to explore?",
                height=100,
                placeholder="e.g., 'Show me monthly sales trends', 'Compare product categories by revenue', 'List top 5 customers by spending'"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("✨ Generate & Run Query", type="primary"):
                    if not exploration_query:
                        st.error("Please enter an exploration query.")
                    else:
                        with st.spinner("Generating and executing query..."):
                            ok, raw, cleaned = ask_model_to_sql_llm(
                                exploration_query, 
                                st.session_state.schema_text, 
                                True  # Read-only for exploration
                            )
                            
                            if ok:
                                st.session_state.exploration_sql = cleaned
                                st.session_state.exploration_raw = raw
                                
                                # Validate and execute
                                is_valid, message, norm_sql = validate_sql(cleaned, True)
                                
                                if is_valid:
                                    try:
                                        df = pd.read_sql_query(text(norm_sql), st.session_state.engine)
                                        st.session_state.exploration_df = df
                                        st.session_state.exploration_sql_executed = norm_sql
                                        
                                        st.success(f"✅ Query executed successfully! Returned {len(df)} rows.")
                                        
                                        # Display results
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Add to history
                                        if "exploration_history" not in st.session_state:
                                            st.session_state.exploration_history = []
                                        st.session_state.exploration_history.append({
                                            "timestamp": datetime.now().isoformat(),
                                            "query": norm_sql,
                                            "rows": len(df)
                                        })
                                        
                                    except Exception as e:
                                        st.error(f"❌ Execution failed: {e}")
                                        st.code(norm_sql, language="sql")
                                else:
                                    st.error(f"❌ Invalid SQL: {message}")
                                    st.code(cleaned, language="sql")
                            else:
                                st.error(f"❌ Failed to generate SQL: {raw}")
            
            with col2:
                if st.button("📈 Generate Visualizations", type="secondary"):
                    if "exploration_df" not in st.session_state or st.session_state.exploration_df.empty:
                        st.error("No data to visualize. Please run a query first.")
                    else:
                        df = st.session_state.exploration_df
                        
                        # Generate insights
                        with st.spinner("Generating insights..."):
                            insights = generate_insights_from_data(df, "Query Results")
                            st.subheader("💡 Data Insights")
                            st.write(insights)
                        
                        # Visualization options
                        st.subheader("📊 Visualization Options")
                        
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        if not numeric_cols:
                            st.warning("No numeric columns found for visualization.")
                        else:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                chart_type = st.selectbox(
                                    "Chart Type",
                                    ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram", "Heatmap"]
                                )
                            
                            with col2:
                                x_col = st.selectbox("X Axis", df.columns.tolist(), index=0)
                            
                            with col3:
                                if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"]:
                                    y_col = st.selectbox("Y Axis", numeric_cols, index=0 if numeric_cols else None)
                                else:
                                    y_col = None
                            
                            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"] and len(df.columns) > 2:
                                group_by = st.selectbox("Group By", [None] + categorical_cols)
                            else:
                                group_by = None
                            
                            title = st.text_input("Chart Title", value=f"{y_col or x_col} Analysis")
                            
                            if st.button("Generate Chart"):
                                fig = create_visualization(df, chart_type, x_col, y_col, group_by, title)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
            
            # Show query history
            if st.session_state.get("exploration_history"):
                with st.expander("🕒 Exploration History", expanded=False):
                    for i, entry in enumerate(reversed(st.session_state.exploration_history[-5:])):
                        cols = st.columns([4, 1])
                        with cols[0]:
                            st.code(entry["query"], language="sql")
                            st.caption(f"Rows: {entry['rows']} • {entry['timestamp'][:19]}")
                        with cols[1]:
                            if st.button("↩️ Load", key=f"load_ex_{i}"):
                                st.session_state.exploration_sql = entry["query"]
                                st.rerun()

    # Footer
    st.markdown("---")
    st.caption("💡 Tip: Be specific with your questions for better results! Include timeframes, entities, and what you want to see. DEVELOPED By Muhammad Awais Laal")

    # Auto-initialize demo DB if not connected
    if not st.session_state.engine and db_type == "SQLite (local demo)":
        if st.sidebar.button("🔄 Auto-setup Demo", help="Automatically setup demo database"):
            with st.spinner("Setting up demo database..."):
                success = init_sqlite_demo("demo_db.sqlite")
                if success:
                    try:
                        engine = make_engine(db_type, sqlite_path="demo_db.sqlite")
                        st.session_state.engine = engine
                        st.session_state.schema_text = get_schema_text(engine)
                        st.session_state.schema_dict = get_schema_dict(engine)
                        st.session_state.current_db_path = "demo_db.sqlite"
                        st.sidebar.success("✅ Demo database setup complete!")
                    except Exception as e:
                        st.sidebar.error(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()


