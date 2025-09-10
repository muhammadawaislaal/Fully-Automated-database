import os
import re
import streamlit as st
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text, inspect
import openai
from datetime import datetime
from typing import Tuple, Dict
import plotly.express as px

# ----------------- Config & OpenAI client -----------------
st.set_page_config(
    page_title="AI Database Assistant", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        background-color: #3498db;
        color: white;
    }
    .stButton button:hover {
        background-color: #2980b9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
if "exploration_df" not in st.session_state:
    st.session_state.exploration_df = pd.DataFrame()
if "exploration_history" not in st.session_state:
    st.session_state.exploration_history = []
if "read_only_mode" not in st.session_state:
    st.session_state.read_only_mode = True

# Set OpenAI API key
try:
    if "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        st.error("No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets or environment.")
        st.stop()
except Exception as e:
    st.error(f"Error setting up OpenAI API: {e}")
    st.stop()

# ----------------- Utility / safety -----------------
FORBIDDEN = {"DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA"}

def fix_group_order_aliases(sql: str) -> str:
    """Remove invalid 'AS alias' inside GROUP BY and ORDER BY clauses."""
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
def init_sqlite_demo(path: str = "demo_db.sqlite") -> bool:
    """Create a demo SQLite DB with sample data."""
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Drop existing tables to avoid conflicts
        tables = ["orders", "products", "customers", "employees", "departments", "projects", "clients", "assignments"]
        for table in tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            except:
                pass
        
        # Create tables with proper schema
        demo_sql = """
CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL
);

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    position TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    hire_date DATE NOT NULL,
    salary REAL NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments (id)
);

CREATE TABLE IF NOT EXISTS clients (
    client_id INTEGER PRIMARY KEY,
    client_name TEXT NOT NULL,
    contact_person TEXT,
    email TEXT,
    phone TEXT,
    address TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    project_id INTEGER PRIMARY KEY,
    project_name TEXT NOT NULL,
    client_id INTEGER,
    start_date DATE,
    end_date DATE,
    budget REAL,
    status TEXT,
    FOREIGN KEY (client_id) REFERENCES clients (client_id)
);

CREATE TABLE IF NOT EXISTS assignments (
    assignment_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees (id),
    FOREIGN KEY (project_id) REFERENCES projects (project_id)
);

-- Insert sample data only if tables are empty
INSERT OR IGNORE INTO departments (id, name, budget) VALUES 
(1, 'Engineering', 500000),
(2, 'Sales', 300000),
(3, 'Marketing', 200000),
(4, 'Support', 150000);

INSERT OR IGNORE INTO employees (id, first_name, last_name, email, phone, position, department_id, hire_date, salary) VALUES 
(1, 'John', 'Smith', 'john@company.com', '555-0101', 'Senior Developer', 1, '2022-01-15', 95000),
(2, 'Sarah', 'Johnson', 'sarah@company.com', '555-0102', 'Project Manager', 1, '2022-03-10', 85000),
(3, 'Mike', 'Chen', 'mike@company.com', '555-0103', 'Sales Executive', 2, '2022-05-20', 75000),
(4, 'Lisa', 'Rodriguez', 'lisa@company.com', '555-0104', 'Marketing Specialist', 3, '2022-07-05', 65000);

INSERT OR IGNORE INTO clients (client_id, client_name, contact_person, email, phone, address) VALUES 
(1, 'TechSolutions Inc', 'David Wilson', 'david@techsolutions.com', '555-0201', '123 Tech Ave'),
(2, 'Global Enterprises', 'Maria Garcia', 'maria@globalent.com', '555-0202', '456 Business Blvd'),
(3, 'Northwest Industries', 'Robert Brown', 'robert@nwindustries.com', '555-0203', '789 Industry St');

INSERT OR IGNORE INTO projects (project_id, project_name, client_id, start_date, end_date, budget, status) VALUES 
(1, 'Website Redesign', 1, '2023-01-15', '2023-06-15', 50000, 'Completed'),
(2, 'Mobile App Development', 2, '2023-03-01', '2023-09-01', 75000, 'In Progress'),
(3, 'Marketing Campaign', 3, '2023-05-01', '2023-08-01', 25000, 'Planning');

INSERT OR IGNORE INTO assignments (assignment_id, employee_id, project_id, role) VALUES 
(1, 1, 1, 'Lead Developer'),
(2, 2, 1, 'Project Manager'),
(3, 1, 2, 'Backend Developer'),
(4, 3, 2, 'Client Liaison'),
(5, 4, 3, 'Campaign Manager');
"""
        
        cursor.executescript(demo_sql)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error creating demo database: {e}")
        return False

def make_engine(db_type: str, sqlite_path: str = None, url: str = None):
    try:
        if db_type == "SQLite (Demo)":
            if not sqlite_path:
                sqlite_path = "demo_db.sqlite"
            return create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
        else:  # PostgreSQL
            if not url:
                raise ValueError("No connection URL provided.")
            return create_engine(url)
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        return None

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

def validate_sql(clean_sql: str, read_only: bool) -> Tuple[bool, str, str]:
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

# ----------------- Enhanced Model prompt / SQL generation -----------------
def ask_model_to_sql_llm(nl_request: str, schema_text: str, read_only: bool) -> Tuple[bool, str, str]:
    """Use the LLM to generate SQL with enhanced understanding."""
    if not nl_request.strip():
        return False, "Empty natural language request.", ""

    system_msg = """You are an expert SQL assistant that understands natural language perfectly. 
Respond with a single SQL statement only (no explanation). 
Default dialect: SQLite. 

CRITICAL RULES:
1. Never use 'AS' inside GROUP BY or ORDER BY clauses
2. Use explicit table.column references when needed
3. Always include all required NOT NULL fields in INSERT statements
4. Use proper date formatting for SQLite: date('now') or specific dates
5. For Employees: always include first_name, last_name, email, position, department_id, hire_date, salary
6. For Projects: always include project_name, client_id is optional but recommended
7. For Assignments: always include employee_id, project_id, role
8. For Clients: always include client_name
9. Use INSERT OR IGNORE to avoid duplicate key errors
10. Use CREATE TABLE IF NOT EXISTS to avoid table creation errors

Understand user intent perfectly and generate accurate SQL."""

    user_msg = f"""Database schema:
{schema_text}

User request: {nl_request}

Generate a perfect SQL query that exactly matches the user's intent. 
Return only the SQL statement without any explanations or code fences."""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=800,
        )
        raw = resp['choices'][0]['message']['content']
        cleaned = extract_sql(raw)
        return True, raw, cleaned
    except Exception as e:
        return False, f"OpenAI error: {e}", ""

# ----------------- Database Creation Functions -----------------
def generate_database_schema_from_description(description: str) -> Tuple[bool, str, str]:
    """Generate a database schema from natural language description."""
    system_msg = """You are an expert database architect. Create complete SQLite database schema.

ESSENTIAL REQUIREMENTS:
1. All tables must have proper PRIMARY KEYs
2. Include all necessary NOT NULL constraints
3. Add appropriate FOREIGN KEY relationships
4. Include sample data with all required fields
5. Ensure data integrity and relationships work properly
6. Use proper SQLite data types (INTEGER, TEXT, REAL, DATE)
7. Include at least 3-5 sample records per table
8. Use CREATE TABLE IF NOT EXISTS to avoid errors
9. Use INSERT OR IGNORE to avoid duplicate errors

Return only the SQL code without any explanations."""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": description}
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        sql_code = resp['choices'][0]['message']['content']
        # Clean up the SQL code
        sql_code = re.sub(r"```(?:sql)?\s*", "", sql_code)
        sql_code = re.sub(r"```\s*$", "", sql_code)
        return True, sql_code, ""
    except Exception as e:
        return False, "", f"Error generating schema: {e}"

def execute_schema_creation(sql_code: str, db_path: str = "custom_db.sqlite") -> bool:
    """Execute the SQL schema creation code."""
    try:
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
    """Create various types of visualizations from DataFrame."""
    try:
        if chart_type == "Bar Chart":
            if group_by and y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=group_by, title=title or f"{y_col} by {x_col}")
            elif y_col:
                fig = px.bar(df, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}")
            else:
                fig = px.bar(df, x=x_col, title=title or f"Count by {x_col}")
                
        elif chart_type == "Line Chart":
            if group_by and y_col:
                fig = px.line(df, x=x_col, y=y_col, color=group_by, title=title or f"{y_col} over {x_col}")
            elif y_col:
                fig = px.line(df, x=x_col, y=y_col, title=title or f"{y_col} over {x_col}")
            else:
                st.warning("Line chart requires a Y-axis column")
                return None
                
        elif chart_type == "Pie Chart":
            if y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=title or f"Distribution of {y_col} by {x_col}")
            else:
                st.warning("Pie chart requires a values column")
                return None
            
        elif chart_type == "Scatter Plot":
            if y_col:
                if group_by:
                    fig = px.scatter(df, x=x_col, y=y_col, color=group_by, title=title or f"{y_col} vs {x_col}")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=title or f"{y_col} vs {x_col}")
            else:
                st.warning("Scatter plot requires a Y-axis column")
                return None
                
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, title=title or f"Distribution of {x_col}")
            
        elif chart_type == "Heatmap":
            if group_by and y_col:
                try:
                    pivot_df = df.pivot_table(values=y_col, index=x_col, columns=group_by, aggfunc='sum')
                    fig = px.imshow(pivot_df, title=title or f"Heatmap of {y_col} by {x_col} and {group_by}")
                except Exception:
                    st.warning("Could not create heatmap with selected columns")
                    return None
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
    """Use AI to generate insights from the data."""
    if df.empty:
        return "No data available for analysis."
    
    summary = f"""
    Table: {table_name or 'Unknown'}
    Shape: {df.shape[0]} rows, {df.shape[1]} columns
    Columns: {', '.join(df.columns)}
    Sample data: {df.head(3).to_string()}
    """
    
    system_msg = """You are a data analyst. Generate 3-5 key insights about the data.
Focus on trends, patterns, anomalies, and business implications. Keep insights concise and actionable."""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": summary}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return resp['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating insights: {e}"

# ----------------- Enhanced UI with better error handling -----------------
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #3498db;'>🤖 AI Database Assistant</h1>", unsafe_allow_html=True)
        
        # Instructions
        with st.expander("📋 How to Use", expanded=True):
            st.markdown("""
            **1. Setup Database**: Connect to a demo database or create your own
            
            **2. Explore Data**: Ask questions in plain English about your data
            
            **3. Manage Data**: Add, update, or delete records using natural language
            
            **4. Visualize**: Create charts and get insights from your data
            
            **Examples**:
            - "Show me all employees with their departments"
            - "Add a new employee named John Doe"
            - "Update Lisa's salary to $70,000"
            - "Create a bar chart of salaries by department"
            """)
        
        st.markdown("---")
        st.header("🔧 Database Configuration")
        db_type = st.selectbox("Database Type", ["SQLite (Demo)", "PostgreSQL (Custom)"])
        
        if db_type == "SQLite (Demo)":
            st.markdown("""
            <div class='info-box'>
            <b>SQLite Demo Database</b><br>
            Contains sample data with employees, departments, projects, and clients.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Initialize Demo Database", use_container_width=True):
                with st.spinner("Creating demo database..."):
                    success = init_sqlite_demo("demo_db.sqlite")
                    if success:
                        st.session_state.current_db_path = "demo_db.sqlite"
                        st.success("✅ Demo database created successfully!")
                        st.session_state.db_initialized = True
                    else:
                        st.error("❌ Failed to create demo database")
            
            if st.button("🔗 Connect to Demo DB", use_container_width=True):
                try:
                    engine = make_engine(db_type, sqlite_path="demo_db.sqlite")
                    if engine:
                        st.session_state.engine = engine
                        st.session_state.schema_text = get_schema_text(engine)
                        st.session_state.schema_dict = get_schema_dict(engine)
                        st.session_state.current_db_path = "demo_db.sqlite"
                        st.success("✅ Connected to database!")
                    else:
                        st.error("❌ Failed to create database engine")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
        
        else:  # PostgreSQL
            st.markdown("""
            <div class='info-box'>
            <b>PostgreSQL Database</b><br>
            Connect to your existing PostgreSQL database using a connection string.
            </div>
            """, unsafe_allow_html=True)
            
            db_url = st.text_input(
                "PostgreSQL Connection URL", 
                placeholder="postgresql://username:password@host:port/database"
            )
            
            if st.button("🔗 Connect to PostgreSQL", use_container_width=True) and db_url:
                try:
                    engine = make_engine(db_type, url=db_url)
                    if engine:
                        st.session_state.engine = engine
                        st.session_state.schema_text = get_schema_text(engine)
                        st.session_state.schema_dict = get_schema_dict(engine)
                        st.success("✅ Connected to PostgreSQL database!")
                    else:
                        st.error("❌ Failed to create database engine")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
        
        # Database status
        st.markdown("---")
        st.subheader("📊 Database Status")
        if st.session_state.engine:
            st.success("✅ Connected to database")
            st.info(f"Database: {st.session_state.current_db_path}")
        else:
            st.warning("⚠️ No database connected")
        
        # Read-only mode toggle
        st.session_state.read_only_mode = st.checkbox(
            "🔒 Read-only Mode", 
            value=st.session_state.read_only_mode,
            help="Prevents INSERT/UPDATE/DELETE operations for safety"
        )
        
        if not st.session_state.read_only_mode:
            st.warning("⚠️ Full access mode enabled. Use with caution!")
        
        st.markdown("---")
        st.caption("💡 Tip: Be specific with your questions for better results!")
        st.caption("Developed by Muhammad Awais Laal")

    # Main content
    st.markdown("<h1 class='main-header'>AI Database Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Interact with your database using natural language</p>", unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🗃️ Create Database", "✏️ Manage Database", "📊 Explore Database"])

    # Tab 1: Create Database
    with tab1:
        st.markdown("<h2 class='sub-header'>Create a New Database</h2>", unsafe_allow_html=True)
        st.write("Describe your database requirements in plain English")
        
        db_description = st.text_area(
            "Describe your database needs:",
            height=150,
            placeholder="e.g., 'I need a database for my software company with tables for employees, projects, clients, and assignments. Employees should have names, emails, positions, departments, salaries, and hire dates...'"
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
                        
                        with st.spinner("Creating database..."):
                            if execute_schema_creation(schema_sql, db_name):
                                st.success(f"Database '{db_name}' created successfully!")
                                try:
                                    engine = make_engine("SQLite (Demo)", sqlite_path=db_name)
                                    if engine:
                                        st.session_state.engine = engine
                                        st.session_state.schema_text = get_schema_text(engine)
                                        st.session_state.schema_dict = get_schema_dict(engine)
                                        st.session_state.current_db_path = db_name
                                        st.success("✅ Connected to your new database!")
                                    else:
                                        st.error("Failed to connect to the new database")
                                except Exception as e:
                                    st.error(f"Connected but encountered error: {e}")
                            else:
                                st.error("Failed to create database. Please check the schema for errors.")
                    else:
                        st.error(f"Failed to generate schema: {error}")

    # Tab 2: Manage Database
    with tab2:
        st.markdown("<h2 class='sub-header'>Manage Your Database</h2>", unsafe_allow_html=True)
        
        if not st.session_state.engine:
            st.warning("Please connect to a database first.")
        else:
            st.write("Use natural language to modify your database")
            
            with st.expander("📊 Current Database Schema", expanded=True):
                st.code(st.session_state.schema_text)
            
            management_query = st.text_area(
                "💬 What would you like to change?",
                height=100,
                placeholder="e.g., 'Add a new employee named Alex Johnson as Developer with email alex@company.com, salary $85000, department Engineering, hired today'"
            )
            
            if st.button("Execute Change", type="primary"):
                if not management_query:
                    st.error("Please enter a management query.")
                else:
                    with st.spinner("Generating and executing change..."):
                        ok, raw, cleaned = ask_model_to_sql_llm(
                            management_query, 
                            st.session_state.schema_text, 
                            st.session_state.read_only_mode
                        )
                        
                        if ok:
                            is_valid, message, norm_sql = validate_sql(cleaned, st.session_state.read_only_mode)
                            
                            if is_valid:
                                try:
                                    with st.session_state.engine.connect() as conn:
                                        result = conn.execute(text(norm_sql))
                                        conn.commit()
                                    
                                    st.success("✅ Change executed successfully!")
                                    
                                    # Refresh schema and data
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
        st.markdown("<h2 class='sub-header'>Explore Your Database</h2>", unsafe_allow_html=True)
        
        if not st.session_state.engine:
            st.warning("Please connect to a database first.")
        else:
            with st.expander("📊 Database Schema", expanded=True):
                st.code(st.session_state.schema_text)
            
            exploration_query = st.text_area(
                "💬 What would you like to explore?",
                height=100,
                placeholder="e.g., 'Show me all employees with their departments and salaries', 'List projects with their budgets and status', 'Find employees earning more than $80000'"
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
                                True
                            )
                            
                            if ok:
                                is_valid, message, norm_sql = validate_sql(cleaned, True)
                                
                                if is_valid:
                                    try:
                                        df = pd.read_sql_query(text(norm_sql), st.session_state.engine)
                                        st.session_state.exploration_df = df
                                        st.session_state.exploration_sql_executed = norm_sql
                                        
                                        st.success(f"✅ Query executed successfully! Returned {len(df)} rows.")
                                        st.dataframe(df, width='stretch')
                                        
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
                        
                        with st.spinner("Generating insights..."):
                            insights = generate_insights_from_data(df, "Query Results")
                            st.subheader("💡 Data Insights")
                            st.write(insights)
                        
                        st.subheader("📊 Visualization Options")
                        
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        if not numeric_cols and not categorical_cols:
                            st.warning("No suitable columns found for visualization.")
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
                                if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"] and numeric_cols:
                                    y_col = st.selectbox("Y Axis", numeric_cols, index=0)
                                else:
                                    y_col = None
                            
                            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"] and len(df.columns) > 2 and categorical_cols:
                                group_by = st.selectbox("Group By", [None] + categorical_cols)
                            else:
                                group_by = None
                            
                            title = st.text_input("Chart Title", value=f"{y_col or x_col} Analysis")
                            
                            if st.button("Generate Chart"):
                                fig = create_visualization(df, chart_type, x_col, y_col, group_by, title)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
