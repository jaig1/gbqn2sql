#!/usr/bin/env python3
"""
🧠 BigQuery Knowledge Graph-Enhanced Text-to-SQL System - Streamlit UI
=====================================================================

A comprehensive web interface for BigQuery-based text-to-SQL systems featuring:
- BigQuery Database Explorer (Direct SQL execution)
- Knowledge Graph Visualization (Coming Soon)
- Schema Context Viewer (Coming Soon) 
- Query Interface with AI Systems (Coming Soon)

This is the BigQuery equivalent of the SQLite-based streamlit_app.py

Requirements:
- streamlit
- google-cloud-bigquery
- python-dotenv
- All BigQuery text-to-SQL system dependencies

Usage:
streamlit run src/bq/GBQ_Streamlit_App.py
"""

import streamlit as st
import sys
import os
import pandas as pd
from typing import Dict, Optional
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import BigQuery client
try:
    from google.cloud import bigquery
    BQ_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import BigQuery client: {e}")
    BQ_AVAILABLE = False

# Import BigQuery Knowledge Graph Visualizer
try:
    from GBQ_KGVisualizer import render_kg_visualization_page
    GBQ_VISUALIZER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import BigQuery Knowledge Graph Visualizer: {e}")
    GBQ_VISUALIZER_AVAILABLE = False

# Import BigQuery Knowledge Graph Context Builder
try:
    from BQKnowledgeGraphContextBuilder import BQKnowledgeGraphContextBuilder
    BQ_CONTEXT_BUILDER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import BigQuery Knowledge Graph Context Builder: {e}")
    BQ_CONTEXT_BUILDER_AVAILABLE = False

# Import GeminiKGText2SQL for AI query processing
try:
    from GeminiKGText2SQL import GeminiKGText2SQL
    GEMINI_KG_TEXT2SQL_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import GeminiKGText2SQL: {e}")
    GEMINI_KG_TEXT2SQL_AVAILABLE = False

# Import GeminiText2SQL for Basic System
try:
    from GeminiText2SQL import GeminiTextToSQLBigQuery
    GEMINI_TEXT2SQL_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import GeminiText2SQL: {e}")
    GEMINI_TEXT2SQL_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🧠 BigQuery Knowledge Graph-Enhanced Text-to-SQL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_bigquery_requirements() -> Dict[str, bool]:
    """Check if all BigQuery requirements are met"""
    checks = {
        "gcp_project": bool(os.getenv('GCP_PROJECT_ID')),
        "bq_dataset": bool(os.getenv('BQ_DATASET_ID')),
        "bigquery_available": BQ_AVAILABLE,
        "gcloud_auth": True,  # Assume gcloud auth is set up
        "kg_visualizer": GBQ_VISUALIZER_AVAILABLE,
        "kg_file": os.path.exists("bqkg/bq_knowledge_graph.ttl"),
        "schema_builder": BQ_CONTEXT_BUILDER_AVAILABLE,
        "gemini_kg_text2sql": GEMINI_KG_TEXT2SQL_AVAILABLE,
        "gemini_text2sql": GEMINI_TEXT2SQL_AVAILABLE
    }
    return checks

def get_bigquery_client():
    """Initialize BigQuery client with environment variables"""
    try:
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            st.error("GCP_PROJECT_ID environment variable not set")
            return None
        
        client = bigquery.Client(project=project_id)
        return client
    except Exception as e:
        st.error(f"Failed to initialize BigQuery client: {e}")
        return None

def execute_bigquery_query(sql_query: str) -> Dict:
    """Execute a BigQuery SQL query"""
    result = {
        "success": False,
        "results": None,
        "row_count": 0,
        "execution_time_ms": None,
        "error": None,
        "columns": None,
        "bytes_processed": None,
        "job_id": None
    }
    
    try:
        start_time = time.time()
        client = get_bigquery_client()
        
        if not client:
            result["error"] = "BigQuery client not available"
            return result
        
        # Configure query job
        job_config = bigquery.QueryJobConfig()
        job_config.use_query_cache = True
        job_config.use_legacy_sql = False
        
        # Execute query
        query_job = client.query(sql_query, job_config=job_config)
        
        # Wait for job to complete and get results
        results = query_job.result()
        
        # Convert to list of dictionaries
        results_list = []
        columns = [field.name for field in results.schema]
        
        for row in results:
            row_dict = {}
            for i, value in enumerate(row):
                row_dict[columns[i]] = value
            results_list.append(row_dict)
        
        execution_time = (time.time() - start_time) * 1000
        
        result.update({
            "success": True,
            "results": results_list,
            "row_count": len(results_list),
            "execution_time_ms": execution_time,
            "columns": columns,
            "bytes_processed": query_job.total_bytes_processed,
            "job_id": query_job.job_id
        })
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
        result.update({
            "error": str(e),
            "execution_time_ms": execution_time
        })
    
    return result

def get_bigquery_sample_queries() -> Dict[str, str]:
    """Get sample BigQuery queries for the insurance_analytics dataset"""
    dataset_id = os.getenv('BQ_DATASET_ID', 'insurance_analytics')
    project_id = os.getenv('GCP_PROJECT_ID', 'your-project-id')
    
    return {
        "project_info": f"""-- Check current project information
SELECT 
    @@project_id as current_project,
    CURRENT_TIMESTAMP() as current_time,
    CURRENT_DATE() as current_date;""",

        "dataset_info": f"""-- List all datasets in your project
SELECT 
    schema_name as dataset_name,
    location,
    creation_time
FROM `{project_id}.INFORMATION_SCHEMA.SCHEMATA`
ORDER BY creation_time DESC;""",

        "table_info": f"""-- List all tables in the insurance_analytics dataset
SELECT 
    table_name,
    table_type,
    creation_time
FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
ORDER BY creation_time DESC;""",

        "agents": f"""-- List all insurance agents
SELECT 
    agent_id, 
    name, 
    email, 
    phone 
FROM `{project_id}.{dataset_id}.insurance_agents` 
ORDER BY name
LIMIT 10;""",
        
        "customers": f"""-- Show customer details with age calculation
SELECT 
    customer_id,
    name,
    email,
    phone,
    address,
    date_of_birth,
    DATE_DIFF(CURRENT_DATE(), date_of_birth, YEAR) as age_years
FROM `{project_id}.{dataset_id}.insurance_customers` 
ORDER BY name
LIMIT 10;""",
        
        "policies": f"""-- Display policies with customer and agent names
SELECT 
    p.policy_id,
    p.policy_number,
    c.name as customer_name,
    a.name as agent_name,
    p.policy_type,
    p.premium_amount,
    p.start_date,
    p.end_date,
    p.status
FROM `{project_id}.{dataset_id}.insurance_policies` p
JOIN `{project_id}.{dataset_id}.insurance_customers` c ON p.customer_id = c.customer_id
JOIN `{project_id}.{dataset_id}.insurance_agents` a ON p.agent_id = a.agent_id
ORDER BY p.start_date DESC
LIMIT 10;""",
        
        "claims": f"""-- Show claims with policy and customer information
SELECT 
    cl.claim_id,
    cl.claim_number,
    c.name as customer_name,
    p.policy_number,
    p.policy_type,
    cl.claim_date,
    cl.claim_amount,
    cl.claim_status,
    cl.description
FROM `{project_id}.{dataset_id}.insurance_claims` cl
JOIN `{project_id}.{dataset_id}.insurance_policies` p ON cl.policy_id = p.policy_id
JOIN `{project_id}.{dataset_id}.insurance_customers` c ON p.customer_id = c.customer_id
ORDER BY cl.claim_date DESC
LIMIT 10;""",
        
        "table_counts": f"""-- Get row counts for all tables
SELECT 
    table_name, 
    COUNT(*) as row_count 
FROM (
    SELECT 'agents' as table_name FROM `{project_id}.{dataset_id}.insurance_agents`
    UNION ALL
    SELECT 'customers' as table_name FROM `{project_id}.{dataset_id}.insurance_customers`
    UNION ALL
    SELECT 'policies' as table_name FROM `{project_id}.{dataset_id}.insurance_policies`
    UNION ALL
    SELECT 'claims' as table_name FROM `{project_id}.{dataset_id}.insurance_claims`
) 
GROUP BY table_name 
ORDER BY table_name;""",
        
        "customer_analysis": f"""-- Customer policy and claims analysis
SELECT 
    c.name as customer_name,
    COUNT(DISTINCT p.policy_id) as total_policies,
    ROUND(SUM(p.premium_amount), 2) as total_premiums,
    COUNT(DISTINCT cl.claim_id) as total_claims,
    ROUND(COALESCE(SUM(cl.claim_amount), 0), 2) as total_claim_amount
FROM `{project_id}.{dataset_id}.insurance_customers` c
LEFT JOIN `{project_id}.{dataset_id}.insurance_policies` p ON c.customer_id = p.customer_id
LEFT JOIN `{project_id}.{dataset_id}.insurance_claims` cl ON p.policy_id = cl.policy_id
GROUP BY c.customer_id, c.name
HAVING total_policies > 0
ORDER BY total_premiums DESC
LIMIT 15;""",

        "agent_performance": f"""-- Agent performance analysis
SELECT 
    a.name as agent_name,
    COUNT(DISTINCT p.policy_id) as policies_sold,
    ROUND(AVG(p.premium_amount), 2) as avg_policy_value,
    COUNT(DISTINCT cl.claim_id) as claims_handled
FROM `{project_id}.{dataset_id}.insurance_agents` a
LEFT JOIN `{project_id}.{dataset_id}.insurance_policies` p ON a.agent_id = p.agent_id
LEFT JOIN `{project_id}.{dataset_id}.insurance_claims` cl ON p.policy_id = cl.policy_id
GROUP BY a.agent_id, a.name
ORDER BY policies_sold DESC
LIMIT 10;"""
    }

def database_explorer_page():
    """BigQuery Database Explorer page for direct SQL interaction"""
    st.title("🗃️ BigQuery Database Explorer")
    st.markdown("Explore the BigQuery insurance_analytics dataset with direct SQL queries")
    
    # Environment info
    project_id = os.getenv('GCP_PROJECT_ID', 'Not Set')
    dataset_id = os.getenv('BQ_DATASET_ID', 'Not Set')
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📋 **Project:** `{project_id}`")
    with col2:
        st.info(f"🗄️ **Dataset:** `{dataset_id}`")
    
    # BigQuery schema reference
    with st.expander("📊 BigQuery Dataset Schema Reference"):
        
        # Check if tables exist first
        st.info("💡 **First Time Setup**: If tables don't exist, try the 'Check Dataset' and 'Check Tables' sample queries first to verify your dataset setup.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏢 insurance_agents")
            st.code("""agent_id (INTEGER)
name (STRING)
email (STRING)
phone (STRING)""", language="sql")
            
            st.subheader("👥 insurance_customers")
            st.code("""customer_id (INTEGER)
name (STRING)
email (STRING)
phone (STRING)
address (STRING)
date_of_birth (DATE)""", language="sql")
        
        with col2:
            st.subheader("📋 insurance_policies")
            st.code("""policy_id (INTEGER)
policy_number (STRING)
customer_id (INTEGER)
agent_id (INTEGER)
policy_type (STRING)
premium_amount (NUMERIC)
start_date (DATE)
end_date (DATE)
status (STRING)""", language="sql")
            
            st.subheader("💰 insurance_claims")
            st.code("""claim_id (INTEGER)
claim_number (STRING)
policy_id (INTEGER)
claim_date (DATE)
claim_amount (NUMERIC)
claim_status (STRING)
description (STRING)""", language="sql")
    
    st.divider()
    
    # Table information help
    with st.expander("🛠️ Dataset Information & Tips"):
        st.markdown("""
        **Your BigQuery Dataset is Ready!**
        
        ✅ **Dataset**: `insurance_analytics` (Location: US)  
        ✅ **Tables Available**:
        - `insurance_agents` (24 rows)
        - `insurance_customers` (45 rows)  
        - `insurance_policies` (68 rows)
        - `insurance_claims` (50 rows)
        
        **Quick Start Tips**:
        1. Try the **"ℹ️ Check current project"** query first
        2. Use **"🔢 Get row counts"** to see data volume  
        3. Start with simple table queries before complex joins
        4. All queries are limited to 10 rows for performance
        """)
    
    
    st.divider()
    
    # SQL Editor
    st.subheader("✍️ BigQuery SQL Editor")
    
    # Handle auto-execution from sample queries
    default_sql = ""
    
    if 'execute_sample_sql' in st.session_state:
        sample_key = st.session_state.execute_sample_sql
        sample_queries = get_bigquery_sample_queries()
        
        # Clear the execution flag
        del st.session_state.execute_sample_sql
        
        # Set the text area to show the sample query
        st.session_state.selected_sample_sql = sample_queries[sample_key]
        
        # Store the sample query execution info for later display
        st.session_state.pending_sample_execution = {
            'query': sample_queries[sample_key],
            'sample_key': sample_key
        }
        
        # Force rerun to populate text area and execute
        st.rerun()
    
    # Handle session state for SQL text area
    if 'selected_sample_sql' in st.session_state:
        default_sql = st.session_state.selected_sample_sql
        del st.session_state.selected_sample_sql
    
    sql_query = st.text_area(
        "Enter your BigQuery SQL query:",
        value=default_sql,
        key="sql_editor",
        height=200,
        help="Write your BigQuery SQL query here. Uses Standard SQL syntax.",
        placeholder=f"""Example:
SELECT c.name, COUNT(p.policy_id) as policy_count
FROM `{project_id}.{dataset_id}.customers` c
LEFT JOIN `{project_id}.{dataset_id}.policies` p ON c.customer_id = p.customer_id
GROUP BY c.customer_id, c.name
ORDER BY policy_count DESC
LIMIT 10;"""
    )
    
    # Sample BigQuery SQL Queries
    with st.expander("💡 Sample BigQuery SQL Queries"):
        sample_queries = get_bigquery_sample_queries()
        
        # Create a more descriptive list of samples
        sample_descriptions = {
            "project_info": "ℹ️ Check current project and time information",
            "dataset_info": "📊 List all datasets in your project",
            "table_info": "📋 List all tables in the insurance_analytics dataset",
            "table_counts": "🔢 Get row counts for all tables",
            "agents": "🏢 List all insurance agents", 
            "customers": "👥 Show customer details with age calculation",
            "policies": "📋 Display policies with customer and agent names",
            "claims": "💰 Show claims with policy and customer information",
            "customer_analysis": "🔗 Customer policy and claims analysis",
            "agent_performance": "🏆 Agent performance analysis"
        }
        
        for key, description in sample_descriptions.items():
            if st.button(f"📝 {description}", key=f"sample_sql_{hash(key)}"):
                st.session_state.execute_sample_sql = key
                st.rerun()
    
    # BigQuery-specific notice
    st.info("🔒 **BigQuery Notice**: This executes queries against your BigQuery dataset. Standard SQL syntax required. Monitor your BigQuery usage and costs.")
    
    # Check for execution (manual button click or pending sample execution)
    execute_manual = st.button("🚀 Execute BigQuery SQL", type="primary") and sql_query.strip()
    execute_sample = 'pending_sample_execution' in st.session_state
    
    # Handle execution for both manual and sample queries
    if execute_manual or execute_sample:
        
        # Determine query source and content
        if execute_sample:
            # Sample query execution
            sample_info = st.session_state.pending_sample_execution
            query_to_execute = sample_info['query']
            execution_source = f"🎯 Sample Query: {sample_info['sample_key'].replace('_', ' ').title()}"
            
            # Clear the pending execution
            del st.session_state.pending_sample_execution
            
            # Inform user that sample query is now in editor
            st.info("💡 The sample query has been copied to the SQL editor above for further editing.")
        else:
            # Manual execution
            query_to_execute = sql_query
            execution_source = "🚀 Manual Execution"
        
        # Display execution source
        if execution_source.startswith("🎯"):
            st.subheader(f"{execution_source} Results")
        
        # Execute the BigQuery query
        sql_upper = query_to_execute.upper().strip()
        dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        
        if any(keyword in sql_upper for keyword in dangerous_keywords):
            st.error("❌ **Safety Error**: Modifying operations (INSERT, UPDATE, DELETE, etc.) are not allowed for data safety.")
        else:
            with st.spinner("🔄 Executing BigQuery SQL query..."):
                result = execute_bigquery_query(query_to_execute)
            
            if result["success"]:
                st.success("✅ BigQuery query executed successfully!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows Returned", result["row_count"])
                with col2:
                    st.metric("Execution Time", f"{result['execution_time_ms']:.2f}ms")
                with col3:
                    if result["bytes_processed"]:
                        bytes_mb = result["bytes_processed"] / (1024 * 1024)
                        st.metric("Data Processed", f"{bytes_mb:.2f} MB")
                with col4:
                    if result["job_id"]:
                        st.metric("Job ID", result["job_id"][:8] + "...")
                
                # Display results
                if result["results"]:
                    st.subheader("📊 BigQuery Query Results")
                    df = pd.DataFrame(result["results"])
                    st.dataframe(df, width="stretch")
                    
                    # Option to download results
                    csv = df.to_csv(index=False)
                    if execute_sample:
                        filename = f"bigquery_{sample_info['sample_key']}_results.csv"
                    else:
                        filename = "bigquery_query_results.csv"
                    
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )
                else:
                    st.info("Query executed successfully but returned no results.")
                    
            else:
                st.error(f"❌ **BigQuery Error**: {result['error']}")
                
                # Show common error hints
                error_msg = result['error'].lower()
                if "not found" in error_msg and ("table" in error_msg or "dataset" in error_msg):
                    st.warning("🚨 **Table/Dataset Not Found Error**")
                    st.info(f"""💡 **Solution Options**:
                    
**Option 1: Check What Exists**
• Try the '🔍 Check Dataset' query to verify your dataset exists
• Try the '📋 Check Tables' query to see available tables

**Option 2: Create Sample Data**
• The insurance tables (agents, customers, policies, claims) need to be created first
• Contact your BigQuery administrator for table creation
                    
**Option 3: Use Different Tables**
• Modify the query to use existing tables in your `{project_id}.{dataset_id}` dataset
• Check what tables are available using the INFORMATION_SCHEMA queries
                    """)
                elif "syntax error" in error_msg or "invalid" in error_msg:
                    st.info("💡 **Hint**: Check your BigQuery SQL syntax. Common issues include missing commas, incorrect table references, or unsupported functions.")
                elif "permission" in error_msg or "access" in error_msg:
                    st.info("💡 **Hint**: Permission error. Check your BigQuery access rights and gcloud authentication.")
                elif "quota" in error_msg or "limit" in error_msg:
                    st.info("💡 **Hint**: BigQuery quota exceeded. Try reducing the query scope or wait for quota reset.")
                else:
                    st.info("💡 **General Hint**: Try the 'Check Dataset' and 'Check Tables' queries first to understand your BigQuery setup.")

@st.cache_resource
def initialize_gemini_kg_system():
    """Initialize and cache the GeminiKGText2SQL system"""
    try:
        project_id = os.getenv('GCP_PROJECT_ID')
        dataset_id = os.getenv('BQ_DATASET_ID')
        
        if not project_id or not dataset_id:
            return None, "Missing GCP_PROJECT_ID or BQ_DATASET_ID environment variables"
        
        if not os.path.exists("bqkg/bq_knowledge_graph.ttl"):
            return None, "BigQuery Knowledge Graph file not found"
        
        # Read model from environment variable, fallback to default
        model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash')
        
        system = GeminiKGText2SQL(
            project_id=project_id,
            dataset_id=dataset_id,
            bq_kg_file='bqkg/bq_knowledge_graph.ttl',
            model=model_name
        )
        
        return system, None
    except Exception as e:
        return None, f"Failed to initialize GeminiKGText2SQL: {str(e)}"

def initialize_basic_gemini_system():
    """Initialize the basic GeminiText2SQL system (without Knowledge Graph)"""
    try:
        project_id = os.getenv('GCP_PROJECT_ID')
        dataset_id = os.getenv('BQ_DATASET_ID')
        vertex_ai_location = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
        model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash-lite')
        
        if not project_id or not dataset_id:
            return None, "Missing GCP_PROJECT_ID or BQ_DATASET_ID environment variables"
        
        system = GeminiTextToSQLBigQuery(
            project_id=project_id,
            dataset_id=dataset_id,
            vertex_ai_location=vertex_ai_location,
            model_name=model_name
        )
        
        return system, None
    except Exception as e:
        return None, f"Failed to initialize GeminiText2SQL: {str(e)}"

def process_basic_system_query(basic_system, question: str) -> Dict:
    """
    Process query using Basic System and convert to UI-compatible format
    
    Args:
        basic_system: GeminiTextToSQLBigQuery instance
        question: Natural language question
    
    Returns:
        Dictionary compatible with display_kg_query_results format
    """
    try:
        # Use the generate_sql method to get enhanced metadata
        sql_query, gen_metadata = basic_system.generate_sql(question)
        
        # Extract enhanced explanation, confidence, and reasoning from metadata
        explanation = gen_metadata.get('explanation', f"Schema-only SQL generation for: {question}")
        confidence = gen_metadata.get('confidence', 'medium')
        reasoning = gen_metadata.get('reasoning', "Generated using basic BigQuery schema context without Knowledge Graph enhancement.")
        
        # If SQL generation was successful, execute the query
        if sql_query and 'error' not in gen_metadata:
            try:
                results, exec_metadata = basic_system.execute_sql(sql_query)
                execution_success = 'error' not in exec_metadata
                error_msg = exec_metadata.get('error') if not execution_success else None
            except Exception as e:
                results = []
                exec_metadata = {}
                execution_success = False
                error_msg = f"Execution error: {str(e)}"
        else:
            # SQL generation failed
            results = []
            exec_metadata = {}
            execution_success = False
            error_msg = gen_metadata.get('error', 'Failed to generate SQL')
        
        # Convert to UI format
        ui_result = {
            'system': 'Basic (Schema-Only)',
            'confidence': confidence,
            'sql_query': sql_query,
            'explanation': explanation,
            'reasoning': reasoning,
            'success': execution_success,
            'execution_results': results,
            'row_count': len(results),
            'execution_time_ms': exec_metadata.get('execution_time_ms', 0),
            'bytes_processed': exec_metadata.get('bytes_processed'),
            'job_id': exec_metadata.get('job_id'),
            'error': error_msg
        }
        
        return ui_result
        
    except Exception as e:
        return {
            'system': 'Basic (Schema-Only)',
            'confidence': 'low',
            'sql_query': '',
            'explanation': f"Error processing question: {str(e)}",
            'reasoning': 'System encountered an unexpected error during processing.',
            'success': False,
            'execution_results': [],
            'row_count': 0,
            'execution_time_ms': 0,
            'error': f"Basic system error: {str(e)}"
        }

def display_kg_query_results(result: Dict, question: str, system_name: str = "KG-Enhanced"):
    """Display results from query processing - works for both KG-Enhanced and Basic Schema Context"""
    
    st.subheader(f"{system_name} Results")
    st.info(f"**Question**: \"{question}\"")
    
    # Show system info
    confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(result.get('confidence', 'low'), "⚪")
    st.write(f"**System**: {result.get('system', system_name)} | **Confidence**: {confidence_color} {result.get('confidence', 'unknown')}")
    
    # Display generated SQL
    if result.get('sql_query'):
        st.subheader("🔍 Generated BigQuery SQL")
        st.code(result['sql_query'], language='sql', wrap_lines=True)
        
        if result.get('explanation'):
            st.write(f"**💡 Explanation**: {result['explanation']}")
        
        if result.get('reasoning'):
            with st.expander("🧩 View AI Reasoning"):
                # Format reasoning with numbered points on separate lines
                reasoning_text = result['reasoning']
                # Split on numbered patterns and rejoin with newlines
                import re
                formatted_reasoning = re.sub(r'(\d+\.)', r'\n\1', reasoning_text).strip()
                st.write(formatted_reasoning)
    
    # Display execution results
    if result.get('success'):
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", result.get('row_count', 0))
        with col2:
            exec_time = result.get('execution_time_ms', 0)
            st.metric("Time", f"{exec_time:.1f}ms" if exec_time else "N/A")
        with col3:
            if result.get('bytes_processed'):
                bytes_mb = result['bytes_processed'] / (1024 * 1024)
                st.metric("Data", f"{bytes_mb:.2f}MB")
            else:
                st.metric("Data", "N/A")
        with col4:
            st.metric("Status", "✅ Success")
        
        # Results table
        st.subheader("Query Results")
        
        # Check if we have results
        execution_results = result.get('execution_results', [])
        row_count = result.get('row_count', 0)
        
        if execution_results and row_count > 0:
            # Display results table
            df = pd.DataFrame(execution_results)
            st.dataframe(df, width="stretch")
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"bigquery_kg_results_{int(time.time())}.csv",
                mime="text/csv"
            )
        else:
            # No results returned
            st.info("**No rows returned** - The query executed successfully but returned no data.")
            st.markdown("💡 **Possible reasons:**")
            st.markdown("- The filters in your query are too restrictive")  
            st.markdown("- The data you're looking for doesn't exist in the database")
            st.markdown("- Try broadening your search criteria or checking different tables")
    
    elif result.get('error'):
        st.error(f"❌ **Error**: {result['error']}")
        
        # Error-specific hints
        error_msg = result['error'].lower()
        if "permission" in error_msg or "access" in error_msg:
            st.info("💡 **Hint**: Check BigQuery permissions and gcloud authentication")
        elif "syntax" in error_msg:
            st.info("💡 **Hint**: SQL syntax error - try rephrasing your question")
        elif "not found" in error_msg:
            st.info("💡 **Hint**: Table or column not found - check schema context")
        elif "quota" in error_msg or "limit" in error_msg:
            st.info("💡 **Hint**: BigQuery quota exceeded - try reducing query scope")
        else:
            st.info("💡 **Hint**: Try asking simpler questions or check system requirements")
    
    st.divider()

def display_side_by_side_comparison(systems: Dict, question: str):
    """Display side-by-side comparison of KG-Enhanced vs Basic systems"""
    
    st.subheader("⚖️ Side-by-Side Comparison Results")
    st.info(f"**Question**: \"{question}\"")
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Initialize results containers
    kg_result = None
    basic_result = None
    
    # Process with both systems
    with col1:
        st.markdown("### KG-Enhanced Context")
        if 'kg' in systems:
            with st.spinner("Processing with KG-Enhanced context..."):
                kg_result = systems['kg'].process_query_for_ui(question)
            
            # Display KG results in first column
            display_comparison_results(kg_result, "KG-Enhanced", question)
        else:
            st.error("❌ KG-Enhanced context not available")
    
    with col2:
        st.markdown("### Basic Schema Context")
        if 'basic' in systems:
            with st.spinner("Processing with Basic schema context..."):
                basic_result = process_basic_system_query(systems['basic'], question)
            
            # Display Basic results in second column
            display_comparison_results(basic_result, "Basic Schema", question)
        else:
            st.error("❌ Basic schema context not available")
    
    # Comparison summary table
    if kg_result and basic_result:
        st.divider()
        st.subheader("📊 Comparison Summary")
        
        # Create comparison metrics
        comparison_data = {
            "Metric": ["Success", "Confidence", "SQL Length", "Execution Time", "Results Count"],
            "KG-Enhanced": [
                "✅" if kg_result.get('success') else "❌",
                kg_result.get('confidence', 'unknown'),
                len(kg_result.get('sql_query', '')) if kg_result.get('sql_query') else 0,
                f"{kg_result.get('execution_time_ms', 0):.2f}ms",
                kg_result.get('row_count', 0)
            ],
            "Basic Schema": [
                "✅" if basic_result.get('success') else "❌", 
                basic_result.get('confidence', 'unknown'),
                len(basic_result.get('sql_query', '')) if basic_result.get('sql_query') else 0,
                f"{basic_result.get('execution_time_ms', 0):.2f}ms",
                basic_result.get('row_count', 0)
            ]
        }
        
        # Display comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Analysis insights
        st.subheader("🔍 Analysis Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if kg_result.get('success') and basic_result.get('success'):
                if kg_result.get('row_count', 0) == basic_result.get('row_count', 0):
                    st.success("✅ **Same Result Count** - Both systems returned identical row counts")
                else:
                    st.warning("**Different Results** - Contexts returned different row counts")
            elif kg_result.get('success') and not basic_result.get('success'):
                st.info("**KG Advantage** - Only KG-Enhanced context succeeded")
            elif not kg_result.get('success') and basic_result.get('success'):
                st.info("**Basic Success** - Only Basic schema context succeeded")
            else:
                st.error("**Both Failed** - Neither context generated valid results")
        
        with col2:
            kg_sql_len = len(kg_result.get('sql_query', ''))
            basic_sql_len = len(basic_result.get('sql_query', ''))
            if kg_sql_len > basic_sql_len * 1.2:
                st.info("**More Complex SQL** - KG context generated more detailed query")
            elif basic_sql_len > kg_sql_len * 1.2:
                st.info("**Simpler SQL** - Basic context generated more concise query")
            else:
                st.success("⚖️ **Similar Complexity** - Both queries have similar length")
        
        with col3:
            kg_time = kg_result.get('execution_time_ms', 0)
            basic_time = basic_result.get('execution_time_ms', 0)
            if kg_time > 0 and basic_time > 0:
                if kg_time < basic_time * 0.8:
                    st.success("**Faster KG** - KG context executed faster")
                elif basic_time < kg_time * 0.8:
                    st.success("**Faster Basic** - Basic context executed faster")
                else:
                    st.info("**Similar Speed** - Both contexts performed similarly")
            else:
                st.info("**Timing Data** - Execution times not available")

def display_comparison_results(result: Dict, system_name: str, question: str):
    """Display results in comparison format (more compact than full display)"""
    
    # System header with confidence
    confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(result.get('confidence', 'low'), "⚪")
    st.write(f"**Confidence**: {confidence_color} {result.get('confidence', 'unknown')}")
    
    # SQL Query (if available)
    if result.get('sql_query'):
        with st.expander("🔍 View Generated SQL"):
            st.code(result['sql_query'], language='sql', wrap_lines=True)
        
        if result.get('explanation'):
            st.write(f"**💡 Explanation**: {result['explanation']}")
        
        if result.get('reasoning'):
            with st.expander("🧩 View AI Reasoning"):
                # Format reasoning with numbered points on separate lines
                reasoning_text = result['reasoning']
                # Split on numbered patterns and rejoin with newlines
                import re
                formatted_reasoning = re.sub(r'(\d+\.)', r'\n\1', reasoning_text).strip()
                st.write(formatted_reasoning)
    
    # Results summary
    if result.get('success'):
        # Compact metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", result.get('row_count', 0))
        with col2:
            exec_time = result.get('execution_time_ms', 0)
            st.metric("Time", f"{exec_time:.1f}ms" if exec_time else "N/A")
        
        # Show results or empty message
        execution_results = result.get('execution_results', [])
        row_count = result.get('row_count', 0)
        
        if execution_results and row_count > 0:
            # Show first few results
            df = pd.DataFrame(execution_results)
            st.write("**Sample Results:**")
            # Show first 3 rows max for comparison view
            display_df = df.head(3)
            st.dataframe(display_df, use_container_width=True)
            
            if len(df) > 3:
                st.caption(f"... and {len(df) - 3} more rows")
        else:
            # No results returned
            st.info("**No rows returned** - Query executed successfully but returned no data.")
        
    elif result.get('error'):
        st.error(f"❌ **Error**: {result['error']}")
        
        # Brief error hints
        error_msg = result['error'].lower()
        if "permission" in error_msg or "access" in error_msg:
            st.caption("💡 Check BigQuery permissions")
        elif "syntax" in error_msg:
            st.caption("💡 SQL syntax error")
        else:
            st.caption("💡 System processing error")

def query_interface_page():
    """BigQuery AI Query Interface page with System Selection"""
    st.title("Query Interface")
    st.markdown("Natural language to BigQuery SQL Generation with System Comparison")
    
    # Check requirements
    if not GEMINI_KG_TEXT2SQL_AVAILABLE and not GEMINI_TEXT2SQL_AVAILABLE:
        st.error("❌ No AI Text2SQL systems available")
        st.info("Please ensure at least one of GeminiKGText2SQL.py or GeminiText2SQL.py is properly installed.")
        return
    
    # Initialize systems
    systems = {}
    
    # Initialize KG-Enhanced system
    if GEMINI_KG_TEXT2SQL_AVAILABLE and os.path.exists("bqkg/bq_knowledge_graph.ttl"):
        with st.spinner("🔄 Initializing Knowledge Graph-Enhanced system..."):
            kg_system, kg_error = initialize_gemini_kg_system()
        if kg_system:
            systems['kg'] = kg_system
        elif kg_error:
            st.warning(f"⚠️ KG-Enhanced system: {kg_error}")
    
    # Initialize Basic system
    if GEMINI_TEXT2SQL_AVAILABLE:
        with st.spinner("🔄 Initializing Basic system..."):
            basic_system, basic_error = initialize_basic_gemini_system()
        if basic_system:
            systems['basic'] = basic_system
        elif basic_error:
            st.warning(f"⚠️ Basic system: {basic_error}")
    
    if not systems:
        st.error("❌ Failed to initialize any AI systems")
        return
    
    # System selection
    st.subheader("System Selection")
    
    available_modes = []
    if 'kg' in systems:
        available_modes.append("KG-Enhanced Context")
    if 'basic' in systems:
        available_modes.append("Basic Schema Context")
    if 'kg' in systems and 'basic' in systems:
        available_modes.append("Side-by-Side Comparison")
    
    mode = st.radio(
        "Choose operation mode:",
        available_modes,
        horizontal=True
    )
    
    # Context comparison (only if both systems available)
    if 'kg' in systems and 'basic' in systems:
        try:
            kg_info = systems['kg'].get_system_info()
            basic_info = {"context_length": 5000}  # Approximate basic context length
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("KG-Enhanced Context", f"{kg_info.get('context_length', 0):,} chars")
            with col2:
                st.metric("Basic Context", f"{basic_info['context_length']:,} chars")
            with col3:
                if kg_info.get('context_length') and basic_info['context_length']:
                    ratio = kg_info['context_length'] / basic_info['context_length']
                    st.metric("Enhancement Ratio", f"{ratio:.1f}x richer")
        except Exception as e:
            st.info("Context metrics unavailable")
    
    st.divider()
    
    # 1. Query input section
    st.subheader("Ask Your Question")
    
    # 2. Text area for accepting user questions
    question = st.text_area(
        "Enter your natural language question about the BigQuery insurance database:",
        placeholder="e.g., 'Show me customers with multiple policies' or 'What's the average premium for life insurance?'",
        height=100,
        key="ai_question_input"
    )
    
    # 3. Example questions dropdown (ABOVE the button)
    with st.expander("Example Questions"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simple Questions Sample")
            simple_examples = [
                "Show me all customers with their email addresses",
                "Find customers with multiple policies",
                "What's the average premium for life insurance?",
                "Which agent has the best claims-to-premium ratio?",
                "Find high-premium customers with low claims",
                "Show me policy performance metrics"
            ]
            
            for example in simple_examples:
                if st.button(f"{example}", key=f"ai_simple_{hash(example)}"):
                    st.session_state.ai_execute_example = example
                    st.rerun()
        
        with col2:
            st.subheader("Complex Questions Sample")
            complex_examples = [
                "Show me all active policies that are currently in force",
                "Show all approved claims ready for payment",
                "Analyze policies in the budget segment",
                "Find all large loss events that require special handling",
                "Show customers in the premium tier segment",
                "List policies that are expiring soon",
                "Identify top performing agents",
                "Show standard tier policies with claim patterns"
            ]
            
            for example in complex_examples:
                if st.button(f"{example}", key=f"ai_complex_{hash(example)}"):
                    st.session_state.ai_execute_example = example
                    st.rerun()
    
    # 4. Generate SQL button
    execute_manual_query = st.button("Generate SQL", type="primary") and question.strip()
    
    # 5. Results display area (below the button)
    # Handle example execution from session state
    if 'ai_execute_example' in st.session_state:
        example_question = st.session_state.ai_execute_example
        del st.session_state.ai_execute_example
        
        # Process the example question based on selected mode
        if mode == "KG-Enhanced Context" and 'kg' in systems:
            with st.spinner(f"Processing example with KG-Enhanced context: {example_question}"):
                result = systems['kg'].process_query_for_ui(example_question)
            display_kg_query_results(result, example_question, "KG-Enhanced Context")
            
        elif mode == "Basic Schema Context" and 'basic' in systems:
            with st.spinner(f"Processing example with Basic schema context: {example_question}"):
                result = process_basic_system_query(systems['basic'], example_question)
            display_kg_query_results(result, example_question, "Basic Schema Context")
            
        elif mode == "Side-by-Side Comparison" and 'kg' in systems and 'basic' in systems:
            display_side_by_side_comparison(systems, example_question)
    
    # Handle manual query execution
    if execute_manual_query:
        if mode == "KG-Enhanced Context" and 'kg' in systems:
            with st.spinner("Processing with Knowledge Graph-Enhanced context..."):
                result = systems['kg'].process_query_for_ui(question.strip())
            display_kg_query_results(result, question.strip(), "KG-Enhanced Context")
            
        elif mode == "Basic Schema Context" and 'basic' in systems:
            with st.spinner("Processing with Basic schema context..."):
                result = process_basic_system_query(systems['basic'], question.strip())
            display_kg_query_results(result, question.strip(), "Basic Schema Context")
            
        elif mode == "Side-by-Side Comparison" and 'kg' in systems and 'basic' in systems:
            display_side_by_side_comparison(systems, question.strip())

def schema_viewer_page():
    """Schema viewer page with BigQuery Knowledge Graph context"""
    st.title("📋 System Context Builder")
    
    # Check if requirements are met
    if not BQ_CONTEXT_BUILDER_AVAILABLE:
        st.error("❌ BigQuery Knowledge Graph Context Builder not available")
        st.info("Please ensure BQKnowledgeGraphContextBuilder.py is properly installed.")
        return
    
    if not os.path.exists("bqkg/bq_knowledge_graph.ttl"):
        st.error("❌ BigQuery Knowledge Graph file not found")
        st.info("Please ensure bqkg/bq_knowledge_graph.ttl file exists.")
        st.code("Expected file: bqkg/bq_knowledge_graph.ttl")
        return
    
    # Initialize the context builder
    try:
        with st.spinner("🔄 Loading BigQuery Knowledge Graph..."):
            kg_context_builder = BQKnowledgeGraphContextBuilder("bqkg/bq_knowledge_graph.ttl")
            schema_context = kg_context_builder.extract_schema_context()
    except Exception as e:
        st.error(f"❌ Failed to load BigQuery Knowledge Graph: {e}")
        return
    
    # Create tabs for different schema views
    tab1, tab2 = st.tabs(["🧠 KG-Enhanced Schema Context", "🔧 Basic Schema Context"])
    
    with tab1:
        st.subheader("🧠 Knowledge Graph-Enhanced Context")
        st.markdown("Knowledge Graph-powered schema enrichment providing semantic annotations, data sensitivity analysis, business rule validation, and canonical relationship patterns to optimize LLM understanding and SQL accuracy.")
        
        # Full context viewer
        with st.expander("📖 View Full KG-Enhanced Context"):
            try:
                full_context = kg_context_builder.build_llm_context()
                st.text_area(
                    "Complete Schema Context",
                    value=full_context,
                    height=800,
                    help="This is the complete context that would be sent to AI systems"
                )
                st.metric("Context Length", f"{len(full_context):,} characters")
            except Exception as e:
                st.error(f"Failed to build full context: {e}")
    
    with tab2:
        st.subheader("🔧 Basic Schema Context")
        st.markdown("Standard BigQuery schema information formatted as system context for LLM")
        
        with st.expander("📖 View Basic Schema Context"):
            try:
                client = get_bigquery_client()
                project_id = os.getenv('GCP_PROJECT_ID')
                dataset_id = os.getenv('BQ_DATASET_ID')
                
                if client and project_id and dataset_id:
                    # Build the same format as GeminiText2SQL.py format_schema_for_prompt method
                    dataset_ref = client.dataset(dataset_id, project=project_id)
                    tables = list(client.list_tables(dataset_ref))
                    
                    # Format schema exactly like GeminiText2SQL.py does
                    schema_text = f"BigQuery Dataset: {project_id}.{dataset_id}\n\n"
                    
                    for table in tables:
                        table_ref = dataset_ref.table(table.table_id)
                        table_obj = client.get_table(table_ref)
                        
                        schema_text += f"Table: {table.table_id}"
                        if table_obj.num_rows is not None:
                            schema_text += f" ({table_obj.num_rows:,} rows)"
                        schema_text += "\nColumns:\n"
                        
                        for field in table_obj.schema:
                            schema_text += f"  - {field.name} ({field.field_type}"
                            if field.mode != 'NULLABLE':
                                schema_text += f", {field.mode}"
                            schema_text += ")"
                            if field.description:
                                schema_text += f" - {field.description}"
                            schema_text += "\n"
                        schema_text += "\n"
                    
                    st.text_area(
                        "System Context for LLM (Basic Schema)",
                        value=schema_text,
                        height=800,
                        help="This is the basic schema context that would be sent to AI systems like Gemini"
                    )
                    st.metric("Context Length", f"{len(schema_text):,} characters")
                else:
                    st.error("BigQuery client or configuration not available")
            except Exception as e:
                st.error(f"Failed to build basic schema context: {e}")
                st.text("Unable to generate basic schema context - check BigQuery connection")

def main():
    """Main application entry point for BigQuery Streamlit App"""
    
    # Sidebar navigation - Now includes Database Explorer and Knowledge Graph
    st.sidebar.title("Navigation")
    
    # Available pages
    page = st.sidebar.radio(
        "Select Page:",
        ["🗃️ Database Explorer", "🕸️ Knowledge Graph", "📋 Schema Context", "🚀 Query Interface"]  # More pages will be added in subsequent steps
    )
    
    # System status in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("System Status")
    
    checks = check_bigquery_requirements()
    
    # Only show the 4 essential system status checks
    status_items = [
        ("GCP Project", checks['gcp_project']),
        ("BigQuery Dataset", checks['bq_dataset']),
        ("Gemini", checks['gemini_kg_text2sql'] or checks['gemini_text2sql']),
        ("Knowledge Graph", checks['kg_file'])
    ]
    
    for item, status in status_items:
        icon = "✅" if status else "❌"
        st.sidebar.write(f"{icon} {item}")
    
    # Environment details
    st.sidebar.divider()
    st.sidebar.subheader("🌍 Environment")
    project_id = os.getenv('GCP_PROJECT_ID', 'Not Set')
    dataset_id = os.getenv('BQ_DATASET_ID', 'Not Set')
    location = os.getenv('VERTEX_AI_LOCATION', 'Not Set')
    
    st.sidebar.write(f"📋 **Project:** `{project_id}`")
    st.sidebar.write(f"🗄️ **Dataset:** `{dataset_id}`")
    st.sidebar.write(f"🌍 **Location:** `{location}`")
    
    # Information
    st.sidebar.divider()
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.write("""
    This is the BigQuery version of the Knowledge Graph-enhanced 
    text-to-SQL system.
    
    **Current Features:**
    • Direct BigQuery SQL execution
    • Sample query library  
    • Schema reference guide
    • Knowledge graph visualization
    
    **Coming Soon:**
    • AI-powered query generation
    • Schema context comparison
    """)
    
    # Check requirements before routing
    core_requirements = ['gcp_project', 'bq_dataset', 'bigquery_available', 'gcloud_auth']
    
    if not all(checks[req] for req in core_requirements):
        st.error("❌ BigQuery Core Requirements Not Met")
        
        if not checks['gcp_project']:
            st.error("• GCP_PROJECT_ID environment variable not found. Set in .env file.")
        if not checks['bq_dataset']:
            st.error("• BQ_DATASET_ID environment variable not found. Set in .env file.")
        if not checks['bigquery_available']:
            st.error("• BigQuery client not available. Install google-cloud-bigquery.")
        if not checks['gcloud_auth']:
            st.error("• GCloud authentication required. Run: gcloud auth application-default login")
            
        st.info("💡 **Setup Help:**\n1. Ensure .env file contains GCP_PROJECT_ID and BQ_DATASET_ID\n2. Run `gcloud auth application-default login`\n3. Install required packages: `pip install google-cloud-bigquery`")
        st.stop()
    
    # Show warnings for optional features
    if not checks['kg_visualizer']:
        st.sidebar.warning("⚠️ Knowledge Graph visualizer not available")
    if not checks['kg_file']:
        st.sidebar.warning("⚠️ BigQuery Knowledge Graph file not found")
    
    # Route to pages
    if page == "🗃️ Database Explorer":
        database_explorer_page()
    elif page == "🕸️ Knowledge Graph":
        if GBQ_VISUALIZER_AVAILABLE and checks['kg_file']:
            render_kg_visualization_page()
        elif not GBQ_VISUALIZER_AVAILABLE:
            st.error("❌ BigQuery Knowledge Graph Visualizer not available.")
            st.info("Please ensure GBQ_KGVisualizer.py is properly installed.")
        elif not checks['kg_file']:
            st.error("❌ BigQuery Knowledge Graph file not found.")
            st.info("Please ensure bqkg/bq_knowledge_graph.ttl file exists.")
            st.code("Expected file: bqkg/bq_knowledge_graph.ttl")
    elif page == "📋 Schema Context":
        schema_viewer_page()
    elif page == "🚀 Query Interface":
        query_interface_page()

if __name__ == "__main__":
    main()