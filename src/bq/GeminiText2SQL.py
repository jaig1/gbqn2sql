#!/usr/bin/env python3
"""
Enhanced BigQuery Text-to-SQL System with Gemini 2.5 Flash Lite
================================================================

This module provides a Google Gemini-powered Text-to-SQL system specifically designed 
for BigQuery databases. It uses the Google Cloud Vertex AI SDK with service account 
impersonation for authentication.

Key Features:
- Gemini 2.5 Flash Lite model for fast, cost-effective SQL generation
- BigQuery schema extraction and query execution
- Service account authentication with impersonation
- Interactive mode for testing queries
- Comprehensive error handling

Usage:
    python GeminiText2SQL.py
    
Environment Variables Required:
    - GCP_PROJECT_ID: Google Cloud Project ID
    - BQ_DATASET_ID: BigQuery Dataset ID
    - VERTEX_AI_LOCATION: Vertex AI location (default: us-central1)
    - GEMINI_MODEL_NAME: Gemini model name (default: gemini-2.5-flash-lite)
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Environment variables should be set manually.")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")
from dataclasses import dataclass

# Google Cloud and Vertex AI imports
from google.cloud import aiplatform
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class BigQueryTable:
    """Represents a BigQuery table with its schema information"""
    name: str
    schema: List[Dict[str, Any]]
    row_count: Optional[int] = None

class BigQuerySchemaExtractor:
    """Extracts schema information from BigQuery datasets"""
    
    def __init__(self, project_id: str, dataset_id: str):
        """
        Initialize BigQuery schema extractor
        
        Args:
            project_id: Google Cloud Project ID
            dataset_id: BigQuery Dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=project_id)
        
    def get_tables_info(self) -> List[BigQueryTable]:
        """
        Get information about all tables in the dataset
        
        Returns:
            List of BigQueryTable objects with schema information
        """
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            tables = list(self.client.list_tables(dataset_ref))
            
            tables_info = []
            for table in tables:
                table_ref = dataset_ref.table(table.table_id)
                table_obj = self.client.get_table(table_ref)
                
                # Extract schema information
                schema = []
                for field in table_obj.schema:
                    schema.append({
                        'name': field.name,
                        'type': field.field_type,
                        'mode': field.mode,
                        'description': field.description or ''
                    })
                
                tables_info.append(BigQueryTable(
                    name=table.table_id,
                    schema=schema,
                    row_count=table_obj.num_rows
                ))
            
            return tables_info
            
        except Exception as e:
            print(f"❌ Error extracting schema: {e}")
            return []
    
    def format_schema_for_prompt(self, tables_info: List[BigQueryTable]) -> str:
        """
        Format schema information for the Gemini prompt
        
        Args:
            tables_info: List of BigQueryTable objects
            
        Returns:
            Formatted schema string for prompt
        """
        if not tables_info:
            return "No tables found in dataset."
        
        schema_text = f"BigQuery Dataset: {self.project_id}.{self.dataset_id}\n\n"
        
        for table in tables_info:
            schema_text += f"Table: {table.name}"
            if table.row_count is not None:
                schema_text += f" ({table.row_count:,} rows)"
            schema_text += "\nColumns:\n"
            
            for field in table.schema:
                schema_text += f"  - {field['name']} ({field['type']}"
                if field['mode'] != 'NULLABLE':
                    schema_text += f", {field['mode']}"
                schema_text += ")"
                if field['description']:
                    schema_text += f" - {field['description']}"
                schema_text += "\n"
            schema_text += "\n"
        
        return schema_text

class GeminiTextToSQLBigQuery:
    """
    Gemini-powered Text-to-SQL system for BigQuery
    """
    
    def __init__(self, 
                 project_id: str, 
                 dataset_id: str,
                 vertex_ai_location: str = "us-central1",
                 model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize the Gemini Text-to-SQL system
        
        Args:
            project_id: Google Cloud Project ID
            dataset_id: BigQuery Dataset ID  
            vertex_ai_location: Vertex AI location
            model_name: Gemini model name
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.vertex_ai_location = vertex_ai_location
        self.model_name = model_name
        
        # Initialize components
        self.schema_extractor = BigQuerySchemaExtractor(project_id, dataset_id)
        self.bigquery_client = bigquery.Client(project=project_id)
        
        # Initialize Vertex AI and Gemini model
        self._initialize_gemini()
        
        # Cache schema information
        self.tables_info = None
        self.schema_text = None
        
    def _initialize_gemini(self):
        """Initialize Vertex AI and Gemini model"""
        try:
            print(f"🚀 Initializing Gemini Text-to-SQL system...")
            print(f"☁️ Gemini system - Project: {self.project_id}")
            print(f"📊 Gemini system - Dataset: {self.dataset_id}")
            print(f"🌍 Gemini system - Location: {self.vertex_ai_location}")
            print(f"🤖 Gemini system - Model: {self.model_name}")
            
            # Initialize Vertex AI with project and location
            vertexai.init(project=self.project_id, location=self.vertex_ai_location)
            
            # Initialize the Gemini model
            self.model = GenerativeModel(self.model_name)
            
            # Set generation configuration
            self.generation_config = GenerationConfig(
                temperature=0.0,  # Deterministic for SQL generation
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
            
            print(f"✅ Gemini model initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise
    
    def load_schema(self) -> bool:
        """
        Load and cache schema information
        
        Returns:
            True if schema loaded successfully, False otherwise
        """
        try:
            print("🔍 Loading BigQuery schema...")
            self.tables_info = self.schema_extractor.get_tables_info()
            
            if not self.tables_info:
                print("❌ No tables found in dataset")
                return False
            
            self.schema_text = self.schema_extractor.format_schema_for_prompt(self.tables_info)
            
            print(f"✅ Schema loaded: {len(self.tables_info)} tables found")
            for table in self.tables_info:
                print(f"  📋 {table.name}: {len(table.schema)} columns, {table.row_count:,} rows")
            
            # Print the system context that will be sent to Gemini
            print(f"\n📋 SYSTEM CONTEXT FOR GEMINI:")
            print("=" * 60)
            print(self.schema_text)
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading schema: {e}")
            return False
    
    def generate_sql(self, user_question: str, show_prompt: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Generate SQL query from natural language question using Gemini
        
        Args:
            user_question: Natural language question
            show_prompt: Whether to print the full prompt sent to Gemini
            
        Returns:
            Tuple of (generated_sql, metadata)
        """
        if not self.schema_text:
            if not self.load_schema():
                return "", {"error": "Failed to load schema"}
        
        # Create the prompt for Gemini
        prompt = self._create_sql_prompt(user_question)
        
        if show_prompt:
            print(f"\n📝 FULL PROMPT SENT TO GEMINI:")
            print("=" * 60)
            print(prompt)
            print("=" * 60)
        
        try:
            # Generate SQL using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Extract SQL from response
            generated_sql = self._extract_sql_from_response(response.text)
            
            # Create metadata
            metadata = {
                "model": self.model_name,
                "project_id": self.project_id,
                "dataset_id": self.dataset_id,
                "prompt_length": len(prompt),
                "response_length": len(response.text) if response.text else 0,
                "usage_metadata": getattr(response, 'usage_metadata', None)
            }
            
            return generated_sql, metadata
            
        except Exception as e:
            error_msg = f"Error generating SQL with Gemini: {str(e)}"
            print(f"❌ {error_msg}")
            return "", {"error": error_msg}
    
    def _create_sql_prompt(self, user_question: str) -> str:
        """
        Create a comprehensive prompt for Gemini to generate SQL
        
        Args:
            user_question: User's natural language question
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert BigQuery SQL assistant. Generate a SQL query based on the user's question and the provided schema.

IMPORTANT GUIDELINES:
1. Use Standard SQL syntax (BigQuery SQL)
2. Always use fully qualified table names: `{self.project_id}.{self.dataset_id}.table_name`
3. Use appropriate BigQuery functions and syntax
4. Ensure the query is efficient and follows BigQuery best practices

CRITICAL: DO NOT HALLUCINATE OR ASSUME:
- Do NOT invent business rules that are not explicitly provided in the schema
- Do NOT assume relationships between tables beyond what column names suggest
- Do NOT create aliases or business terminology not present in the schema
- Do NOT assume enumeration values or constraints beyond basic data types
- Do NOT add complex business logic not derivable from the schema alone
- Base your SQL ONLY on the table structure, column names, and data types provided
- If information is not available in the schema, use simple direct queries without assumptions

## Response Format:

YOU MUST respond with a valid JSON object containing exactly these fields:
- "sql": The BigQuery SQL query (string)
- "explanation": Brief explanation of what the query does, referencing only schema elements that were actually used (string)
- "confidence": Your confidence level - "high", "medium", or "low" (string)
- "reasoning": Step-by-step reasoning for the query construction, citing only the schema information that guided your decisions (string)

## Schema-Only Citation Requirements:
When writing your explanation and reasoning, you MUST:
- **Reference only schema elements** actually used (table names, column names, data types)
- **Cite column relationships** only when evident from column names (e.g., "customer_id foreign key")
- **Mention data type considerations** when relevant (e.g., "DATE columns for date comparison")
- **State assumptions clearly** if making logical inferences from column names
- **Avoid business logic** that cannot be derived directly from schema structure
- If making assumptions, state: "Assuming [assumption] based on column name/type"

## Example Response:
{{
    "sql": "SELECT customer_id, name, email FROM `{self.project_id}.{self.dataset_id}.insurance_customers` LIMIT 100",
    "explanation": "This query retrieves customer identification, name, and email columns from the insurance_customers table as requested. Used schema columns: customer_id (INTEGER), name (STRING), email (STRING).",
    "confidence": "high",
    "reasoning": "1. Identified insurance_customers table contains the requested customer information. 2. Selected customer_id, name, and email columns based on schema definition. 3. Added LIMIT 100 for BigQuery best practices. 4. No joins required as all data is in single table."
}}

## Important Notes:
- ALWAYS return valid JSON - no extra text before or after the JSON object
- Reference only schema elements you can see in the DATABASE SCHEMA section below
- If you cannot generate a query, set "sql" to null and explain why in "reasoning"
- Always ensure the query is syntactically correct BigQuery SQL
- Use fully qualified table names: `{self.project_id}.{self.dataset_id}.table_name`

The SQL you generate must be fully explainable using only the DATABASE SCHEMA provided below.

DATABASE SCHEMA:
{self.schema_text}

USER QUESTION: {user_question}

Generate a BigQuery SQL query that answers this question:"""
        
        return prompt
    
    def _extract_sql_from_response(self, response_text: str) -> Tuple[str, str, str, str]:
        """
        Extract SQL query and metadata from Gemini JSON response
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Tuple of (sql_query, explanation, confidence, reasoning)
        """
        if not response_text:
            return "", "No response received", "low", "Empty response from Gemini"
        
        # Strip markdown code block wrapper if present
        if response_text.strip().startswith('```json'):
            # Remove ```json from start and ``` from end
            lines = response_text.strip().split('\n')
            if lines[0].strip() == '```json' and lines[-1].strip() == '```':
                response_text = '\n'.join(lines[1:-1])
            elif lines[0].strip().startswith('```json'):
                # Handle case where ```json is on same line as content
                first_line = lines[0].replace('```json', '').strip()
                if first_line:
                    lines[0] = first_line
                else:
                    lines = lines[1:]
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_text = '\n'.join(lines)
        
        try:
            parsed_response = json.loads(response_text)
            sql_query = parsed_response.get("sql", "")
            explanation = parsed_response.get("explanation", "")
            confidence = parsed_response.get("confidence", "medium")
            reasoning = parsed_response.get("reasoning", "")
            
            return sql_query, explanation, confidence, reasoning
            
        except json.JSONDecodeError as e:
            # Fallback to old behavior for non-JSON responses
            return self._extract_raw_sql(response_text), f"Non-JSON response received: {str(e)}", "low", "Failed to parse JSON response"
    
    def _extract_raw_sql(self, response_text: str) -> str:
        """
        Fallback method to extract raw SQL from non-JSON responses
        
        Args:
            response_text: Raw response text
            
        Returns:
            Cleaned SQL query
        """
        if not response_text:
            return ""
        
        # Remove markdown code blocks if present
        if "```sql" in response_text:
            lines = response_text.split('\n')
            sql_lines = []
            in_sql_block = False
            
            for line in lines:
                if line.strip().startswith("```sql"):
                    in_sql_block = True
                    continue
                elif line.strip() == "```" and in_sql_block:
                    break
                elif in_sql_block:
                    sql_lines.append(line)
            
            return '\n'.join(sql_lines).strip()
        
        # Remove any leading/trailing whitespace and common prefixes
        sql = response_text.strip()
        
        # Remove common explanatory text
        if sql.lower().startswith("here is") or sql.lower().startswith("here's"):
            lines = sql.split('\n')
            if len(lines) > 1:
                sql = '\n'.join(lines[1:]).strip()
        
        return sql

    def generate_sql(self, user_question: str, show_prompt: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Generate SQL query from natural language question using Gemini
        
        Args:
            user_question: Natural language question
            show_prompt: Whether to print the full prompt sent to Gemini
            
        Returns:
            Tuple of (generated_sql, metadata)
        """
        if not self.schema_text:
            if not self.load_schema():
                return "", {"error": "Failed to load schema"}

        # Create the prompt for Gemini
        prompt = self._create_sql_prompt(user_question)

        if show_prompt:
            print(f"\n📝 FULL PROMPT SENT TO GEMINI:")
            print("=" * 60)
            print(prompt)
            print("=" * 60)

        try:
            # Generate SQL using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            # Extract SQL and metadata from response
            generated_sql, explanation, confidence, reasoning = self._extract_sql_from_response(response.text)

            # Create metadata including explanation details
            metadata = {
                "model": self.model_name,
                "project_id": self.project_id,
                "dataset_id": self.dataset_id,
                "prompt_length": len(prompt),
                "response_length": len(response.text) if response.text else 0,
                "explanation": explanation,
                "confidence": confidence,
                "reasoning": reasoning,
                "usage_metadata": getattr(response, 'usage_metadata', None)
            }

            return generated_sql, metadata

        except Exception as e:
            error_msg = f"Error generating SQL with Gemini: {str(e)}"
            print(f"❌ {error_msg}")
            return "", {"error": error_msg}
    
    def execute_sql(self, sql_query: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Execute SQL query against BigQuery
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Tuple of (results, metadata)
        """
        try:
            print(f"🔍 Executing query...")
            print(f"SQL: {sql_query}")
            
            # Configure query job
            job_config = bigquery.QueryJobConfig()
            job_config.dry_run = False
            job_config.use_query_cache = True
            
            # Execute query
            query_job = self.bigquery_client.query(sql_query, job_config=job_config)
            results = query_job.result()
            
            # Convert results to list of dictionaries
            rows = []
            for row in results:
                rows.append(dict(row))
            
            # Create metadata
            metadata = {
                "total_rows": len(rows),
                "job_id": query_job.job_id,
                "bytes_processed": query_job.total_bytes_processed,
                "bytes_billed": query_job.total_bytes_billed,
                "slot_ms": query_job.slot_millis,
                "creation_time": query_job.created.isoformat() if query_job.created else None,
                "query": sql_query
            }
            
            print(f"✅ Query executed successfully!")
            print(f"📊 Results: {len(rows)} rows")
            if query_job.total_bytes_processed:
                print(f"💾 Bytes processed: {query_job.total_bytes_processed:,}")
            
            return rows, metadata
            
        except Exception as e:
            error_msg = f"Error executing SQL: {str(e)}"
            print(f"❌ {error_msg}")
            return [], {"error": error_msg, "query": sql_query}
    
    def process_question(self, user_question: str, execute: bool = True) -> Dict[str, Any]:
        """
        Process a natural language question end-to-end
        
        Args:
            user_question: Natural language question
            execute: Whether to execute the generated SQL
            
        Returns:
            Dictionary containing results and metadata
        """
        print(f"\n🔤 Question: {user_question}")
        
        # Generate SQL
        sql_query, gen_metadata = self.generate_sql(user_question)
        
        if not sql_query:
            return {
                "question": user_question,
                "sql": "",
                "error": gen_metadata.get("error", "Failed to generate SQL"),
                "generation_metadata": gen_metadata
            }
        
        print(f"🔧 Generated SQL:\n{sql_query}")
        
        result = {
            "question": user_question,
            "sql": sql_query,
            "generation_metadata": gen_metadata
        }
        
        if execute:
            # Execute SQL
            rows, exec_metadata = self.execute_sql(sql_query)
            result.update({
                "results": rows,
                "execution_metadata": exec_metadata
            })
            
            if "error" not in exec_metadata and rows:
                print(f"📋 Sample results:")
                for i, row in enumerate(rows[:3]):
                    print(f"  Row {i+1}: {row}")
                if len(rows) > 3:
                    print(f"  ... and {len(rows)-3} more rows")
        
        return result

def run_pretest_queries(text_to_sql: GeminiTextToSQLBigQuery) -> bool:
    """
    Run 5 hardcoded test queries to validate the system is working
    
    Args:
        text_to_sql: Initialized GeminiTextToSQLBigQuery instance
        
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "="*80)
    print("🧪 RUNNING PRETEST QUERIES")
    print("="*80)
    
    test_queries = [
        "List all customers",
        "How many total policies are there?",
        "Show me the top 5 agents by name",
        "What are the different policy types available?",
        "Find customers who are older than 35 years"
    ]
    
    passed_tests = 0
    total_tests = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}/{total_tests}: {query}")
        print("-" * 60)
        
        try:
            # Generate SQL (don't execute to save time and resources)
            sql_query, gen_metadata = text_to_sql.generate_sql(query)
            
            if sql_query and "error" not in gen_metadata:
                print(f"✅ Test {i} PASSED")
                print(f"🔧 Generated SQL:\n{sql_query}")
                passed_tests += 1
            else:
                print(f"❌ Test {i} FAILED")
                print(f"Error: {gen_metadata.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test {i} FAILED with exception: {e}")
    
    print(f"\n📊 PRETEST RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All pretest queries passed! System is working correctly.")
        return True
    else:
        print("⚠️ Some pretest queries failed. Please check the system configuration.")
        return False

def interactive_mode():
    """Run the system in interactive mode"""
    print("\n" + "="*80)
    print("🔧 GEMINI BIGQUERY TEXT-TO-SQL (WITH VERTEX AI)")
    print("="*80)
    
    # Load configuration from environment
    project_id = os.getenv('GCP_PROJECT_ID')
    dataset_id = os.getenv('BQ_DATASET_ID')
    vertex_ai_location = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
    model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash-lite')
    
    if not project_id or not dataset_id:
        print("❌ Error: Please set GCP_PROJECT_ID and BQ_DATASET_ID environment variables")
        return
    
    try:
        # Initialize the system
        text_to_sql = GeminiTextToSQLBigQuery(
            project_id=project_id,
            dataset_id=dataset_id,
            vertex_ai_location=vertex_ai_location,
            model_name=model_name
        )
        
        # Load schema
        if not text_to_sql.load_schema():
            print("❌ Failed to load schema. Exiting.")
            return
        
        # Run pretest queries
        print(f"\n🧪 Running pretest validation...")
        pretest_passed = run_pretest_queries(text_to_sql)
        
        if not pretest_passed:
            print("❌ Pretest failed. Exiting.")
            return
        
        print(f"\n✅ System ready! Using Gemini model: {model_name}")
        print("💡 Ask questions about your data in natural language.")
        print("💡 Type 'exit' to quit, 'schema' to view schema, 'help' for commands.\n")
        
        while True:
            try:
                user_input = input("🔤 Your question: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() in ['schema', 'tables']:
                    print(f"\n📋 Database Schema:\n{text_to_sql.schema_text}")
                    continue
                elif user_input.lower() == 'test':
                    print(f"\n🧪 Running pretest queries again...")
                    run_pretest_queries(text_to_sql)
                    continue
                elif user_input.lower().startswith('prompt '):
                    question = user_input[7:].strip()
                    if question:
                        print(f"\n🔍 Generating SQL with full prompt display...")
                        sql_query, metadata = text_to_sql.generate_sql(question, show_prompt=True)
                        if sql_query:
                            print(f"\n🔧 Generated SQL:\n{sql_query}")
                        else:
                            print(f"❌ Failed to generate SQL: {metadata.get('error', 'Unknown error')}")
                    else:
                        print("❌ Please provide a question after 'prompt'. Example: prompt List all customers")
                    continue
                elif user_input.lower() == 'help':
                    print("""
🔧 Available Commands:
  - Ask any question about your data in natural language
  - 'schema' or 'tables' - View database schema
  - 'test' - Run the 5 pretest queries again
  - 'prompt <question>' - Show the full prompt sent to Gemini for a question
  - 'exit', 'quit', or 'q' - Exit the program
  - 'help' - Show this help message
""")
                    continue
                elif not user_input:
                    continue
                
                # Process the question
                result = text_to_sql.process_question(user_input)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

def main():
    """Main entry point"""
    import sys
    
    # Check for test mode
    test_mode = len(sys.argv) > 1 and sys.argv[1] in ['--test', '-t', 'test']
    
    try:
        if test_mode:
            print("\n" + "="*80)
            print("🧪 GEMINI TEXT2SQL - TEST MODE")
            print("="*80)
            
            # Load configuration from environment
            project_id = os.getenv('GCP_PROJECT_ID')
            dataset_id = os.getenv('BQ_DATASET_ID')
            vertex_ai_location = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
            model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash-lite')
            
            if not project_id or not dataset_id:
                print("❌ Error: Please set GCP_PROJECT_ID and BQ_DATASET_ID environment variables")
                return
            
            # Initialize the system
            text_to_sql = GeminiTextToSQLBigQuery(
                project_id=project_id,
                dataset_id=dataset_id,
                vertex_ai_location=vertex_ai_location,
                model_name=model_name
            )
            
            # Load schema and run tests
            if text_to_sql.load_schema():
                run_pretest_queries(text_to_sql)
            else:
                print("❌ Failed to load schema.")
        else:
            interactive_mode()
            
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()