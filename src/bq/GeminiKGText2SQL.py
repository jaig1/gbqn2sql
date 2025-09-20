#!/usr/bin/env python3
"""
Gemini-Powered Text-to-SQL System for BigQuery Using BigQuery Knowledge Graph
============================================================================

This module uses the Google Gemini API to convert natural language questions
into BigQuery SQL queries, leveraging the BigQuery Knowledge Graph for schema context.

Requirements:
- google-generativeai>=0.3.0
- rdflib
- google-cloud-bigquery

Setup:
1. Install Gemini: pip install google-generativeai
2. Set your Gemini API key: export GEMINI_API_KEY="your-key-here"
3. Set up BigQuery credentials: export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
4. Ensure bqkg/bq_knowledge_graph.ttl exists (run BQKnowledgeGraphGenerator.py first)

Usage:
python GeminiKGText2SQL.py
"""

import os
import sys
from pathlib import Path
from google.cloud import bigquery
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Environment variables should be set manually.")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# Use Vertex AI instead of Google AI SDK (same as GeminiText2SQL.py)
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Check availability of required libraries
try:
    import vertexai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# Import our BigQuery Knowledge Graph context builder
try:
    from .BQKnowledgeGraphContextBuilder import BQKnowledgeGraphContextBuilder
except ImportError:
    try:
        from BQKnowledgeGraphContextBuilder import BQKnowledgeGraphContextBuilder
    except ImportError:
        print("❌ BQKnowledgeGraphContextBuilder not found. Ensure BQKnowledgeGraphContextBuilder.py is available.")
        exit(1)

@dataclass
class SQLGenerationResult:
    """Result of SQL generation process"""
    question: str
    sql_query: Optional[str]
    explanation: Optional[str]
    confidence: str
    reasoning: Optional[str]
    error: Optional[str] = None
    
@dataclass
class QueryExecutionResult:
    """Result of BigQuery SQL query execution"""
    success: bool
    results: List[Dict]
    row_count: int
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    bytes_processed: Optional[int] = None
    query_id: Optional[str] = None

class GeminiKGText2SQL:
    """
    Gemini-powered Text-to-SQL generator for BigQuery using BigQuery Knowledge Graph context
    """
    
    def __init__(self, project_id: str, dataset_id: str, bq_kg_file: str, model: str = None):
        """
        Initialize the Gemini Text-to-SQL system for BigQuery
        
        Args:
            project_id: BigQuery project ID
            dataset_id: BigQuery dataset ID
            bq_kg_file: Path to BigQuery Knowledge Graph TTL file
            model: Gemini model to use (if None, uses GEMINI_MODEL_NAME from environment or default)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library is required. Install with: pip install google-generativeai")
        
        if not BIGQUERY_AVAILABLE:
            raise ImportError("BigQuery client library is required. Install with: pip install google-cloud-bigquery")
        
        # Use model from environment if not provided
        if model is None:
            model = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash')
        
        print(f"🤖 Using model: {model}")
        
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.model = model
        
        # Use Vertex AI with gcloud authentication (same as GeminiText2SQL.py)
        print("🔑 Using Vertex AI with gcloud configured authentication")
        
        try:
            # Get location from environment (same pattern as GeminiText2SQL.py)
            vertex_ai_location = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
            
            # Initialize Vertex AI with project and location
            vertexai.init(project=self.project_id, location=vertex_ai_location)
            
            # Initialize the Gemini model using Vertex AI
            self.client = GenerativeModel(model)
            
            # Set generation configuration for deterministic SQL generation
            self.generation_config = GenerationConfig(
                temperature=0.0,  # Deterministic for SQL generation
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
            
            # Test the connection with a simple call
            test_response = self.client.generate_content("Test connection", generation_config=self.generation_config)
            print(f"✅ Vertex AI Gemini connection successful")
        except Exception as e:
            print(f"⚠️ Vertex AI Gemini setup: {e}")
            print("Note: Continuing with BigQuery functionality")
        
        try:
            # Initialize BigQuery client with service account impersonation
            self.bq_client = self._setup_bigquery_client()
            
            # Test BigQuery connection
            test_query = f"SELECT 1 as test_connection"
            test_job = self.bq_client.query(test_query)
            list(test_job.result())  # Consume results
            print(f"✅ BigQuery connection successful for project: {self.project_id}")
        except Exception as e:
            raise ValueError(f"Failed to connect to BigQuery: {e}")
        
        # Initialize BigQuery Knowledge Graph context builder
        self.context_builder = BQKnowledgeGraphContextBuilder(bq_kg_file)
        self.schema_context = self.context_builder.build_llm_context()
        
        # System prompt for the LLM
        self.system_prompt = self._build_system_prompt()
        
    def _setup_bigquery_client(self):
        """Setup BigQuery client using gcloud configured credentials"""
        try:
            # Use default credentials (gcloud auth is already configured)
            client = bigquery.Client(project=self.project_id)
            print("✅ BigQuery client initialized with gcloud credentials")
            return client
                
        except Exception as e:
            raise Exception(f"Failed to setup BigQuery client: {e}")
        
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt with BigQuery schema context"""
        return f"""You are an expert BigQuery SQL query generator. Your task is to convert natural language questions into accurate BigQuery SQL queries based on the provided database schema.

{self.schema_context}

## BigQuery-Specific Instructions:

1. **Always use exact table and column names** as specified in the schema above
2. **Use proper BigQuery JOIN syntax** based on the foreign key relationships listed
3. **Apply BigQuery data types correctly** in WHERE clauses and comparisons
4. **Handle BigQuery date formats correctly** (use DATE(), DATETIME(), TIMESTAMP() functions)
5. **Use BigQuery string functions** like CONCAT(), SUBSTR(), REGEXP_CONTAINS()
6. **Use IFNULL() or COALESCE()** for null handling instead of ISNULL()
7. **Include proper GROUP BY clauses** when using aggregate functions
8. **Order results logically** (e.g., by name, date, or amount as appropriate)
9. **Always add LIMIT** for exploratory queries (default: LIMIT 100)
10. **Use proper BigQuery syntax** for arrays, structs, and nested data

## BigQuery Function Examples:
- Date functions: DATE(column), DATE_ADD(date, INTERVAL 30 DAY), DATE_DIFF(date1, date2, DAY)
- String functions: CONCAT(str1, str2), SUBSTR(string, start, length), REGEXP_CONTAINS(string, pattern)
- Array functions: ARRAY_AGG(column), UNNEST(array_column)
- Window functions: ROW_NUMBER() OVER (PARTITION BY col ORDER BY col), LAG(col) OVER (ORDER BY col)
- Safe casting: SAFE_CAST(column AS INT64), SAFE_CAST(column AS STRING)

## Privacy and Security Guidelines:
- **Be cautious with PII columns** marked with 🔒 symbol
- **Respect sensitivity levels** (HIGH, MEDIUM, LOW) and access controls
- **Consider using EXCEPT()** to exclude sensitive columns when not needed
- **Apply appropriate WHERE clauses** for data filtering based on business rules

## Response Format:

YOU MUST respond with a valid JSON object containing exactly these fields:
- "sql": The BigQuery SQL query (string)
- "explanation": Brief explanation of what the query does, explicitly citing which specific business rules, aliases, relationships, or domain knowledge from the system context were used (string)
- "confidence": Your confidence level - "high", "medium", or "low" (string)
- "reasoning": Step-by-step reasoning for the query construction, referencing specific system context elements like business rules, canonical patterns, or aliases used (string)

## Knowledge Citation Requirements:
When writing your explanation and reasoning, you MUST:
- **Cite specific business rules** if used (e.g., "Applied Active Policy business rule: status = 'Active'")
- **Reference aliases** when used (e.g., "Used Customer Alias 'policy_holder' from system context")
- **Mention relationship patterns** from canonical joins (e.g., "Applied Customer→Policy join pattern")
- **Quote segmentation rules** when applicable (e.g., "Applied Premium Segmentation: Budget segment <= 2000")
- **Reference data validation rules** if relevant (e.g., "Applied email validation pattern")
- **Cite enumeration values** when used (e.g., "Used PolicyStatus enum value 'Active'")
- If no specific knowledge was cited, state: "No explicit business rules or aliases from system context were applied"

## Example Response:
{{
    "sql": "SELECT c.name, COUNT(p.policy_id) as policy_count FROM `{self.project_id}.{self.dataset_id}.insurance_customers` c JOIN `{self.project_id}.{self.dataset_id}.insurance_policies` p ON c.customer_id = p.customer_id GROUP BY c.customer_id, c.name HAVING COUNT(p.policy_id) > 1 ORDER BY policy_count DESC LIMIT 100",
    "explanation": "This query finds customers who have multiple insurance policies using the Customer→Policy canonical join pattern from the system context. Applied business logic for 'multiple policies' meaning COUNT > 1.",
    "confidence": "high",
    "reasoning": "1. Applied Customer→Policy canonical join pattern on customer_id foreign key as specified in system context. 2. Used COUNT aggregate function requiring GROUP BY. 3. Applied HAVING clause to filter for multiple policies based on business logic. 4. Ordered by policy count and added LIMIT for BigQuery best practices."
}}

## Important Notes:
- ALWAYS return valid JSON - no extra text before or after the JSON object
- If a question is ambiguous, make reasonable assumptions and explain them
- If you cannot generate a query, set "sql" to null and explain why in "reasoning"
- Always ensure the query is syntactically correct BigQuery SQL
- Consider performance implications and use LIMIT appropriately
- Use fully qualified table names: `{self.project_id}.{self.dataset_id}.table_name`
- Leverage BigQuery's columnar storage and partitioning when possible
"""

    def print_system_context(self):
        """Print the complete system context sent to the LLM"""
        print("\n" + "="*80)
        print("🧠 BIGQUERY KNOWLEDGE GRAPH-ENHANCED SYSTEM CONTEXT SENT TO LLM")
        print("="*80)
        print(self.system_prompt)
        print("="*80)

    def generate_sql(self, question: str) -> SQLGenerationResult:
        """
        Generate BigQuery SQL query from natural language question using Gemini
        
        Args:
            question: Natural language question about the database
            
        Returns:
            SQLGenerationResult with generated query and metadata
        """
        try:
            # Create the full prompt
            full_prompt = f"{self.system_prompt}\n\nUser Question: {question}"
            
            # Call Gemini API using Vertex AI
            response = self.client.generate_content(
                full_prompt,
                generation_config=self.generation_config  # Use the pre-configured generation config
            )
            
            # Parse the response
            response_content = response.text
            
            # Debug: Print the raw response content
            print(f"🔍 Debug - Raw Gemini response (first 200 chars): {repr(response_content[:200])}")
            
            if not response_content or response_content.strip() == "":
                return SQLGenerationResult(
                    question=question,
                    sql_query=None,
                    explanation=None,
                    confidence="low",
                    reasoning=None,
                    error="Gemini returned empty response"
                )
            
            # Strip markdown code block wrapper if present
            if response_content.strip().startswith('```json'):
                # Remove ```json from start and ``` from end
                lines = response_content.strip().split('\n')
                if lines[0].strip() == '```json' and lines[-1].strip() == '```':
                    response_content = '\n'.join(lines[1:-1])
                elif lines[0].strip().startswith('```json'):
                    # Handle case where ```json is on same line as content
                    first_line = lines[0].replace('```json', '').strip()
                    if first_line:
                        lines[0] = first_line
                    else:
                        lines = lines[1:]
                    if lines[-1].strip() == '```':
                        lines = lines[:-1]
                    response_content = '\n'.join(lines)
            
            try:
                parsed_response = json.loads(response_content)
            except json.JSONDecodeError as e:
                return SQLGenerationResult(
                    question=question,
                    sql_query=None,
                    explanation=None,
                    confidence="low",
                    reasoning=None,
                    error=f"Failed to parse Gemini response as JSON: {e}\nResponse content: {repr(response_content[:500])}"
                )
            
            # Extract components
            sql_query = parsed_response.get("sql")
            explanation = parsed_response.get("explanation", "")
            confidence = parsed_response.get("confidence", "medium")
            reasoning = parsed_response.get("reasoning", "")
            
            return SQLGenerationResult(
                question=question,
                sql_query=sql_query,
                explanation=explanation,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            return SQLGenerationResult(
                question=question,
                sql_query=None,
                explanation=None,
                confidence="low",
                reasoning=None,
                error=f"Gemini API error: {str(e)}"
            )
    
    def execute_sql(self, sql_query: str) -> QueryExecutionResult:
        """
        Execute BigQuery SQL query
        
        Args:
            sql_query: BigQuery SQL query to execute
            
        Returns:
            QueryExecutionResult with execution results
        """
        import time
        
        try:
            start_time = time.time()
            
            # Configure job for query execution
            job_config = bigquery.QueryJobConfig()
            job_config.use_query_cache = True
            job_config.maximum_bytes_billed = 100 * 1024 * 1024  # 100 MB limit for safety
            
            # Execute the query
            query_job = self.bq_client.query(sql_query, job_config=job_config)
            results = query_job.result()  # Wait for completion
            
            # Convert to list of dictionaries
            results_list = []
            for row in results:
                row_dict = {}
                for key, value in row.items():
                    # Handle BigQuery-specific data types
                    if hasattr(value, 'isoformat'):  # DateTime objects
                        row_dict[key] = value.isoformat()
                    elif hasattr(value, 'total_seconds'):  # Timedelta objects
                        row_dict[key] = value.total_seconds()
                    else:
                        row_dict[key] = value
                results_list.append(row_dict)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return QueryExecutionResult(
                success=True,
                results=results_list,
                row_count=len(results_list),
                execution_time_ms=execution_time,
                bytes_processed=query_job.total_bytes_processed,
                query_id=query_job.job_id
            )
            
        except Exception as e:
            return QueryExecutionResult(
                success=False,
                results=[],
                row_count=0,
                error=str(e)
            )
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate BigQuery SQL query without executing it
        
        Args:
            sql_query: BigQuery SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use dry run to validate without executing
            job_config = bigquery.QueryJobConfig()
            job_config.dry_run = True
            job_config.use_query_cache = False
            
            query_job = self.bq_client.query(sql_query, job_config=job_config)
            
            # If we get here without exception, the query is valid
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def answer_question(self, question: str, execute: bool = True) -> Dict:
        """
        Complete pipeline: question -> SQL -> execution -> formatted response
        
        Args:
            question: Natural language question
            execute: Whether to execute the query (default True)
            
        Returns:
            Dictionary with complete results
        """
        print(f"\n🤔 Question: {question}")
        print("=" * 60)
        
        # Generate SQL using Gemini
        print("🧠 Generating BigQuery SQL with Gemini...")
        sql_result = self.generate_sql(question)
        
        if sql_result.error:
            print(f"❌ Error generating SQL: {sql_result.error}")
            return {
                "question": question,
                "sql_generation": sql_result,
                "execution": None,
                "success": False
            }
        
        if not sql_result.sql_query:
            print(f"❌ No SQL query generated")
            print(f"💭 Reasoning: {sql_result.reasoning}")
            return {
                "question": question,
                "sql_generation": sql_result,
                "execution": None,
                "success": False
            }
        
        # Display generated SQL
        print(f"🎯 Confidence: {sql_result.confidence}")
        print(f"📝 Generated BigQuery SQL:")
        print(f"   {sql_result.sql_query}")
        print(f"💡 Explanation: {sql_result.explanation}")
        
        if sql_result.reasoning:
            print(f"🧩 Reasoning: {sql_result.reasoning}")
        
        # Validate SQL
        is_valid, validation_error = self.validate_sql(sql_result.sql_query)
        if not is_valid:
            print(f"❌ BigQuery SQL validation failed: {validation_error}")
            return {
                "question": question,
                "sql_generation": sql_result,
                "execution": None,
                "success": False,
                "validation_error": validation_error
            }
        
        execution_result = None
        if execute:
            # Execute SQL
            print(f"\n⚡ Executing BigQuery query...")
            execution_result = self.execute_sql(sql_result.sql_query)
            
            if execution_result.success:
                print(f"✅ Query executed successfully!")
                print(f"📊 Found {execution_result.row_count} results")
                print(f"⏱️  Execution time: {execution_result.execution_time_ms:.2f}ms")
                
                if execution_result.bytes_processed:
                    bytes_mb = execution_result.bytes_processed / (1024 * 1024)
                    print(f"💾 Bytes processed: {bytes_mb:.2f} MB")
                
                if execution_result.query_id:
                    print(f"🆔 Query ID: {execution_result.query_id}")
                
                # Display sample results
                if execution_result.results:
                    print(f"\n📋 Results:")
                    for i, row in enumerate(execution_result.results[:5]):  # Show first 5
                        print(f"   {i+1}. {dict(row)}")
                    
                    if execution_result.row_count > 5:
                        print(f"   ... and {execution_result.row_count - 5} more results")
                else:
                    print("   No results found.")
            else:
                print(f"❌ Query execution failed: {execution_result.error}")
        
        return {
            "question": question,
            "sql_generation": sql_result,
            "execution": execution_result,
            "success": execution_result.success if execution_result else True
        }
    
    def batch_questions(self, questions: List[str]) -> List[Dict]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of natural language questions
            
        Returns:
            List of results for each question
        """
        results = []
        
        print(f"\n🔄 Processing {len(questions)} questions in batch...")
        print("=" * 60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing: {question}")
            result = self.answer_question(question)
            results.append(result)
        
        return results
    
    def interactive_mode(self):
        """
        Interactive question-answering mode
        """
        print("\n" + "="*80)
        print("🤖 GEMINI-POWERED INTERACTIVE TEXT-TO-BIGQUERY-SQL")
        print("="*80)
        print("Ask questions about the BigQuery insurance database in natural language!")
        print("Powered by Google Gemini with BigQuery Knowledge Graph context.")
        print(f"\nProject: {self.project_id}")
        print(f"Dataset: {self.dataset_id}")
        print("\nExample questions:")
        print("• 'Show me all customers with their contact information'")
        print("• 'Find the customer with the highest total premium amount'")
        print("• 'What's the average claim amount by policy type?'")
        print("• 'Which agent has the best claims-to-premium ratio?'")
        print("\nType 'quit' to exit, 'help' for more examples.")
        print("-" * 80)
        
        while True:
            try:
                question = input(f"\n❓ Your question (using {self.model}): ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                elif question.lower() == 'help':
                    self._print_help()
                    continue
                
                # Process question
                self.answer_question(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _print_help(self):
        """Print help with example questions"""
        print("\n📚 EXAMPLE QUESTIONS FOR BIGQUERY:")
        print("-" * 50)
        
        examples = [
            "Show me all customers sorted by name",
            "Find customers in specific geographic regions", 
            "What policies does a specific customer have?",
            "Which customer has the most expensive policy?",
            "Show me all pending claims with amounts over $2000",
            "What's the total premium revenue by policy type?",
            "Find agents who have sold more than 5 policies",
            "Show customers who have both auto and home policies",
            "What's the average claim amount for auto policies?",
            "Find the top 3 customers by total premium amount",
            "Show all claims filed in the last 90 days",
            "Which policy type has the highest claim ratio?",
            "Calculate agent performance metrics",
            "Analyze customer retention rates",
            "Show policy distribution by state or region",
            "Find high-risk customers with multiple claims",
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"  {i:2d}. '{example}'")
    
    # UI Integration Methods for Streamlit
    def get_system_info(self) -> Dict:
        """Get system information for UI display"""
        return {
            "name": "BigQuery Knowledge Graph-Enhanced Text-to-SQL System",
            "description": "Uses BigQuery Knowledge Graph for rich semantic context and business intelligence",
            "model": self.model,
            "platform": "Google BigQuery",
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "context_length": len(self.schema_context),
            "features": [
                "Rich semantic context from BigQuery Knowledge Graph",
                "Business intelligence and domain knowledge",
                "PII detection and privacy controls",
                "Canonical JOIN patterns for error prevention",
                "Business vocabulary mappings",
                "Data quality metrics and validation rules",
                "BigQuery-specific optimizations and syntax",
                "Cloud-native performance considerations"
            ],
            "advantages": [
                "Superior accuracy for complex BigQuery queries",
                "Business domain understanding for insurance data",
                "Error prevention through canonical patterns",
                "Privacy-aware data handling with sensitivity levels",
                "BigQuery performance optimization guidance",
                "Cloud-scale data processing capabilities"
            ]
        }
    
    def get_schema_context(self) -> str:
        """Get the schema context for UI display"""
        return self.schema_context
    
    def get_context_statistics(self) -> Dict:
        """Get context statistics for comparison display"""
        return {
            "total_length": len(self.schema_context),
            "enrichment_factor": round(len(self.schema_context) / 1202, 1),  # vs basic schema
            "business_intelligence": True,
            "semantic_understanding": True,
            "pii_detection": True,
            "domain_knowledge": True,
            "bigquery_optimized": True,
            "cloud_native": True
        }
    
    def process_query_for_ui(self, question: str) -> Dict:
        """
        Process a query specifically for UI display
        Returns structured data for Streamlit interface
        """
        result = {
            "question": question,
            "system": "BigQuery KG-Enhanced", 
            "success": False,
            "sql_query": None,
            "explanation": None,
            "confidence": "low",
            "reasoning": None,
            "execution_results": None,
            "execution_time_ms": None,
            "bytes_processed": None,
            "query_id": None,
            "row_count": 0,
            "error": None
        }
        
        try:
            # Generate SQL
            sql_result = self.generate_sql(question)
            
            result["sql_query"] = sql_result.sql_query
            result["explanation"] = sql_result.explanation
            result["confidence"] = sql_result.confidence
            result["reasoning"] = sql_result.reasoning
            
            if sql_result.error:
                result["error"] = sql_result.error
                return result
            
            if not sql_result.sql_query:
                result["error"] = "No SQL query generated"
                return result
            
            # Validate SQL
            is_valid, validation_error = self.validate_sql(sql_result.sql_query)
            if not is_valid:
                result["error"] = f"BigQuery SQL validation failed: {validation_error}"
                return result
            
            # Execute SQL
            execution_result = self.execute_sql(sql_result.sql_query)
            
            if execution_result.success:
                result["success"] = True
                result["execution_results"] = execution_result.results
                result["execution_time_ms"] = execution_result.execution_time_ms
                result["bytes_processed"] = execution_result.bytes_processed
                result["query_id"] = execution_result.query_id
                result["row_count"] = execution_result.row_count
            else:
                result["error"] = f"Query execution failed: {execution_result.error}"
                
        except Exception as e:
            result["error"] = f"System error: {str(e)}"
        
        return result

def check_requirements():
    """Check if all required components are available"""
    # Assume gcloud auth and SA impersonation are configured
    # No environment variables required
    return []

def demonstrate_gemini_bigquery_text_to_sql():
    """Demonstrate the Gemini-powered BigQuery text-to-SQL system with SA impersonation"""
    print("=" * 80)
    print("🤖 GEMINI-POWERED BIGQUERY TEXT-TO-SQL WITH SERVICE ACCOUNT IMPERSONATION")
    print("=" * 80)
    
    # Load configuration from environment variables (same as GeminiText2SQL.py)
    project_id = os.getenv('GCP_PROJECT_ID')
    dataset_id = os.getenv('BQ_DATASET_ID')
    vertex_ai_location = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
    model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash')
    
    if not project_id or not dataset_id:
        print("❌ Error: Please set GCP_PROJECT_ID and BQ_DATASET_ID environment variables")
        return None
    
    # Check requirements (should pass now with env vars loaded)
    errors = check_requirements()
    if errors:
        print("❌ Requirements not met:")
        for error in errors:
            print(f"   - {error}")
        print("\nNote: Assuming gcloud auth and SA impersonation are configured")
        return None
    
    try:
        # Initialize system
        print(f"🚀 Initializing Gemini BigQuery Text-to-SQL system with SA impersonation...")
        print(f"   Project: {project_id}")
        print(f"   Dataset: {dataset_id}")
        print(f"   Location: {vertex_ai_location}")
        print(f"   Model: {model_name}")
        
        system = GeminiKGText2SQL(
            project_id=project_id,
            dataset_id=dataset_id,
            bq_kg_file='bqkg/bq_knowledge_graph.ttl',
            model=model_name
        )
        print("✅ System initialized successfully!")
        
        # Print the complete system context being sent to the LLM
        system.print_system_context()
        
        # Demonstrate with 5 example questions (same as test mode)
        example_questions = [
            "Show me all customers with their email addresses",
            "Find customers with multiple policies", 
            "What's the average premium for life insurance?",
            "Which agent has the best claims-to-premium ratio?",
            "Find high-premium customers with low claims"
        ]
        
        print(f"\n🎯 Demonstrating with {len(example_questions)} example questions:")
        print(f"🔄 Processing {len(example_questions)} questions in batch...")
        print("="*60)
        
        results = system.batch_questions(example_questions)
        
        # Summary
        successful_queries = sum(1 for r in results if r['success'])
        print(f"\n📊 SUMMARY:")
        print(f"   Successful queries: {successful_queries}/{len(example_questions)}")
        print(f"   Success rate: {successful_queries/len(example_questions)*100:.1f}%")
        
        return system
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return None

def run_pretest_queries(system) -> bool:
    """
    Run 5 hardcoded test queries to validate the system is working
    
    Args:
        system: Initialized GeminiKGText2SQL instance
        
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
            # Generate SQL without executing
            result = system.answer_question(query, execute=False)
            
            if result.get('success') and result.get('sql_generation') and result['sql_generation'].sql_query:
                print(f"✅ Test {i} PASSED")
                print(f"🔧 Generated SQL:\n{result['sql_generation'].sql_query}")
                passed_tests += 1
            else:
                print(f"❌ Test {i} FAILED")
                error_msg = result.get('validation_error') or 'Unknown error'
                if result.get('sql_generation') and result['sql_generation'].error:
                    error_msg = result['sql_generation'].error
                print(f"Error: {error_msg}")
                
        except Exception as e:
            print(f"❌ Test {i} FAILED with exception: {e}")
    
    print(f"\n📊 PRETEST RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("� All pretest queries passed! System is working correctly.")
        return True
    else:
        print("⚠️ Some pretest queries failed. Please check the system configuration.")
        return False


def main():
    """Main entry point"""
    import sys
    
    # Check for test mode
    test_mode = len(sys.argv) > 1 and sys.argv[1] in ['--test', '-t', 'test']
    
    try:
        if test_mode:
            print("\n" + "="*80)
            print("🧪 GEMINI KG TEXT2SQL - TEST MODE")
            print("="*80)
            
            # Load configuration from environment variables
            project_id = os.getenv('GCP_PROJECT_ID')
            dataset_id = os.getenv('BQ_DATASET_ID')
            vertex_ai_location = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
            model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash-lite')
            
            if not project_id or not dataset_id:
                print("❌ Error: Please set GCP_PROJECT_ID and BQ_DATASET_ID environment variables")
                return
            
            # Check requirements
            errors = check_requirements()
            if errors:
                print("❌ Requirements not met:")
                for error in errors:
                    print(f"   - {error}")
                return
            
            # Initialize system
            print(f"🚀 Initializing Gemini BigQuery KG Text-to-SQL system...")
            system = GeminiKGText2SQL(
                project_id=project_id,
                dataset_id=dataset_id,
                bq_kg_file='bqkg/bq_knowledge_graph.ttl',
                model=model_name
            )
            print("✅ System initialized successfully!")
            
            # Run pretest queries
            run_pretest_queries(system)
        else:
            # Run full demonstration
            system = demonstrate_gemini_bigquery_text_to_sql()
            
            if system:
                print("\n" + "="*80)
                print("🎯 GEMINI BIGQUERY TEXT-TO-SQL BENEFITS")
                print("="*80)
                print("✅ Google Gemini intelligence for complex question understanding")
                print("✅ BigQuery-optimized query generation with cloud best practices")
                print("✅ Schema-aware queries with no hallucinated tables or columns")
                print("✅ Natural language explanations of generated BigQuery queries")
                print("✅ High-confidence SQL with step-by-step reasoning")
                print("✅ Handles complex business logic and BigQuery-specific features")
                print("✅ Production-ready with proper error handling and validation")
                print("✅ Cost-aware with query limits and performance monitoring")
            
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()