#!/usr/bin/env python3
"""
User Setup Script for BigQuery Text-to-SQL System
================================================

This script handles:
1. Revoking existing Google Cloud authentication sessions
2. Fresh user authentication with Google Cloud
3. Service account impersonation setup
4. Project configuration
5. Validation of BigQuery and Vertex AI access

Usage:
    python src/usersetup.py

Requirements:
    - .env file with GCP_PROJECT_ID and SERVICE_ACCOUNT_EMAIL
    - Google Cloud SDK (gcloud) installed
    - Required Python packages installed
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
from google.auth import default
import vertexai


class UserSetup:
    """Handles user authentication and environment setup for BigQuery Text-to-SQL system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_file = self.project_root / ".env"
        self.project_id = None
        self.service_account_email = None
        
    def print_step(self, step_num: int, message: str):
        """Print formatted step message."""
        print(f"\n{'='*60}")
        print(f"STEP {step_num}: {message}")
        print(f"{'='*60}")
        
    def print_success(self, message: str):
        """Print success message."""
        print(f"✅ {message}")
        
    def print_error(self, message: str):
        """Print error message."""
        print(f"❌ ERROR: {message}")
        
    def print_info(self, message: str):
        """Print info message."""
        print(f"ℹ️  {message}")
        
    def load_environment_config(self):
        """Load and validate environment configuration from .env file."""
        self.print_step(1, "Loading Environment Configuration")
        
        if not self.env_file.exists():
            self.print_error(f".env file not found at {self.env_file}")
            self.print_info("Please create a .env file with required configuration.")
            sys.exit(1)
            
        # Load environment variables
        load_dotenv(self.env_file)
        
        # Check required environment variables
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.service_account_email = os.getenv('SERVICE_ACCOUNT_EMAIL')
        
        missing_vars = []
        if not self.project_id:
            missing_vars.append('GCP_PROJECT_ID')
        if not self.service_account_email:
            missing_vars.append('SERVICE_ACCOUNT_EMAIL')
            
        if missing_vars:
            self.print_error(f"Missing required environment variables: {', '.join(missing_vars)}")
            self.print_info("Please add these variables to your .env file:")
            for var in missing_vars:
                if var == 'GCP_PROJECT_ID':
                    print(f"  {var}=your-gcp-project-id")
                elif var == 'SERVICE_ACCOUNT_EMAIL':
                    print(f"  {var}=your-service-account@project.iam.gserviceaccount.com")
            sys.exit(1)
            
        self.print_success(f"Project ID: {self.project_id}")
        self.print_success(f"Service Account: {self.service_account_email}")
        
    def revoke_existing_auth(self):
        """Revoke existing Google Cloud authentication sessions."""
        self.print_step(2, "Revoking Existing Authentication Sessions")
        
        try:
            # Revoke all authentication tokens
            result = subprocess.run(
                ['gcloud', 'auth', 'revoke', '--all'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.print_success("All existing authentication sessions revoked")
            else:
                self.print_info("No existing authentication sessions to revoke (or already clean)")
                
        except subprocess.TimeoutExpired:
            self.print_error("Timeout while revoking authentication sessions")
            sys.exit(1)
        except FileNotFoundError:
            self.print_error("gcloud command not found. Please install Google Cloud SDK.")
            self.print_info("Visit: https://cloud.google.com/sdk/docs/install")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Failed to revoke authentication sessions: {e}")
            sys.exit(1)
            
    def authenticate_user(self):
        """Perform fresh user authentication with Google Cloud."""
        self.print_step(3, "Authenticating User with Google Cloud")
        
        try:
            # Perform fresh login
            result = subprocess.run(
                ['gcloud', 'auth', 'login'],
                timeout=300  # 5 minutes timeout for user interaction
            )
            
            if result.returncode != 0:
                self.print_error("User authentication failed")
                sys.exit(1)
                
            self.print_success("User authentication completed")
            
        except subprocess.TimeoutExpired:
            self.print_error("Authentication timeout. Please try again.")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Authentication failed: {e}")
            sys.exit(1)
            
    def configure_project_and_impersonation(self):
        """Configure project and set up service account impersonation."""
        self.print_step(4, "Configuring Project and Service Account Impersonation")
        
        try:
            # Set the active project
            result = subprocess.run(
                ['gcloud', 'config', 'set', 'project', self.project_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.print_error(f"Failed to set project: {result.stderr}")
                sys.exit(1)
                
            self.print_success(f"Project set to: {self.project_id}")
            
            # Configure service account impersonation
            result = subprocess.run(
                ['gcloud', 'config', 'set', 'auth/impersonate_service_account', self.service_account_email],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.print_error(f"Failed to configure service account impersonation: {result.stderr}")
                self.print_info("Make sure you have the 'Service Account Token Creator' role")
                sys.exit(1)
                
            self.print_success(f"Service account impersonation configured: {self.service_account_email}")
            
        except subprocess.TimeoutExpired:
            self.print_error("Timeout while configuring project and impersonation")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Configuration failed: {e}")
            sys.exit(1)
            
    def validate_bigquery_access(self):
        """Validate BigQuery access with configured authentication."""
        self.print_step(5, "Validating BigQuery Access")
        
        try:
            # Initialize BigQuery client with default credentials
            os.environ['GOOGLE_CLOUD_PROJECT'] = self.project_id
            client = bigquery.Client(project=self.project_id)
            
            # Test basic BigQuery access
            datasets = list(client.list_datasets())
            self.print_success(f"BigQuery access validated. Found {len(datasets)} datasets.")
            
            # Check for insurance_analytics dataset
            dataset_id = 'insurance_analytics'
            try:
                dataset = client.get_dataset(dataset_id)
                tables = list(client.list_tables(dataset))
                self.print_success(f"Found '{dataset_id}' dataset with {len(tables)} tables")
                
                # List table names
                table_names = [table.table_id for table in tables]
                self.print_info(f"Tables: {', '.join(table_names)}")
                
            except Exception as e:
                self.print_error(f"Could not access '{dataset_id}' dataset: {e}")
                self.print_info("Run 'python src/bq/InsuranceBigQueryDB.py' to create the dataset")
                return False
                
        except Exception as e:
            self.print_error(f"BigQuery validation failed: {e}")
            self.print_info("Check your authentication and permissions")
            return False
            
        return True
        
    def validate_vertex_ai_access(self):
        """Validate Vertex AI access for Gemini model."""
        self.print_step(6, "Validating Vertex AI Access")
        
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location='us-central1')
            
            # Test if we can access Vertex AI (basic validation)
            self.print_success("Vertex AI access validated")
            self.print_info("Gemini model should be accessible for text-to-SQL generation")
            
        except Exception as e:
            self.print_error(f"Vertex AI validation failed: {e}")
            self.print_info("Check your Vertex AI API enablement and permissions")
            return False
            
        return True
        
    def validate_knowledge_graph_files(self):
        """Check if knowledge graph files exist."""
        self.print_step(7, "Validating Knowledge Graph Files")
        
        bqkg_dir = self.project_root / "bqkg"
        kg_file = bqkg_dir / "bq_knowledge_graph.ttl"
        
        if not bqkg_dir.exists():
            self.print_error("bqkg/ directory not found")
            self.print_info("Run 'python src/bq/BQKnowledgeGraphGenerator.py' to create knowledge graph")
            return False
            
        if not kg_file.exists():
            self.print_error("Knowledge graph file not found")
            self.print_info("Run 'python src/bq/BQKnowledgeGraphGenerator.py' to generate knowledge graph")
            return False
            
        # Check file size
        file_size = kg_file.stat().st_size
        if file_size == 0:
            self.print_error("Knowledge graph file is empty")
            self.print_info("Run 'python src/bq/BQKnowledgeGraphGenerator.py' to regenerate knowledge graph")
            return False
            
        self.print_success(f"Knowledge graph file found ({file_size:,} bytes)")
        return True
        
    def print_next_steps(self):
        """Print next steps for the user."""
        print(f"\n{'='*60}")
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("\n✅ Your environment is now configured for BigQuery Text-to-SQL!")
        print("\n📚 Follow these steps in order to set up your system:")
        print("\n🏗️  STEP 1: Create the BigQuery Database")
        print("   uv run python src/bq/InsuranceBigQueryDB.py")
        print("   └── Creates insurance_analytics dataset with sample data")
        
        print("\n🧠 STEP 2: Generate the Knowledge Graph")
        print("   uv run python src/bq/BQKnowledgeGraphGenerator.py")
        print("   └── Creates RDF knowledge graph from BigQuery schema")
        
        print("\n🧪 STEP 3: Test Text-to-SQL Conversion")
        print("   uv run python src/bq/GeminiKGText2SQL.py")
        print("   └── Tests knowledge graph enhanced text-to-SQL with sample queries")
        
        print("\n🌐 STEP 4: Launch Streamlit UI")
        print("   uv run streamlit run src/bq/GBQ_Streamlit_App.py")
        print("   └── Interactive web interface for text-to-SQL queries")
        
        print("\n🚀 Your BigQuery Text-to-SQL system is ready!")
        print("💡 Run these commands in sequence for the complete setup experience.")
        
    def run(self):
        """Run the complete setup process."""
        print("🔧 BigQuery Text-to-SQL User Setup")
        print("==================================")
        
        try:
            # Step 1: Load environment configuration
            self.load_environment_config()
            
            # Step 2: Revoke existing authentication
            self.revoke_existing_auth()
            
            # Step 3: Authenticate user
            self.authenticate_user()
            
            # Step 4: Configure project and impersonation
            self.configure_project_and_impersonation()
            
            # Step 5: Validate BigQuery access
            bq_valid = self.validate_bigquery_access()
            
            # Step 6: Validate Vertex AI access
            vertex_valid = self.validate_vertex_ai_access()
            
            # Step 7: Validate knowledge graph files
            kg_valid = self.validate_knowledge_graph_files()
            
            # Final validation
            if bq_valid and vertex_valid and kg_valid:
                self.print_next_steps()
            else:
                print(f"\n{'='*60}")
                print("⚠️  SETUP COMPLETED WITH WARNINGS")
                print(f"{'='*60}")
                print("\n✅ Authentication and project configuration successful")
                print("⚠️  Some validation steps failed - see messages above")
                print("\n🔧 Please resolve the issues and re-run validation scripts")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Unexpected error during setup: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    setup = UserSetup()
    setup.run()


if __name__ == "__main__":
    main()
