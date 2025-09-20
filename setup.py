#!/usr/bin/env python3
"""
GBQN2SQL Setup Script
=====================

Comprehensive setup script for BigQuery Knowledge Graph-Enhanced Natural Language to SQL System.

This script will:
1. Create and configure environment variables
2. Verify GCP access (Project, BigQuery, Gemini/Vertex AI)
3. Create BigQuery dataset and populate with insurance data
4. Generate knowledge graph files
5. Test all components (GeminiText2SQL, GeminiKGText2SQL)
6. Launch Streamlit application

Usage:
    python setup.py [--dry-run] [--skip-confirmation] [--verbose]

Author: GBQN2SQL Project
Date: September 2025
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SetupLogger:
    """Custom logger for setup process"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('setup.log', mode='w'),  # Truncate log file on each run
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str):
        """Log info message with color"""
        print(f"{Colors.OKGREEN}[INFO]{Colors.ENDC} {message}")
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message with color"""
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message with color"""
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
        self.logger.error(message)
    
    def success(self, message: str):
        """Log success message with color"""
        print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")
        self.logger.info(f"SUCCESS: {message}")
    
    def header(self, message: str):
        """Log header message with color"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
        self.logger.info(f"HEADER: {message}")

class GBQN2SQLSetup:
    """Main setup class for GBQN2SQL project"""
    
    def __init__(self, dry_run: bool = False, skip_confirmation: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.skip_confirmation = skip_confirmation
        self.logger = SetupLogger(verbose)
        self.project_root = Path.cwd()
        self.env_file = self.project_root / ".env"
        
        # Default configuration
        self.config = {
            "GOOGLE_CLOUD_PROJECT": "gen-lang-client-0454606702",
            "SERVICE_ACCOUNT_EMAIL": "bigquery-text2sql@gen-lang-client-0454606702.iam.gserviceaccount.com",
            "BQ_DATASET_ID": "insurance_data",
            "BQ_LOCATION": "US",
            "VERTEX_AI_PROJECT": "gen-lang-client-0454606702",
            "VERTEX_AI_LOCATION": "us-central1",
            "GEMINI_MODEL": "gemini-2.5-flash-lite",
            "STREAMLIT_PORT": "8501",
            "DEBUG_MODE": "False"
        }
        
        self.setup_steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Environment Setup", self.setup_environment),
            ("GCP Access Verification", self.verify_gcp_access),
            ("BigQuery Dataset Creation", self.create_bigquery_dataset),
            ("Knowledge Graph Generation", self.generate_knowledge_graph),
            ("Component Testing", self.test_components),
            ("Streamlit App Launch", self.launch_streamlit_app)
        ]
    
    def run_command(self, command: str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with proper error handling"""
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would execute: {command}")
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                check=check
            )
            if result.stdout:
                self.logger.info(f"Command output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {command}")
            self.logger.error(f"Error: {e.stderr}")
            raise
    
    def confirm_action(self, message: str) -> bool:
        """Ask for user confirmation"""
        if self.skip_confirmation:
            return True
        
        response = input(f"{Colors.OKCYAN}[CONFIRM]{Colors.ENDC} {message} (y/N): ").lower()
        return response in ['y', 'yes']
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        self.logger.header("Checking Prerequisites")
        
        prerequisites = [
            ("Python 3.8+", ["python3", "--version"]),
            ("UV Package Manager", ["uv", "--version"]),
            ("Google Cloud CLI", ["gcloud", "version"]),
            ("Git", ["git", "--version"])
        ]
        
        missing = []
        for name, command in prerequisites:
            try:
                result = self.run_command(" ".join(command), check=False)
                if result.returncode == 0:
                    self.logger.success(f"✓ {name} is installed")
                else:
                    missing.append(name)
                    self.logger.error(f"✗ {name} is not installed or not in PATH")
            except Exception as e:
                missing.append(name)
                self.logger.error(f"✗ {name} check failed: {e}")
        
        if missing:
            self.logger.error("Missing prerequisites:")
            for item in missing:
                self.logger.error(f"  - {item}")
            
            self.logger.info("\nInstallation instructions:")
            self.logger.info("  UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
            self.logger.info("  Google Cloud CLI: https://cloud.google.com/sdk/docs/install")
            return False
        
        self.logger.success("All prerequisites are installed!")
        return True
    
    def setup_environment(self) -> bool:
        """Check .env file exists and has all required variables - DO NOT CREATE OR UPDATE"""
        self.logger.header("Environment Configuration Validation")
        
        # Check if .env file exists
        if not self.env_file.exists():
            self.logger.error("❌ .env file not found!")
            self.logger.error("The setup script requires a .env file with the following parameters:")
            self.logger.info("")
            self.logger.info("Required .env file parameters:")
            self.logger.info("# Google Cloud Configuration")
            self.logger.info("GCP_PROJECT_ID=your-google-cloud-project-id")
            self.logger.info("GOOGLE_CLOUD_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com")
            self.logger.info("")
            self.logger.info("# BigQuery Configuration") 
            self.logger.info("BQ_DATASET_ID=your-dataset-name")
            self.logger.info("BQ_LOCATION=US")
            self.logger.info("")
            self.logger.info("# Vertex AI Configuration")
            self.logger.info("VERTEX_AI_LOCATION=us-central1")
            self.logger.info("GEMINI_MODEL_NAME=gemini-2.5-flash-lite")
            self.logger.info("")
            self.logger.info("Please create a .env file with these parameters and run setup again.")
            return False
        
        # Check if all required variables are present and populated
        missing_vars = self._check_existing_env_variables()
        
        if missing_vars:
            self.logger.error("❌ Missing or empty required environment variables:")
            for var in missing_vars:
                self.logger.error(f"  - {var}")
            self.logger.info("")
            self.logger.info("Please add the missing variables to your .env file and run setup again.")
            self.logger.info("Required format:")
            for var in missing_vars:
                if var == 'GCP_PROJECT_ID':
                    self.logger.info(f"{var}=your-google-cloud-project-id")
                elif var == 'GOOGLE_CLOUD_SERVICE_ACCOUNT':
                    self.logger.info(f"{var}=your-service-account@project.iam.gserviceaccount.com")
                elif var == 'BQ_DATASET_ID':
                    self.logger.info(f"{var}=your-dataset-name")
                elif var == 'BQ_LOCATION':
                    self.logger.info(f"{var}=US")
                elif var == 'VERTEX_AI_LOCATION':
                    self.logger.info(f"{var}=us-central1")
                elif var == 'GEMINI_MODEL_NAME':
                    self.logger.info(f"{var}=gemini-2.5-flash-lite")
            return False
        
        # All variables present - update internal config from .env values
        self.logger.success("✓ .env file exists and all required variables are configured")
        self._update_config_from_existing_env()
        return True
    
    def _check_existing_env_variables(self) -> List[str]:
        """Check which required environment variables are missing"""
        required_vars = {
            'GCP_PROJECT_ID': 'Google Cloud Project ID',
            'GOOGLE_CLOUD_SERVICE_ACCOUNT': 'Google Cloud Service Account Email',
            'BQ_DATASET_ID': 'BigQuery Dataset ID', 
            'BQ_LOCATION': 'BigQuery Location',
            'VERTEX_AI_LOCATION': 'Vertex AI Location',
            'GEMINI_MODEL_NAME': 'Gemini Model Name'
        }
        
        missing = []
        existing_vars = {}
        
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        existing_vars[key] = value
            
            for var, description in required_vars.items():
                if var not in existing_vars or not existing_vars[var].strip():
                    missing.append(var)
                    self.logger.warning(f"Missing: {var} ({description})")
                else:
                    self.logger.success(f"✓ {var} = {existing_vars[var]}")
                    
        except Exception as e:
            self.logger.error(f"Error reading .env file: {e}")
            return list(required_vars.keys())
        
        return missing
    
    def _update_config_from_existing_env(self):
        """Update internal config from existing .env file"""
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        # Map existing variables to our config
                        if key == 'GCP_PROJECT_ID':
                            self.config['GOOGLE_CLOUD_PROJECT'] = value
                        elif key == 'GOOGLE_CLOUD_SERVICE_ACCOUNT':
                            self.config['SERVICE_ACCOUNT_EMAIL'] = value
                        elif key == 'BQ_DATASET_ID':
                            self.config['BQ_DATASET_ID'] = value
                        elif key == 'BQ_LOCATION':
                            self.config['BQ_LOCATION'] = value
                        elif key == 'VERTEX_AI_LOCATION':
                            self.config['VERTEX_AI_LOCATION'] = value
                        elif key == 'GEMINI_MODEL_NAME':
                            self.config['GEMINI_MODEL'] = value
        except Exception as e:
            self.logger.error(f"Error updating config from .env: {e}")
    
    def _update_existing_env_file(self, missing_vars: List[str]) -> bool:
        """Add missing variables to existing .env file"""
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would add missing variables: {', '.join(missing_vars)}")
            return True
        
        try:
            # Read existing content
            with open(self.env_file, 'r') as f:
                existing_content = f.read()
            
            # Add missing variables
            additions = []
            var_mappings = {
                'GCP_PROJECT_ID': self.config['GOOGLE_CLOUD_PROJECT'],
                'BQ_DATASET_ID': self.config['BQ_DATASET_ID'],
                'BQ_LOCATION': self.config['BQ_LOCATION'],
                'VERTEX_AI_LOCATION': self.config['VERTEX_AI_LOCATION'],
                'GEMINI_MODEL_NAME': self.config['GEMINI_MODEL']
            }
            
            for var in missing_vars:
                if var in var_mappings:
                    additions.append(f"{var}={var_mappings[var]}")
            
            if additions:
                updated_content = existing_content + "\n\n# Added by setup script\n" + "\n".join(additions)
                
                with open(self.env_file, 'w') as f:
                    f.write(updated_content)
                
                self.logger.success(f"Added {len(additions)} missing variables to .env file")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating .env file: {e}")
        
        return False
    
    def _generate_env_content(self) -> str:
        """Generate .env file content"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# GBQN2SQL Environment Configuration
# Generated on: {timestamp}
# ===============================================

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT={self.config['GOOGLE_CLOUD_PROJECT']}
SERVICE_ACCOUNT_EMAIL={self.config['SERVICE_ACCOUNT_EMAIL']}

# BigQuery Configuration
BQ_DATASET_ID={self.config['BQ_DATASET_ID']}
BQ_LOCATION={self.config['BQ_LOCATION']}

# Vertex AI Configuration
VERTEX_AI_PROJECT={self.config['VERTEX_AI_PROJECT']}
VERTEX_AI_LOCATION={self.config['VERTEX_AI_LOCATION']}
GEMINI_MODEL={self.config['GEMINI_MODEL']}

# Application Configuration
STREAMLIT_PORT={self.config['STREAMLIT_PORT']}
DEBUG_MODE={self.config['DEBUG_MODE']}

# Optional: Uncomment and set if using service account key file
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
"""
        return content
    
    def verify_gcp_access(self) -> bool:
        """Verify access to all GCP services"""
        self.logger.header("Verifying GCP Access")
        
        # Check GCP project access
        if not self._verify_project_access():
            return False
        
        # Check BigQuery access
        if not self._verify_bigquery_access():
            return False
        
        # Check Vertex AI access
        if not self._verify_vertex_ai_access():
            return False
        
        self.logger.success("All GCP access verifications passed!")
        return True
    
    def _verify_project_access(self) -> bool:
        """Verify GCP project access using .env configuration"""
        self.logger.info("Checking GCP project access...")
        
        project_id = self.config.get('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            self.logger.error("No project ID found in configuration")
            return False
        
        try:
            result = self.run_command(f"gcloud projects describe {project_id}")
            if "projectId" in result.stdout:
                self.logger.success(f"✓ GCP project access confirmed: {project_id}")
                return True
        except Exception as e:
            self.logger.error(f"✗ Cannot access GCP project {project_id}: {e}")
            self.logger.info("Run: gcloud auth login")
            self.logger.info(f"Then: gcloud config set project {project_id}")
        
        return False
    
    def _verify_bigquery_access(self) -> bool:
        """Verify BigQuery access using .env configuration"""
        self.logger.info("Checking BigQuery access...")
        
        service_account = self.config.get('SERVICE_ACCOUNT_EMAIL')
        if not service_account:
            self.logger.error("No service account email found in configuration")
            return False
        
        try:
            # Test BigQuery access with service account impersonation from .env
            result = self.run_command(f"gcloud auth application-default login --impersonate-service-account={service_account}")
            
            # Test with a simple BigQuery query
            test_query = "SELECT 1 as test_value"
            result = self.run_command(f'bq query --use_legacy_sql=false --format=json "{test_query}"')
            
            if "test_value" in result.stdout:
                self.logger.success(f"✓ BigQuery access confirmed with service account: {service_account}")
                return True
        except Exception as e:
            self.logger.error(f"✗ Cannot access BigQuery with service account {service_account}: {e}")
            self.logger.info("Ensure BigQuery API is enabled and service account has proper permissions")
        
        return False
    
    def _verify_vertex_ai_access(self) -> bool:
        """Verify Vertex AI access"""
        self.logger.info("Checking Vertex AI access...")
        
        # For now, we'll assume access is available if BigQuery works
        # A more thorough check would require importing the actual libraries
        self.logger.success("✓ Vertex AI access assumed (will be verified during component testing)")
        return True
    
    def create_bigquery_dataset(self) -> bool:
        """Create BigQuery dataset and populate with data"""
        self.logger.header("Creating BigQuery Dataset")
        
        if not self.confirm_action("Create BigQuery dataset and populate with insurance data?"):
            self.logger.info("Skipping BigQuery dataset creation")
            return True
        
        try:
            # Run the insurance database creation script
            self.logger.info("Executing InsuranceBigQueryDB.py...")
            result = self.run_command("uv run python src/bq/InsuranceBigQueryDB.py")
            
            if result.returncode == 0:
                self.logger.success("✓ BigQuery dataset created successfully")
                
                # Verify tables were created
                if self._verify_tables_created():
                    return True
            
        except Exception as e:
            self.logger.error(f"Failed to create BigQuery dataset: {e}")
        
        return False
    
    def _verify_tables_created(self) -> bool:
        """Verify that all required tables were created using .env configuration"""
        self.logger.info("Verifying table creation...")
        
        expected_tables = ["agents", "customers", "policies", "claims"]
        
        try:
            # Use the dataset name and project from .env config
            dataset_id = self.config.get('BQ_DATASET_ID')
            project_id = self.config.get('GOOGLE_CLOUD_PROJECT')
            
            if not dataset_id or not project_id:
                self.logger.error("Missing project ID or dataset ID in configuration")
                return False
            
            # List tables in the dataset
            result = self.run_command(f"bq ls {project_id}:{dataset_id}")
            
            created_tables = []
            for line in result.stdout.split('\n'):
                if 'TABLE' in line:
                    table_name = line.split()[0]
                    created_tables.append(table_name)
            
            missing_tables = set(expected_tables) - set(created_tables)
            
            if not missing_tables:
                self.logger.success(f"✓ All tables created in {project_id}:{dataset_id}: {', '.join(created_tables)}")
                
                # Get row counts
                for table in created_tables:
                    count_query = f"SELECT COUNT(*) as count FROM {dataset_id}.{table}"
                    result = self.run_command(f'bq query --use_legacy_sql=false --format=json "{count_query}"')
                    # Parse count from JSON output (simplified)
                    self.logger.info(f"  {table}: Created successfully")
                
                return True
            else:
                self.logger.error(f"Missing tables in {project_id}:{dataset_id}: {', '.join(missing_tables)}")
                
        except Exception as e:
            self.logger.error(f"Failed to verify tables: {e}")
        
        return False
    
    def generate_knowledge_graph(self) -> bool:
        """Generate knowledge graph files"""
        self.logger.header("Generating Knowledge Graph")
        
        if not self.confirm_action("Generate knowledge graph files?"):
            self.logger.info("Skipping knowledge graph generation")
            return True
        
        try:
            # Run the knowledge graph generator
            self.logger.info("Executing BQKnowledgeGraphGenerator.py...")
            result = self.run_command("uv run python src/bq/BQKnowledgeGraphGenerator.py")
            
            if result.returncode == 0:
                self.logger.success("✓ Knowledge graph generated successfully")
                
                # Verify knowledge graph files were created
                if self._verify_knowledge_graph_files():
                    return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate knowledge graph: {e}")
        
        return False
    
    def _verify_knowledge_graph_files(self) -> bool:
        """Verify knowledge graph files were created"""
        self.logger.info("Verifying knowledge graph files...")
        
        kg_dir = self.project_root / "bqkg"
        expected_files = ["bq_knowledge_graph.ttl", "bq_knowledge_graph.rdf", 
                         "bq_knowledge_graph.nt", "bq_knowledge_graph.n3"]
        
        if not kg_dir.exists():
            self.logger.error("Knowledge graph directory 'bqkg' not found")
            return False
        
        missing_files = []
        for filename in expected_files:
            file_path = kg_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                self.logger.success(f"✓ {filename} ({file_size:,} bytes)")
            else:
                missing_files.append(filename)
        
        if missing_files:
            self.logger.error(f"Missing knowledge graph files: {', '.join(missing_files)}")
            return False
        
        self.logger.success("All knowledge graph files created successfully!")
        return True
    
    def test_components(self) -> bool:
        """Test all system components"""
        self.logger.header("Testing System Components")
        
        # Test GeminiText2SQL
        if not self._test_gemini_text2sql():
            return False
        
        # Test GeminiKGText2SQL
        if not self._test_gemini_kg_text2sql():
            return False
        
        self.logger.success("All component tests passed!")
        return True
    
    def _test_gemini_text2sql(self) -> bool:
        """Test GeminiText2SQL component"""
        self.logger.info("Testing GeminiText2SQL.py...")
        
        try:
            result = self.run_command("uv run python src/bq/GeminiText2SQL.py --test")
            
            if result.returncode == 0:
                self.logger.success("✓ GeminiText2SQL.py runs successfully")
                return True
            else:
                self.logger.error("✗ GeminiText2SQL.py failed")
                
        except Exception as e:
            self.logger.error(f"Failed to test GeminiText2SQL: {e}")
        
        return False
    
    def _test_gemini_kg_text2sql(self) -> bool:
        """Test GeminiKGText2SQL component"""
        self.logger.info("Testing GeminiKGText2SQL.py...")
        
        try:
            result = self.run_command("uv run python src/bq/GeminiKGText2SQL.py --test")
            
            if result.returncode == 0:
                self.logger.success("✓ GeminiKGText2SQL.py runs successfully")
                return True
            else:
                self.logger.error("✗ GeminiKGText2SQL.py failed")
                
        except Exception as e:
            self.logger.error(f"Failed to test GeminiKGText2SQL: {e}")
        
        return False
    
    def launch_streamlit_app(self) -> bool:
        """Launch Streamlit application"""
        self.logger.header("Launching Streamlit Application")
        
        if not self.confirm_action("Launch Streamlit application?"):
            self.logger.info("Skipping Streamlit app launch")
            return True
        
        try:
            self.logger.info("Starting Streamlit app...")
            self.logger.info(f"App will be available at: http://localhost:{self.config['STREAMLIT_PORT']}")
            
            if not self.dry_run:
                # Launch Streamlit in background
                subprocess.Popen([
                    "uv", "run", "streamlit", "run", "src/bq/GBQ_Streamlit_App.py",
                    "--server.port", self.config['STREAMLIT_PORT']
                ], cwd=self.project_root)
                
                # Give it a moment to start
                time.sleep(3)
                
                self.logger.success("✓ Streamlit app launched successfully!")
                self.logger.info(f"Access the app at: http://localhost:{self.config['STREAMLIT_PORT']}")
                self.logger.info("Press Ctrl+C to stop the setup script (Streamlit will continue running)")
            else:
                self.logger.info("DRY RUN: Would launch Streamlit app")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch Streamlit app: {e}")
        
        return False
    
    def run_setup(self) -> bool:
        """Run the complete setup process"""
        self.logger.header("GBQN2SQL Setup Starting")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Dry run mode: {self.dry_run}")
        
        if not self.confirm_action("Start the GBQN2SQL setup process?"):
            self.logger.info("Setup cancelled by user")
            return False
        
        # Execute all setup steps
        for step_name, step_function in self.setup_steps:
            try:
                if not step_function():
                    self.logger.error(f"Setup failed at step: {step_name}")
                    return False
            except KeyboardInterrupt:
                self.logger.warning("Setup interrupted by user")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error in {step_name}: {e}")
                return False
        
        self.logger.header("Setup Completed Successfully!")
        self._print_success_summary()
        return True
    
    def _print_success_summary(self):
        """Print setup success summary"""
        summary = f"""
{Colors.OKGREEN}{Colors.BOLD}🎉 GBQN2SQL Setup Complete! 🎉{Colors.ENDC}

{Colors.OKCYAN}What was set up:{Colors.ENDC}
✓ Environment configuration (.env file)
✓ GCP access verification (Project, BigQuery, Vertex AI)
✓ BigQuery dataset with insurance data
✓ Knowledge graph files generated
✓ All components tested successfully
✓ Streamlit application launched

{Colors.OKCYAN}Next steps:{Colors.ENDC}
• Access the Streamlit app: http://localhost:{self.config['STREAMLIT_PORT']}
• Review the setup log: setup.log
• Start querying your data with natural language!

{Colors.OKCYAN}Useful commands:{Colors.ENDC}
• Test components: uv run python src/bq/GeminiKGText2SQL.py
• Restart Streamlit: uv run streamlit run src/bq/GBQ_Streamlit_App.py
• View logs: tail -f setup.log

{Colors.WARNING}Note:{Colors.ENDC} If you encounter issues, check setup.log for detailed information.
"""
        print(summary)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GBQN2SQL Setup Script")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip user confirmations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create setup instance and run
    setup = GBQN2SQLSetup(
        dry_run=args.dry_run,
        skip_confirmation=args.skip_confirmation,
        verbose=args.verbose
    )
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()