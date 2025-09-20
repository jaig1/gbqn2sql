"""
BigQuery Insurance Database Creator

This module creates and populates BigQuery tables with comprehensive insurance sample data.
It mirrors the SQLite implementation but optimized for BigQuery with:
- Proper BigQuery data types
- Table partitioning and clustering
- Batch inserts for performance
- Comprehensive error handling
- Business rule validation

Author: Generated for n2sqlwkg Knowledge Graph Text-to-SQL System
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, BadRequest
from google.oauth2 import service_account
from google.auth import impersonated_credentials, default
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceBigQueryDB:
    """
    BigQuery Insurance Database Manager
    
    Handles creation, population, and management of insurance sample data in BigQuery.
    Supports the same business rules and data structure as the SQLite version.
    """
    
    def __init__(self):
        """Initialize BigQuery client and configuration"""
        self.client = None
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.dataset_id = os.getenv('BQ_DATASET_ID', 'insurance_analytics')
        self.location = os.getenv('BQ_LOCATION', 'US')
        self.table_prefix = os.getenv('BQ_TABLE_PREFIX', 'insurance_')
        self.enable_partitioning = os.getenv('BQ_ENABLE_PARTITIONING', 'true').lower() == 'true'
        self.enable_clustering = os.getenv('BQ_ENABLE_CLUSTERING', 'true').lower() == 'true'
        
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize BigQuery client with proper authentication"""
        try:
            # Option 1: Service Account Impersonation (recommended for development)
            service_account_email = os.getenv('GOOGLE_CLOUD_SERVICE_ACCOUNT')
            if service_account_email:
                from google.auth import impersonated_credentials
                from google.auth import default
                
                source_credentials, _ = default()
                target_credentials = impersonated_credentials.Credentials(
                    source_credentials=source_credentials,
                    target_principal=service_account_email,
                    target_scopes=['https://www.googleapis.com/auth/cloud-platform'],
                )
                self.client = bigquery.Client(credentials=target_credentials, project=self.project_id)
                logger.info(f"BigQuery client initialized with service account impersonation: {service_account_email}")
                return
            
            # Option 2: Service Account JSON file
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = bigquery.Client(credentials=credentials, project=self.project_id)
                logger.info(f"BigQuery client initialized with service account file: {credentials_path}")
                return
            
            # Option 3: Service Account JSON string
            credentials_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
            if credentials_json:
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                self.client = bigquery.Client(credentials=credentials, project=self.project_id)
                logger.info("BigQuery client initialized with service account JSON string")
                return
            
            # Option 4: Application Default Credentials
            self.client = bigquery.Client(project=self.project_id)
            logger.info("BigQuery client initialized with Application Default Credentials")
            
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise Exception(f"BigQuery authentication failed: {str(e)}")
    
    def create_dataset(self) -> bool:
        """Create BigQuery dataset if it doesn't exist"""
        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
                return True
            except NotFound:
                pass
            
            # Create dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.location
            dataset.description = "Insurance Analytics Dataset - Sample data for Knowledge Graph Text-to-SQL demonstration"
            
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {self.dataset_id} in {self.location}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            return False
    
    def _get_table_schema(self, table_name: str) -> List[bigquery.SchemaField]:
        """Get BigQuery schema for each table"""
        schemas = {
            'agents': [
                bigquery.SchemaField("agent_id", "INT64", mode="REQUIRED", description="Unique agent identifier"),
                bigquery.SchemaField("name", "STRING", mode="REQUIRED", description="Agent full name"),
                bigquery.SchemaField("email", "STRING", mode="NULLABLE", description="Agent email address"),
                bigquery.SchemaField("phone", "STRING", mode="NULLABLE", description="Agent phone number"),
            ],
            'customers': [
                bigquery.SchemaField("customer_id", "INT64", mode="REQUIRED", description="Unique customer identifier"),
                bigquery.SchemaField("name", "STRING", mode="REQUIRED", description="Customer full name"),
                bigquery.SchemaField("email", "STRING", mode="NULLABLE", description="Customer email address"),
                bigquery.SchemaField("phone", "STRING", mode="NULLABLE", description="Customer phone number"),
                bigquery.SchemaField("address", "STRING", mode="NULLABLE", description="Customer address"),
                bigquery.SchemaField("date_of_birth", "DATE", mode="NULLABLE", description="Customer date of birth"),
            ],
            'policies': [
                bigquery.SchemaField("policy_id", "INT64", mode="REQUIRED", description="Unique policy identifier"),
                bigquery.SchemaField("policy_number", "STRING", mode="REQUIRED", description="Policy number"),
                bigquery.SchemaField("customer_id", "INT64", mode="REQUIRED", description="Reference to customer"),
                bigquery.SchemaField("agent_id", "INT64", mode="REQUIRED", description="Reference to agent"),
                bigquery.SchemaField("policy_type", "STRING", mode="REQUIRED", description="Type of insurance policy"),
                bigquery.SchemaField("premium_amount", "NUMERIC", mode="REQUIRED", description="Annual premium amount"),
                bigquery.SchemaField("start_date", "DATE", mode="REQUIRED", description="Policy start date"),
                bigquery.SchemaField("end_date", "DATE", mode="REQUIRED", description="Policy end date"),
                bigquery.SchemaField("status", "STRING", mode="REQUIRED", description="Policy status"),
            ],
            'claims': [
                bigquery.SchemaField("claim_id", "INT64", mode="REQUIRED", description="Unique claim identifier"),
                bigquery.SchemaField("claim_number", "STRING", mode="REQUIRED", description="Claim number"),
                bigquery.SchemaField("policy_id", "INT64", mode="REQUIRED", description="Reference to policy"),
                bigquery.SchemaField("claim_date", "DATE", mode="REQUIRED", description="Date claim was filed"),
                bigquery.SchemaField("claim_amount", "NUMERIC", mode="REQUIRED", description="Claim amount"),
                bigquery.SchemaField("claim_status", "STRING", mode="REQUIRED", description="Current claim status"),
                bigquery.SchemaField("description", "STRING", mode="NULLABLE", description="Claim description"),
            ]
        }
        
        return schemas.get(table_name, [])
    
    def _create_table(self, table_name: str) -> bool:
        """Create a single BigQuery table with optimized configuration"""
        try:
            full_table_name = f"{self.table_prefix}{table_name}"
            table_ref = self.client.dataset(self.dataset_id).table(full_table_name)
            
            # Check if table exists
            try:
                self.client.get_table(table_ref)
                logger.info(f"Table {full_table_name} already exists")
                return True
            except NotFound:
                pass
            
            # Create table
            schema = self._get_table_schema(table_name)
            table = bigquery.Table(table_ref, schema=schema)
            table.description = f"Insurance {table_name} data for Knowledge Graph Text-to-SQL demonstration"
            
            # Configure partitioning
            if self.enable_partitioning and table_name == 'policies':
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="start_date"
                )
            elif self.enable_partitioning and table_name == 'claims':
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="claim_date"
                )
            
            # Configure clustering
            if self.enable_clustering:
                if table_name == 'policies':
                    table.clustering_fields = ["policy_type", "status"]
                elif table_name == 'claims':
                    table.clustering_fields = ["claim_status"]
                elif table_name == 'customers':
                    table.clustering_fields = ["name"]
                elif table_name == 'agents':
                    table.clustering_fields = ["name"]
            
            table = self.client.create_table(table)
            logger.info(f"Created table {full_table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """Create all insurance tables"""
        tables = ['agents', 'customers', 'policies', 'claims']
        
        for table_name in tables:
            if not self._create_table(table_name):
                return False
        
        logger.info("All tables created successfully")
        return True
    
    def _get_sample_data(self) -> Dict[str, List[Dict]]:
        """Get the same sample data as SQLite version, formatted for BigQuery"""
        today = date.today()
        
        data = {
            'agents': [
                {'agent_id': 1, 'name': 'John Smith', 'email': 'john.smith@insurance.com', 'phone': '555-0101'},
                {'agent_id': 2, 'name': 'Sarah Johnson', 'email': 'sarah.johnson@insurance.com', 'phone': '555-0102'},
                {'agent_id': 3, 'name': 'Mike Brown', 'email': 'mike.brown@insurance.com', 'phone': '555-0103'},
                {'agent_id': 4, 'name': 'Emily Davis', 'email': 'emily.davis@insurance.com', 'phone': '555-0104'},
                {'agent_id': 5, 'name': 'Robert Wilson', 'email': 'robert.wilson@insurance.com', 'phone': '555-0105'},
                {'agent_id': 6, 'name': 'Jessica Martinez', 'email': 'jessica.martinez@insurance.com', 'phone': '555-0106'},
                {'agent_id': 7, 'name': 'David Thompson', 'email': 'david.thompson@insurance.com', 'phone': '555-0107'},
                {'agent_id': 8, 'name': 'Lisa Anderson', 'email': 'lisa.anderson@insurance.com', 'phone': '555-0108'}
            ],
            
            'customers': [
                # Complete profiles
                {'customer_id': 1, 'name': 'Alice Wilson', 'email': 'alice.wilson@email.com', 'phone': '555-1001', 'address': '123 Main St, Springfield', 'date_of_birth': date(1985, 3, 15)},
                {'customer_id': 2, 'name': 'Bob Davis', 'email': 'bob.davis@email.com', 'phone': '555-1002', 'address': '456 Oak Ave, Springfield', 'date_of_birth': date(1978, 7, 22)},
                {'customer_id': 3, 'name': 'Carol Martinez', 'email': 'carol.martinez@email.com', 'phone': '555-1003', 'address': '789 Pine St, Springfield', 'date_of_birth': date(1992, 11, 8)},
                {'customer_id': 4, 'name': 'David Lee', 'email': 'david.lee@email.com', 'phone': '555-1004', 'address': '321 Elm St, Springfield', 'date_of_birth': date(1988, 5, 30)},
                {'customer_id': 5, 'name': 'Emma Thompson', 'email': 'emma.thompson@email.com', 'phone': '555-1005', 'address': '567 Maple Ave, Springfield', 'date_of_birth': date(1990, 12, 3)},
                {'customer_id': 6, 'name': 'Frank Miller', 'email': 'frank.miller@email.com', 'phone': '555-1006', 'address': '890 Cedar St, Springfield', 'date_of_birth': date(1982, 8, 18)},
                {'customer_id': 7, 'name': 'Grace Chen', 'email': 'grace.chen@email.com', 'phone': '555-1007', 'address': '234 Birch Rd, Springfield', 'date_of_birth': date(1987, 4, 25)},
                {'customer_id': 8, 'name': 'Henry Garcia', 'email': 'henry.garcia@email.com', 'phone': '555-1008', 'address': '678 Walnut Dr, Springfield', 'date_of_birth': date(1995, 1, 12)},
                {'customer_id': 9, 'name': 'Isabella Rodriguez', 'email': 'isabella.rodriguez@email.com', 'phone': '555-1009', 'address': '345 Ash Ln, Springfield', 'date_of_birth': date(1989, 9, 7)},
                {'customer_id': 10, 'name': 'James White', 'email': 'james.white@email.com', 'phone': '555-1010', 'address': '901 Spruce St, Springfield', 'date_of_birth': date(1975, 6, 14)},
                
                # Incomplete profiles for data quality demonstration
                {'customer_id': 11, 'name': 'Kevin Brown', 'email': None, 'phone': '555-1011', 'address': '123 Unknown St, Springfield', 'date_of_birth': date(1986, 3, 20)},
                {'customer_id': 12, 'name': 'Laura Jones', 'email': 'laura.jones@email.com', 'phone': None, 'address': '456 Mystery Ave, Springfield', 'date_of_birth': None},
                {'customer_id': 13, 'name': 'Unknown Customer', 'email': 'unknown@email.com', 'phone': '555-1013', 'address': '789 Anonymous Rd, Springfield', 'date_of_birth': date(1991, 7, 15)},
                {'customer_id': 14, 'name': 'Michael Taylor', 'email': 'michael.taylor@email.com', 'phone': '555-1014', 'address': None, 'date_of_birth': None},
                {'customer_id': 15, 'name': 'Nancy Wilson', 'email': None, 'phone': None, 'address': '234 Incomplete St, Springfield', 'date_of_birth': date(1983, 11, 30)}
            ],
            
            'policies': [
                # BUDGET SEGMENT (≤ $2000)
                {'policy_id': 1, 'policy_number': 'POL-2024-001', 'customer_id': 1, 'agent_id': 1, 'policy_type': 'Auto', 'premium_amount': 800.00, 'start_date': date(2024, 1, 1), 'end_date': date(2024, 12, 31), 'status': 'Active'},
                {'policy_id': 2, 'policy_number': 'POL-2024-002', 'customer_id': 2, 'agent_id': 1, 'policy_type': 'Auto', 'premium_amount': 1200.00, 'start_date': date(2024, 2, 1), 'end_date': date(2025, 1, 31), 'status': 'Active'},
                {'policy_id': 3, 'policy_number': 'POL-2024-003', 'customer_id': 3, 'agent_id': 1, 'policy_type': 'Home', 'premium_amount': 1500.00, 'start_date': date(2024, 3, 1), 'end_date': date(2025, 2, 28), 'status': 'Active'},
                {'policy_id': 4, 'policy_number': 'POL-2024-004', 'customer_id': 4, 'agent_id': 1, 'policy_type': 'Auto', 'premium_amount': 1800.00, 'start_date': date(2024, 4, 1), 'end_date': date(2025, 3, 31), 'status': 'Active'},
                {'policy_id': 5, 'policy_number': 'POL-2024-005', 'customer_id': 5, 'agent_id': 1, 'policy_type': 'Renters', 'premium_amount': 600.00, 'start_date': date(2024, 5, 1), 'end_date': date(2025, 4, 30), 'status': 'Active'},
                
                # STANDARD SEGMENT ($2000-$5000)
                {'policy_id': 6, 'policy_number': 'POL-2024-006', 'customer_id': 6, 'agent_id': 2, 'policy_type': 'Home', 'premium_amount': 2500.00, 'start_date': date(2024, 1, 15), 'end_date': date(2025, 1, 14), 'status': 'Active'},
                {'policy_id': 7, 'policy_number': 'POL-2024-007', 'customer_id': 7, 'agent_id': 2, 'policy_type': 'Auto', 'premium_amount': 3200.00, 'start_date': date(2024, 2, 15), 'end_date': date(2025, 2, 14), 'status': 'Active'},
                {'policy_id': 8, 'policy_number': 'POL-2024-008', 'customer_id': 8, 'agent_id': 2, 'policy_type': 'Life', 'premium_amount': 4000.00, 'start_date': date(2024, 3, 15), 'end_date': date(2025, 3, 14), 'status': 'Active'},
                {'policy_id': 9, 'policy_number': 'POL-2024-009', 'customer_id': 9, 'agent_id': 2, 'policy_type': 'Home', 'premium_amount': 3800.00, 'start_date': date(2024, 4, 15), 'end_date': date(2025, 4, 14), 'status': 'Active'},
                {'policy_id': 10, 'policy_number': 'POL-2024-010', 'customer_id': 10, 'agent_id': 2, 'policy_type': 'Auto', 'premium_amount': 2800.00, 'start_date': date(2024, 5, 15), 'end_date': date(2025, 5, 14), 'status': 'Active'},
                
                # PREMIUM SEGMENT (> $5000)
                {'policy_id': 11, 'policy_number': 'POL-2024-011', 'customer_id': 1, 'agent_id': 3, 'policy_type': 'Life', 'premium_amount': 6500.00, 'start_date': date(2024, 1, 20), 'end_date': date(2025, 1, 19), 'status': 'Active'},
                {'policy_id': 12, 'policy_number': 'POL-2024-012', 'customer_id': 2, 'agent_id': 3, 'policy_type': 'Home', 'premium_amount': 8200.00, 'start_date': date(2024, 2, 20), 'end_date': date(2025, 2, 19), 'status': 'Active'},
                {'policy_id': 13, 'policy_number': 'POL-2024-013', 'customer_id': 3, 'agent_id': 3, 'policy_type': 'Auto', 'premium_amount': 7500.00, 'start_date': date(2024, 3, 20), 'end_date': date(2025, 3, 19), 'status': 'Active'},
                {'policy_id': 14, 'policy_number': 'POL-2024-014', 'customer_id': 4, 'agent_id': 3, 'policy_type': 'Life', 'premium_amount': 9800.00, 'start_date': date(2024, 4, 20), 'end_date': date(2025, 4, 19), 'status': 'Active'},
                {'policy_id': 15, 'policy_number': 'POL-2024-015', 'customer_id': 5, 'agent_id': 3, 'policy_type': 'Umbrella', 'premium_amount': 12000.00, 'start_date': date(2024, 5, 20), 'end_date': date(2025, 5, 19), 'status': 'Active'},
                
                # EXPIRED POLICIES
                {'policy_id': 16, 'policy_number': 'POL-2023-001', 'customer_id': 6, 'agent_id': 4, 'policy_type': 'Auto', 'premium_amount': 1400.00, 'start_date': date(2023, 1, 1), 'end_date': date(2023, 12, 31), 'status': 'Expired'},
                {'policy_id': 17, 'policy_number': 'POL-2023-002', 'customer_id': 7, 'agent_id': 4, 'policy_type': 'Home', 'premium_amount': 2200.00, 'start_date': date(2023, 2, 1), 'end_date': date(2024, 1, 31), 'status': 'Expired'},
                {'policy_id': 18, 'policy_number': 'POL-2023-003', 'customer_id': 8, 'agent_id': 4, 'policy_type': 'Life', 'premium_amount': 3500.00, 'start_date': date(2023, 3, 1), 'end_date': date(2024, 2, 29), 'status': 'Expired'},
                
                # EXPIRING SOON
                {'policy_id': 19, 'policy_number': 'POL-2024-019', 'customer_id': 9, 'agent_id': 5, 'policy_type': 'Auto', 'premium_amount': 1600.00, 'start_date': date(2024, 1, 1), 'end_date': today + timedelta(days=15), 'status': 'Active'},
                {'policy_id': 20, 'policy_number': 'POL-2024-020', 'customer_id': 10, 'agent_id': 5, 'policy_type': 'Home', 'premium_amount': 2400.00, 'start_date': date(2024, 2, 1), 'end_date': today + timedelta(days=25), 'status': 'Active'},
                {'policy_id': 21, 'policy_number': 'POL-2024-021', 'customer_id': 11, 'agent_id': 5, 'policy_type': 'Auto', 'premium_amount': 1300.00, 'start_date': date(2024, 3, 1), 'end_date': today + timedelta(days=28), 'status': 'Active'},
                
                # Additional policies for agent performance demonstration
                {'policy_id': 22, 'policy_number': 'POL-2024-022', 'customer_id': 12, 'agent_id': 6, 'policy_type': 'Auto', 'premium_amount': 1100.00, 'start_date': date(2024, 8, 1), 'end_date': date(2025, 7, 31), 'status': 'Active'},
                {'policy_id': 23, 'policy_number': 'POL-2024-023', 'customer_id': 13, 'agent_id': 6, 'policy_type': 'Home', 'premium_amount': 1900.00, 'start_date': date(2024, 8, 15), 'end_date': date(2025, 8, 14), 'status': 'Active'},
                {'policy_id': 24, 'policy_number': 'POL-2024-024', 'customer_id': 14, 'agent_id': 7, 'policy_type': 'Auto', 'premium_amount': 1000.00, 'start_date': date(2024, 9, 1), 'end_date': date(2025, 8, 31), 'status': 'Active'},
                {'policy_id': 25, 'policy_number': 'POL-2024-025', 'customer_id': 15, 'agent_id': 8, 'policy_type': 'Renters', 'premium_amount': 700.00, 'start_date': date(2024, 9, 15), 'end_date': date(2025, 9, 14), 'status': 'Active'},
                
                # More policies for top performers
                {'policy_id': 26, 'policy_number': 'POL-2024-026', 'customer_id': 1, 'agent_id': 1, 'policy_type': 'Umbrella', 'premium_amount': 5500.00, 'start_date': date(2024, 6, 1), 'end_date': date(2025, 5, 31), 'status': 'Active'},
                {'policy_id': 27, 'policy_number': 'POL-2024-027', 'customer_id': 6, 'agent_id': 1, 'policy_type': 'Life', 'premium_amount': 4200.00, 'start_date': date(2024, 6, 15), 'end_date': date(2025, 6, 14), 'status': 'Active'},
                {'policy_id': 28, 'policy_number': 'POL-2024-028', 'customer_id': 11, 'agent_id': 1, 'policy_type': 'Auto', 'premium_amount': 1650.00, 'start_date': date(2024, 7, 1), 'end_date': date(2025, 6, 30), 'status': 'Active'},
                
                {'policy_id': 29, 'policy_number': 'POL-2024-029', 'customer_id': 2, 'agent_id': 2, 'policy_type': 'Umbrella', 'premium_amount': 6800.00, 'start_date': date(2024, 6, 1), 'end_date': date(2025, 5, 31), 'status': 'Active'},
                {'policy_id': 30, 'policy_number': 'POL-2024-030', 'customer_id': 12, 'agent_id': 2, 'policy_type': 'Life', 'premium_amount': 3900.00, 'start_date': date(2024, 6, 15), 'end_date': date(2025, 6, 14), 'status': 'Active'},
                {'policy_id': 31, 'policy_number': 'POL-2024-031', 'customer_id': 7, 'agent_id': 2, 'policy_type': 'Home', 'premium_amount': 2600.00, 'start_date': date(2024, 7, 1), 'end_date': date(2025, 6, 30), 'status': 'Active'},
                
                {'policy_id': 32, 'policy_number': 'POL-2024-032', 'customer_id': 3, 'agent_id': 3, 'policy_type': 'Umbrella', 'premium_amount': 7200.00, 'start_date': date(2024, 6, 1), 'end_date': date(2025, 5, 31), 'status': 'Active'},
                {'policy_id': 33, 'policy_number': 'POL-2024-033', 'customer_id': 13, 'agent_id': 3, 'policy_type': 'Life', 'premium_amount': 5100.00, 'start_date': date(2024, 6, 15), 'end_date': date(2025, 6, 14), 'status': 'Active'},
                {'policy_id': 34, 'policy_number': 'POL-2024-034', 'customer_id': 8, 'agent_id': 3, 'policy_type': 'Auto', 'premium_amount': 1750.00, 'start_date': date(2024, 7, 1), 'end_date': date(2025, 6, 30), 'status': 'Active'}
            ],
            
            'claims': [
                # APPROVED CLAIMS (ready for payment)
                {'claim_id': 1, 'claim_number': 'CLM-2024-001', 'policy_id': 1, 'claim_date': date(2024, 6, 15), 'claim_amount': 3500.00, 'claim_status': 'Approved', 'description': 'Minor collision repair - windshield and bumper'},
                {'claim_id': 2, 'claim_number': 'CLM-2024-002', 'policy_id': 6, 'claim_date': date(2024, 7, 1), 'claim_amount': 8200.00, 'claim_status': 'Approved', 'description': 'Water damage from burst pipe'},
                {'claim_id': 3, 'claim_number': 'CLM-2024-003', 'policy_id': 11, 'claim_date': date(2024, 7, 15), 'claim_amount': 12500.00, 'claim_status': 'Approved', 'description': 'Theft recovery - total loss vehicle'},
                {'claim_id': 4, 'claim_number': 'CLM-2024-004', 'policy_id': 26, 'claim_date': date(2024, 8, 1), 'claim_amount': 2800.00, 'claim_status': 'Approved', 'description': 'Hail damage to roof'},
                {'claim_id': 5, 'claim_number': 'CLM-2024-005', 'policy_id': 8, 'claim_date': date(2024, 8, 15), 'claim_amount': 4200.00, 'claim_status': 'Approved', 'description': 'Kitchen fire damage'},
                
                # LARGE LOSS EVENTS (> $50,000)
                {'claim_id': 6, 'claim_number': 'CLM-2024-006', 'policy_id': 12, 'claim_date': date(2024, 5, 20), 'claim_amount': 75000.00, 'claim_status': 'Under Review', 'description': 'House fire - total loss'},
                {'claim_id': 7, 'claim_number': 'CLM-2024-007', 'policy_id': 13, 'claim_date': date(2024, 6, 10), 'claim_amount': 85000.00, 'claim_status': 'Approved', 'description': 'Major collision - multiple vehicles'},
                {'claim_id': 8, 'claim_number': 'CLM-2024-008', 'policy_id': 15, 'claim_date': date(2024, 6, 25), 'claim_amount': 120000.00, 'claim_status': 'Under Review', 'description': 'Storm damage - roof and interior'},
                {'claim_id': 9, 'claim_number': 'CLM-2024-009', 'policy_id': 29, 'claim_date': date(2024, 7, 5), 'claim_amount': 95000.00, 'claim_status': 'Approved', 'description': 'Flooding - basement and first floor'},
                {'claim_id': 10, 'claim_number': 'CLM-2024-010', 'policy_id': 32, 'claim_date': date(2024, 7, 20), 'claim_amount': 65000.00, 'claim_status': 'Pending', 'description': 'Vandalism - extensive property damage'},
                
                # PENDING CLAIMS
                {'claim_id': 11, 'claim_number': 'CLM-2024-011', 'policy_id': 2, 'claim_date': date(2024, 8, 1), 'claim_amount': 1500.00, 'claim_status': 'Pending', 'description': 'Windshield replacement'},
                {'claim_id': 12, 'claim_number': 'CLM-2024-012', 'policy_id': 7, 'claim_date': date(2024, 8, 10), 'claim_amount': 3200.00, 'claim_status': 'Pending', 'description': 'Fender bender repair'},
                {'claim_id': 13, 'claim_number': 'CLM-2024-013', 'policy_id': 19, 'claim_date': date(2024, 8, 15), 'claim_amount': 5800.00, 'claim_status': 'Pending', 'description': 'Tree fell on car'},
                {'claim_id': 14, 'claim_number': 'CLM-2024-014', 'policy_id': 20, 'claim_date': date(2024, 8, 20), 'claim_amount': 8500.00, 'claim_status': 'Pending', 'description': 'Bathroom leak damage'},
                {'claim_id': 15, 'claim_number': 'CLM-2024-015', 'policy_id': 4, 'claim_date': date(2024, 8, 25), 'claim_amount': 2100.00, 'claim_status': 'Pending', 'description': 'Door replacement after break-in'},
                
                # UNDER REVIEW CLAIMS
                {'claim_id': 16, 'claim_number': 'CLM-2024-016', 'policy_id': 3, 'claim_date': date(2024, 7, 1), 'claim_amount': 15000.00, 'claim_status': 'Under Review', 'description': 'Suspicious fire - investigation pending'},
                {'claim_id': 17, 'claim_number': 'CLM-2024-017', 'policy_id': 9, 'claim_date': date(2024, 7, 10), 'claim_amount': 22000.00, 'claim_status': 'Under Review', 'description': 'Multiple claims - fraud check'},
                {'claim_id': 18, 'claim_number': 'CLM-2024-018', 'policy_id': 14, 'claim_date': date(2024, 7, 15), 'claim_amount': 18500.00, 'claim_status': 'Under Review', 'description': 'High-value jewelry theft'},
                
                # DENIED CLAIMS
                {'claim_id': 19, 'claim_number': 'CLM-2024-019', 'policy_id': 5, 'claim_date': date(2024, 6, 1), 'claim_amount': 4500.00, 'claim_status': 'Denied', 'description': 'Pre-existing damage not covered'},
                {'claim_id': 20, 'claim_number': 'CLM-2024-020', 'policy_id': 10, 'claim_date': date(2024, 6, 10), 'claim_amount': 6200.00, 'claim_status': 'Denied', 'description': 'Policy lapsed at time of incident'},
                
                # Additional claims
                {'claim_id': 21, 'claim_number': 'CLM-2024-021', 'policy_id': 22, 'claim_date': date(2024, 8, 30), 'claim_amount': 800.00, 'claim_status': 'Approved', 'description': 'Minor scratch repair'},
                {'claim_id': 22, 'claim_number': 'CLM-2024-022', 'policy_id': 23, 'claim_date': date(2024, 9, 1), 'claim_amount': 1200.00, 'claim_status': 'Pending', 'description': 'Window replacement'},
                {'claim_id': 23, 'claim_number': 'CLM-2024-023', 'policy_id': 24, 'claim_date': date(2024, 9, 5), 'claim_amount': 950.00, 'claim_status': 'Approved', 'description': 'Tire replacement'},
                {'claim_id': 24, 'claim_number': 'CLM-2024-024', 'policy_id': 25, 'claim_date': date(2024, 9, 10), 'claim_amount': 1800.00, 'claim_status': 'Pending', 'description': 'Appliance damage from power surge'},
                {'claim_id': 25, 'claim_number': 'CLM-2024-025', 'policy_id': 21, 'claim_date': date(2024, 9, 12), 'claim_amount': 2400.00, 'claim_status': 'Under Review', 'description': 'Carpet replacement after flood'}
            ]
        }
        
        return data
    
    def _insert_table_data(self, table_name: str, data: List[Dict]) -> bool:
        """Insert data into a BigQuery table using batch insert"""
        try:
            full_table_name = f"{self.table_prefix}{table_name}"
            table_ref = self.client.dataset(self.dataset_id).table(full_table_name)
            table = self.client.get_table(table_ref)
            
            # Convert date objects to strings for BigQuery compatibility
            processed_data = []
            for row in data:
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, date):
                        processed_row[key] = value.strftime('%Y-%m-%d')
                    else:
                        processed_row[key] = value
                processed_data.append(processed_row)
            
            # Use native BigQuery insertion instead of pandas
            errors = self.client.insert_rows_json(table, processed_data)
            
            if errors:
                logger.error(f"Failed to insert data into {table_name}: {errors}")
                return False
            
            logger.info(f"Inserted {len(data)} rows into {full_table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {str(e)}")
            return False
    
    def populate_tables(self) -> bool:
        """Populate all tables with sample data"""
        sample_data = self._get_sample_data()
        
        # Insert data in dependency order
        table_order = ['agents', 'customers', 'policies', 'claims']
        
        for table_name in table_order:
            if table_name in sample_data:
                if not self._insert_table_data(table_name, sample_data[table_name]):
                    return False
        
        logger.info("All tables populated successfully")
        return True
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate inserted data and business rules"""
        validation_results = {}
        
        try:
            # Count records in each table
            tables = ['agents', 'customers', 'policies', 'claims']
            for table_name in tables:
                full_table_name = f"{self.dataset_id}.{self.table_prefix}{table_name}"
                query = f"SELECT COUNT(*) as count FROM `{full_table_name}`"
                result = list(self.client.query(query))[0]
                validation_results[f"{table_name}_count"] = result.count
            
            # Premium segments validation
            full_policies_table = f"{self.dataset_id}.{self.table_prefix}policies"
            
            # Budget segment
            query = f"SELECT COUNT(*) as count FROM `{full_policies_table}` WHERE premium_amount <= 2000"
            result = list(self.client.query(query))[0]
            validation_results['budget_policies'] = result.count
            
            # Standard segment
            query = f"SELECT COUNT(*) as count FROM `{full_policies_table}` WHERE premium_amount > 2000 AND premium_amount <= 5000"
            result = list(self.client.query(query))[0]
            validation_results['standard_policies'] = result.count
            
            # Premium segment
            query = f"SELECT COUNT(*) as count FROM `{full_policies_table}` WHERE premium_amount > 5000"
            result = list(self.client.query(query))[0]
            validation_results['premium_policies'] = result.count
            
            # Agent performance validation
            query = f"""
                SELECT 
                    agent_id,
                    COUNT(*) as policy_count,
                    CASE 
                        WHEN COUNT(*) >= 5 THEN 'Top Performer'
                        ELSE 'New Agent'
                    END as performance_tier
                FROM `{full_policies_table}` 
                GROUP BY agent_id 
                ORDER BY policy_count DESC
            """
            agent_results = list(self.client.query(query))
            top_performers = sum(1 for r in agent_results if r.performance_tier == 'Top Performer')
            new_agents = sum(1 for r in agent_results if r.performance_tier == 'New Agent')
            validation_results['top_performers'] = top_performers
            validation_results['new_agents'] = new_agents
            
            # Claims status validation
            full_claims_table = f"{self.dataset_id}.{self.table_prefix}claims"
            query = f"""
                SELECT 
                    claim_status,
                    COUNT(*) as count
                FROM `{full_claims_table}`
                GROUP BY claim_status
                ORDER BY count DESC
            """
            claim_status_results = list(self.client.query(query))
            for result in claim_status_results:
                validation_results[f"claims_{result.claim_status.lower().replace(' ', '_')}"] = result.count
            
            # Large loss events
            query = f"SELECT COUNT(*) as count FROM `{full_claims_table}` WHERE claim_amount > 50000"
            result = list(self.client.query(query))[0]
            validation_results['large_loss_events'] = result.count
            
            # Data quality - complete customer profiles
            full_customers_table = f"{self.dataset_id}.{self.table_prefix}customers"
            query = f"""
                SELECT COUNT(*) as count 
                FROM `{full_customers_table}` 
                WHERE name IS NOT NULL 
                AND email IS NOT NULL 
                AND date_of_birth IS NOT NULL
            """
            result = list(self.client.query(query))[0]
            validation_results['complete_customer_profiles'] = result.count
            
            query = f"SELECT COUNT(*) as count FROM `{full_customers_table}`"
            total_customers = list(self.client.query(query))[0].count
            validation_results['incomplete_customer_profiles'] = total_customers - validation_results['complete_customer_profiles']
            
            logger.info("Data validation completed successfully")
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {"error": str(e)}
    
    def create_insurance_database(self) -> bool:
        """Main method to create the complete insurance database"""
        try:
            logger.info("Starting BigQuery Insurance Database creation...")
            
            # Step 1: Create dataset
            if not self.create_dataset():
                return False
            
            # Step 2: Create tables
            if not self.create_tables():
                return False
            
            # Step 3: Populate with sample data
            if not self.populate_tables():
                return False
            
            # Step 4: Validate data
            validation_results = self.validate_data()
            if "error" in validation_results:
                return False
            
            # Print success summary
            self._print_success_summary(validation_results)
            
            logger.info("BigQuery Insurance Database created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create insurance database: {str(e)}")
            return False
    
    def _print_success_summary(self, validation_results: Dict[str, Any]):
        """Print detailed success summary"""
        print("\n" + "="*80)
        print("🎉 BIGQUERY INSURANCE DATABASE CREATED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n📊 **Database Configuration:**")
        print(f"   • Project ID: {self.project_id}")
        print(f"   • Dataset: {self.dataset_id}")
        print(f"   • Location: {self.location}")
        print(f"   • Table Prefix: {self.table_prefix}")
        print(f"   • Partitioning: {'Enabled' if self.enable_partitioning else 'Disabled'}")
        print(f"   • Clustering: {'Enabled' if self.enable_clustering else 'Disabled'}")
        
        print(f"\n📋 **Tables Created:**")
        print(f"   • {self.table_prefix}agents: {validation_results.get('agents_count', 0)} records")
        print(f"   • {self.table_prefix}customers: {validation_results.get('customers_count', 0)} records") 
        print(f"   • {self.table_prefix}policies: {validation_results.get('policies_count', 0)} records")
        print(f"   • {self.table_prefix}claims: {validation_results.get('claims_count', 0)} records")
        
        print(f"\n🎯 **Business Rules Validation:**")
        print(f"   • Premium Segments:")
        print(f"     - Budget (≤$2K): {validation_results.get('budget_policies', 0)} policies")
        print(f"     - Standard ($2K-$5K): {validation_results.get('standard_policies', 0)} policies")
        print(f"     - Premium (>$5K): {validation_results.get('premium_policies', 0)} policies")
        
        print(f"   • Agent Performance:")
        print(f"     - Top Performers (≥5 policies): {validation_results.get('top_performers', 0)} agents")
        print(f"     - New Agents (<5 policies): {validation_results.get('new_agents', 0)} agents")
        
        print(f"   • Claims Status Distribution:")
        print(f"     - Approved: {validation_results.get('claims_approved', 0)} claims")
        print(f"     - Pending: {validation_results.get('claims_pending', 0)} claims")
        print(f"     - Under Review: {validation_results.get('claims_under_review', 0)} claims")
        print(f"     - Denied: {validation_results.get('claims_denied', 0)} claims")
        
        print(f"   • Special Categories:")
        print(f"     - Large Loss Events (>$50K): {validation_results.get('large_loss_events', 0)} claims")
        print(f"     - Complete Customer Profiles: {validation_results.get('complete_customer_profiles', 0)}")
        print(f"     - Incomplete Customer Profiles: {validation_results.get('incomplete_customer_profiles', 0)}")
        
        print(f"\n✅ **Knowledge Graph Support:**")
        print(f"   • Supports all 10 KG demonstration questions")
        print(f"   • Premium segmentation business rules")
        print(f"   • Agent performance categorization")
        print(f"   • Policy lifecycle management")
        print(f"   • Claims status workflows")
        print(f"   • Data quality analysis")
        
        print(f"\n🔗 **Usage Examples:**")
        print(f"   • Query tables: SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.table_prefix}policies` LIMIT 10")
        print(f"   • Premium analysis: Analyze policies by premium segments")
        print(f"   • Agent performance: Find top performing vs new agents")
        print(f"   • Claims processing: Review approved claims ready for payment")
        
        print("="*80)

def main():
    """Main execution function"""
    try:
        # Create BigQuery database instance
        db = InsuranceBigQueryDB()
        
        # Create the complete insurance database
        success = db.create_insurance_database()
        
        if success:
            print("\n🎊 BigQuery Insurance Database is ready for Knowledge Graph Text-to-SQL demonstrations!")
        else:
            print("\n❌ Failed to create BigQuery Insurance Database")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()