# GBQN2SQL Setup Script

## Overview

The `setup.py` script provides a comprehensive, automated setup process for the BigQuery Knowledge Graph-Enhanced Natural Language to SQL System (GBQN2SQL).

## Features

- **Interactive Setup**: Guided setup with user confirmations at each step
- **Comprehensive Validation**: Verifies all prerequisites and access permissions
- **Error Handling**: Robust error handling with detailed logging
- **Dry Run Mode**: Preview changes without executing them
- **Backup & Safety**: Automatically backs up existing configuration files

## Prerequisites

Before running the setup script, ensure you have:

1. **Python 3.8+** - Required for the application
2. **UV Package Manager** - Modern Python dependency management
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Google Cloud CLI** - For BigQuery and Vertex AI access
   ```bash
   # Install from: https://cloud.google.com/sdk/docs/install
   gcloud auth login
   gcloud config set project gen-lang-client-0454606702
   ```
4. **Git** - For version control (usually pre-installed)

## Usage

### Basic Setup
```bash
python setup.py
```

### Advanced Options
```bash
# Preview what will be done without executing
python setup.py --dry-run

# Skip user confirmations (automated mode)
python setup.py --skip-confirmation

# Enable verbose logging
python setup.py --verbose

# Combine options
python setup.py --dry-run --verbose
```

## Setup Process

The script performs the following steps:

### 1. Prerequisites Check
- Verifies Python, UV, Google Cloud CLI, and Git installation
- Ensures all required tools are in PATH

### 2. Environment Setup
- Creates `.env` file with project configuration
- Backs up existing `.env` file if present
- Configures Google Cloud project settings

### 3. GCP Access Verification
- Tests Google Cloud project access
- Verifies BigQuery API access and permissions
- Confirms Vertex AI endpoint availability

### 4. BigQuery Dataset Creation
- Executes `src/bq/InsuranceBigQueryDB.py`
- Creates `insurance_data` dataset
- Populates with 4 tables: agents, customers, policies, claims
- Verifies table creation and data population

### 5. Knowledge Graph Generation
- Executes `src/bq/BQKnowledgeGraphGenerator.py`
- Generates knowledge graph files in multiple formats (.ttl, .rdf, .nt, .n3)
- Verifies file creation and content

### 6. Component Testing
- Tests `src/bq/GeminiText2SQL.py` (basic text-to-SQL)
- Tests `src/bq/GeminiKGText2SQL.py` (knowledge graph-enhanced)
- Validates successful execution of both components

### 7. Streamlit App Launch
- Launches `src/bq/GBQ_Streamlit_App.py`
- Makes app available at http://localhost:8501
- Provides access instructions

## Environment Configuration

The setup script creates a `.env` file with the following variables:

```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=gen-lang-client-0454606702
SERVICE_ACCOUNT_EMAIL=bigquery-text2sql@gen-lang-client-0454606702.iam.gserviceaccount.com

# BigQuery Configuration
BQ_DATASET_ID=insurance_data
BQ_LOCATION=US

# Vertex AI Configuration
VERTEX_AI_PROJECT=gen-lang-client-0454606702
VERTEX_AI_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash-lite

# Application Configuration
STREAMLIT_PORT=8501
DEBUG_MODE=False
```

## Troubleshooting

### Common Issues

1. **Google Cloud Authentication**
   ```bash
   gcloud auth login
   gcloud auth application-default login --impersonate-service-account=bigquery-text2sql@gen-lang-client-0454606702.iam.gserviceaccount.com
   ```

2. **UV Package Manager Not Found**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc  # or ~/.zshrc
   ```

3. **Permission Errors**
   - Ensure your Google Cloud account has proper permissions
   - Verify service account has BigQuery and Vertex AI access

4. **Port Already in Use**
   - Change `STREAMLIT_PORT` in `.env` file
   - Or stop existing Streamlit processes: `pkill -f streamlit`

### Logs

The setup script creates detailed logs in `setup.log`. Check this file for:
- Detailed error messages
- Command execution logs
- Step-by-step progress

### Manual Recovery

If setup fails partway through, you can:

1. **Check the log**: `tail -f setup.log`
2. **Run specific components manually**:
   ```bash
   uv run python src/bq/InsuranceBigQueryDB.py
   uv run python src/bq/BQKnowledgeGraphGenerator.py
   uv run python src/bq/GeminiKGText2SQL.py
   uv run streamlit run src/bq/GBQ_Streamlit_App.py
   ```
3. **Re-run setup**: The script is idempotent and can be safely re-run

## Post-Setup

After successful setup:

1. **Access the Streamlit app**: http://localhost:8501
2. **Test queries**: Try natural language questions about insurance data
3. **Explore components**:
   - `src/bq/GeminiText2SQL.py` - Basic text-to-SQL
   - `src/bq/GeminiKGText2SQL.py` - Knowledge graph-enhanced
   - `bqkg/` - Knowledge graph files

## Support

For issues or questions:
1. Check `setup.log` for detailed error information
2. Verify all prerequisites are installed
3. Ensure Google Cloud permissions are properly configured
4. Try running individual components manually to isolate issues

## Script Details

- **Language**: Python 3.8+
- **Dependencies**: Standard library only (no external deps for setup script)
- **Size**: ~400 lines with comprehensive error handling
- **Features**: Colored output, progress tracking, dry-run mode, logging