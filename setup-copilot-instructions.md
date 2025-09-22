# GitHub Copilot Setup Instructions for BigQuery Text-to-SQL Project

## Copilot Prompt: First-Time Project Setup (Part 1)

**Instruction to GitHub Copilot:**

You are helping a developer set up the BigQuery Text-to-SQL project in VS Code for the first time after checking out the repository. This project uses Google Cloud BigQuery, Vertex AI Gemini, Python 3.11+, UV package manager, and Streamlit. Guide the user through pre-requisites installation and environment configuration step-by-step.

---

## Pre-requisites Setup

**Copilot, help me verify and install the following pre-requisites:**

### 1. System Requirements Check
- Verify Python 3.11+ is installed: `python --version`
- Check available RAM (minimum 8GB required)
- Verify at least 2GB free disk space

### 2. Google Cloud SDK Installation
**If gcloud is not installed, guide me through installation:**
- **macOS**: `brew install google-cloud-sdk`
- **Linux**: `curl https://sdk.cloud.google.com | bash && exec -l $SHELL`
- **Windows**: Download from Google Cloud SDK documentation
- **Verify**: `gcloud --version`

### 3. UV Package Manager Installation
**If UV is not installed, help me install it:**
- **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Verify**: `uv --version`

### 4. VS Code Extensions Installation
**Help me install these essential extensions:**

**Required:**
- Python (`ms-python.python`)
- Pylance (`ms-python.vscode-pylance`) 
- Python Debugger (`ms-python.debugpy`)

**Recommended:**
- Thunder Client (`rangav.vscode-thunder-client`)
- GitLens (`eamodio.gitlens`)
- Material Icon Theme (`pkief.material-icon-theme`)
- Streamlit (`rderik.vscode-streamlit`)

---

## Environment Configuration

**Copilot, walk me through setting up the development environment:**

### Step 1: Repository Setup
```bash
# Clone repository (if not already done)
git clone https://github.com/jaig1/gbqn2sql.git
cd gbqn2sql

# Open in VS Code
code .
```

### Step 2: Environment Configuration File
**Create a `.env` file in the project root with this template:**

```properties
# Google Cloud Configuration
# ========================

# Your Google Cloud Project ID (REPLACE WITH ACTUAL PROJECT ID)
GCP_PROJECT_ID=ford-4b725ca30ab26163fb141019

# Service Account for BigQuery access (REPLACE WITH ACTUAL SA EMAIL)
SERVICE_ACCOUNT_EMAIL=sa-developer@ford-4b725ca30ab26163fb141019.iam.gserviceaccount.com

# BigQuery Dataset Configuration
BQ_DATASET_ID=insurance_analytics
BQ_LOCATION=US

# Vertex AI Configuration
VERTEX_AI_LOCATION=us-central1
GEMINI_MODEL_NAME=gemini-2.5-flash-lite

# Optional: Streamlit Configuration
STREAMLIT_PORT=8501
```

**Copilot, remind me to:**
- Use the provided Ford project ID: `ford-4b725ca30ab26163fb141019`
- Verify this is the correct project for my environment
- Keep this file secure and never commit it to git

### Step 3: Python Environment Setup
**Guide me through UV-based dependency installation:**

```bash
# Create virtual environment using UV
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install all dependencies
uv pip install -r requirements.txt
```

**Alternative single command:**
```bash
uv sync
```

### Step 4: VS Code Python Interpreter Configuration
**Help me configure VS Code to use the correct Python interpreter:**

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type: `Python: Select Interpreter`
3. Select: `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows)

### Step 5: Environment Validation
**Run these validation commands and help me troubleshoot any issues:**

```bash
# Verify Python setup
python --version
uv pip list

# Test Google Cloud SDK
gcloud --version

# Test critical Python packages
python -c "import google.cloud.bigquery; print('✅ BigQuery client available')"
python -c "import vertexai; print('✅ Vertex AI available')"
python -c "import streamlit; print('✅ Streamlit available')"
python -c "import rdflib; print('✅ RDF library available')"
```

---

## Expected Project Structure

**Copilot, verify my project structure matches this layout:**

```
gbqn2sql/
├── .env                    # Environment config (I created this)
├── .venv/                  # Virtual environment
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
├── SETUP_GUIDE_PART1.md   # This setup guide
├── src/
│   ├── __init__.py
│   ├── usersetup.py       # Authentication setup script
│   └── bq/                # BigQuery modules
│       ├── BQKnowledgeGraphGenerator.py
│       ├── GeminiKGText2SQL.py
│       ├── GBQ_Streamlit_App.py
│       └── InsuranceBigQueryDB.py
└── bqkg/                  # Knowledge graph files (created later)
```

---

## Troubleshooting Assistance

**Copilot, if I encounter these issues, help me resolve them:**

### Common Problems:
- **"Python not found"**: Check PATH, verify Python 3.11+ installation
- **"UV command not found"**: Restart terminal, check UV installation
- **"Permission denied"**: Use appropriate permissions for file operations
- **"Module not found"**: Verify virtual environment activation and package installation
- **"VS Code interpreter issues"**: Reload window, manually select interpreter

### Validation Failures:
- **Google Cloud packages fail**: Check internet connection, verify UV installation
- **Import errors**: Ensure all dependencies installed correctly
- **Path issues**: Verify working directory and file locations

---

## Success Criteria

**Copilot, confirm setup is complete when:**

- ✅ All pre-requisites installed and verified
- ✅ Repository cloned and opened in VS Code
- ✅ `.env` file created with project-specific values
- ✅ Python virtual environment created and activated
- ✅ All dependencies installed via UV
- ✅ VS Code Python interpreter configured correctly
- ✅ All validation commands pass successfully
- ✅ Project structure verified

---

## Next Steps

**After Part 1 completion, Copilot should guide me through:**

1. **Authentication Setup** - Running `uv run python src/usersetup.py`
2. **Database Creation** - BigQuery dataset and tables setup
3. **Knowledge Graph Generation** - RDF graph creation from schema
4. **System Testing** - Text-to-SQL functionality validation
5. **UI Launch** - Streamlit application startup

**Copilot, remind me that Part 1 focuses only on environment setup. Authentication and system initialization will be covered in subsequent steps.**

---

## Context for Copilot

**Project Information:**
- **Technology Stack**: Python 3.11+, Google Cloud BigQuery, Vertex AI Gemini, UV, Streamlit, RDF/SPARQL
- **Purpose**: Knowledge graph enhanced text-to-SQL conversion for insurance analytics
- **Authentication**: Service account impersonation with Google Cloud
- **Development**: VS Code with Python extensions
- **Package Management**: UV for fast dependency resolution

**Copilot Behavior:**
- Provide step-by-step guidance
- Offer troubleshooting for common issues
- Verify each step before proceeding
- Explain the purpose of each configuration
- Warn about security considerations (`.env` file handling)
- Suggest best practices for development workflow
