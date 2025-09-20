# GBQN2SQL: Gemini-Powered BigQuery Text-to-SQL System

A production-ready natural language to BigQuery SQL conversion system powered by Google Gemini AI, featuring knowledge graph enhancement and automated setup.

## 🎯 Overview

GBQN2SQL transforms natural language questions into optimized BigQuery SQL queries using Google's Gemini AI models. The system includes two powerful components:

- **GeminiText2SQL**: Direct text-to-SQL conversion with schema awareness
- **GeminiKGText2SQL**: Enhanced conversion using BigQuery Knowledge Graph for superior context understanding

## ✨ Key Features

- 🤖 **Google Gemini Intelligence**: Advanced natural language understanding
- 📊 **BigQuery Optimization**: Cloud-native query generation with best practices
- 🔗 **Knowledge Graph Enhancement**: Semantic relationships for complex queries
- 🛡️ **Schema Validation**: Prevents hallucinated tables and columns
- 🚀 **Production Ready**: Comprehensive error handling and monitoring
- 🎮 **Interactive Web UI**: Streamlit-based interface for easy testing
- ⚙️ **Automated Setup**: One-command deployment with validation

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Natural        │    │  Gemini AI       │    │  BigQuery       │
│  Language       │───▶│  Text-to-SQL     │───▶│  Execution      │
│  Questions      │    │  Engine          │    │  Engine         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  BigQuery        │
                       │  Knowledge Graph │
                       │  (Semantic       │
                       │   Enhancement)   │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- UV Package Manager
- Google Cloud CLI
- Git
- Active Google Cloud Project with BigQuery API enabled

### Automated Setup

Run the automated setup script to configure everything:

```bash
python3 setup.py
```

The setup script will:
1. ✅ Verify all prerequisites
2. ✅ Validate environment configuration
3. ✅ Test GCP access (BigQuery, Vertex AI)
4. ✅ Create sample insurance dataset
5. ✅ Generate BigQuery knowledge graph
6. ✅ Test all system components
7. ✅ Launch Streamlit web interface

### Manual Setup

If you prefer manual setup:

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your GCP settings

# 3. Create BigQuery dataset
uv run python src/bq/InsuranceBigQueryDB.py

# 4. Generate knowledge graph
uv run python src/bq/BQKnowledgeGraphGenerator.py

# 5. Test components
uv run python src/bq/GeminiText2SQL.py --test
uv run python src/bq/GeminiKGText2SQL.py --test

# 6. Launch web interface
uv run streamlit run src/bq/GBQ_Streamlit_App.py
```

## 🔧 Configuration

Create a `.env` file with your Google Cloud settings:

```env
# Google Cloud Configuration
GCP_PROJECT_ID=your-project-id
GOOGLE_CLOUD_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
BQ_DATASET_ID=insurance_analytics
BQ_LOCATION=US
VERTEX_AI_LOCATION=us-central1
GEMINI_MODEL_NAME=gemini-2.5-flash-lite
```

## 🎮 Usage

### Command Line Interface

**Basic Text-to-SQL:**
```bash
uv run python src/bq/GeminiText2SQL.py
```

**Knowledge Graph Enhanced:**
```bash
uv run python src/bq/GeminiKGText2SQL.py
```

**Test Mode:**
```bash
uv run python src/bq/GeminiText2SQL.py --test
uv run python src/bq/GeminiKGText2SQL.py --test
```

### Web Interface

Launch the Streamlit web application:
```bash
uv run streamlit run src/bq/GBQ_Streamlit_App.py
```

Access at: http://localhost:8501

### Example Queries

Try these natural language questions:

- "Show me all customers with their email addresses"
- "Find customers with multiple policies"
- "What's the average premium for life insurance?"
- "Which agent has the best claims-to-premium ratio?"
- "Find high-premium customers with low claims"
- "Show me all active policies that are currently in force"
- "List policies that are expiring soon"

## 📊 Sample Data

The system includes a comprehensive insurance database with:

- **Agents**: 32 insurance agents with contact information
- **Customers**: 60 customers with demographics
- **Policies**: 102 insurance policies (Auto, Home, Life, Renters)
- **Claims**: 75 claims with various statuses and amounts

## 🧠 Knowledge Graph Features

The BigQuery Knowledge Graph provides:

- **Business Rules**: Premium segments, policy lifecycle, risk assessment
- **Join Patterns**: Optimized relationship mappings
- **Data Validation**: Type checking and constraint enforcement
- **Query Optimization**: Performance-aware SQL generation

## 🔍 Testing & Validation

### Automated Testing

Both components include comprehensive test suites:

```bash
# Run all tests
python3 setup.py

# Individual component tests
uv run python src/bq/GeminiText2SQL.py --test
uv run python src/bq/GeminiKGText2SQL.py --test
```

### Test Coverage

- ✅ Schema loading and validation
- ✅ Natural language processing
- ✅ SQL generation accuracy
- ✅ BigQuery compatibility
- ✅ Error handling and recovery
- ✅ Performance benchmarks

## 📁 Project Structure

```
gbqn2sql/
├── src/
│   ├── bq/                           # BigQuery components
│   │   ├── GeminiText2SQL.py         # Basic text-to-SQL
│   │   ├── GeminiKGText2SQL.py       # Knowledge graph enhanced
│   │   ├── BQKnowledgeGraphGenerator.py  # KG generation
│   │   ├── InsuranceBigQueryDB.py    # Sample data creation
│   │   └── GBQ_Streamlit_App.py      # Web interface
│   └── __init__.py
├── bqkg/                             # Knowledge graph files
│   ├── bq_knowledge_graph.ttl        # Turtle format
│   ├── bq_knowledge_graph.rdf        # RDF/XML format
│   ├── bq_knowledge_graph.nt         # N-Triples format
│   └── bq_knowledge_graph.n3         # N3 format
├── tests/                            # Test suite
├── setup.py                          # Automated setup script
├── pyproject.toml                    # Project configuration
├── requirements.txt                  # Dependencies
├── .env                              # Environment configuration
└── README.md                         # This file
```

## 🛠️ Development

### Prerequisites for Development

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repository-url>
cd gbqn2sql

# Install dependencies
uv sync
```

### Running Tests

```bash
# Run setup validation
python3 setup.py --dry-run

# Test individual components
uv run python src/bq/GeminiText2SQL.py --test
uv run python src/bq/GeminiKGText2SQL.py --test

# Run Python tests
uv run pytest tests/
```

### Code Structure

- **Text-to-SQL Engine**: Core natural language processing
- **Knowledge Graph**: Semantic enhancement layer
- **BigQuery Interface**: Cloud database integration
- **Web Interface**: Streamlit-based UI
- **Setup System**: Automated configuration and deployment

## 🔐 Security & Best Practices

- **Service Account Authentication**: Secure GCP access with impersonation
- **Environment Variables**: Sensitive data isolation
- **SQL Injection Protection**: Parameterized query validation
- **Access Controls**: Role-based BigQuery permissions
- **Audit Logging**: Comprehensive operation tracking

## 📈 Performance

### Benchmarks

- **Query Generation**: ~2-3 seconds average
- **SQL Validation**: ~100ms average
- **BigQuery Execution**: Varies by query complexity
- **Knowledge Graph Loading**: ~1 second initial load

### Optimization Features

- **Schema Caching**: Reduced BigQuery API calls
- **Query Limits**: Automatic LIMIT clauses for exploration
- **Parallel Processing**: Batch query support
- **Error Recovery**: Graceful failure handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**Authentication Errors:**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**BigQuery Permission Issues:**
- Ensure your service account has BigQuery Admin role
- Verify dataset location matches configuration

**Streamlit Port Conflicts:**
```bash
uv run streamlit run src/bq/GBQ_Streamlit_App.py --server.port 8502
```

### Getting Help

- Check the setup log: `tail -f setup.log`
- Review component tests: `--test` mode
- Verify environment: `.env` file configuration

## 🎉 Acknowledgments

- Google Cloud BigQuery team for excellent documentation
- Google AI team for Gemini API access
- Streamlit team for the web framework
- RDFLib community for knowledge graph support

---

**Built with ❤️ using Google Gemini AI and BigQuery**