#!/usr/bin/env python3
"""
📊 Ontology Views Module for BigQuery Knowledge Graph Streamlit App
=================================================================

This module provides ontology visualization functionality including:
- Business Semantic Ontology view
- Query Pattern Ontology view  
- Synonym & Natural Language Ontology view

Each view includes interactive expandable sections and SVG downloads.

Requirements:
- streamlit
- SVG files in ../../svg/ directory
"""

import streamlit as st
import os

def render_business_semantic_view():
    """Render the Business Semantic Ontology slice."""
    
    # Description
    st.markdown("### Overview")
    st.write("""
    This slice demonstrates the critical integration between the database structure layer (SQL namespace) 
    and the business semantics layer (Business namespace). It shows how natural language queries are 
    transformed into SQL by mapping business terms to database entities through synonym mappings, 
    business rules, and semantic relationships.
    """)
    
    # SVG Content
    svg_path = os.path.join(os.path.dirname(__file__), '../../svg/sql_business_integration.svg')
    try:
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        # Display SVG
        st.components.v1.html(
            svg_content, 
            height=950,
            scrolling=True
        )
        
        # Interactive expandable sections for Business Semantic view
        render_business_semantic_expandable_sections()
        
        # Download option
        st.download_button(
            label="⬇️ Download Business Semantic SVG",
            data=svg_content,
            file_name="sql_business_integration.svg",
            mime="image/svg+xml",
            help="Download this diagram as an SVG file"
        )
        
    except FileNotFoundError:
        st.error("❌ SVG file not found: sql_business_integration.svg")
        st.info("Please ensure the file exists in the svg/ directory")

def render_query_pattern_view():
    """Render the Query Pattern Ontology slice."""
    
    # Description
    st.markdown("### Overview")
    st.write("""
    This slice focuses on reusable query knowledge captured in the ontology. It shows how query patterns,
    join patterns, and query guidelines work together to provide templates for common database operations.
    This knowledge base enables intelligent query generation and optimization by codifying institutional
    database knowledge and best practices.
    """)
    
    # SVG Content
    svg_path = os.path.join(os.path.dirname(__file__), '../../svg/query_patterns_ontology.svg')
    try:
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        # Display SVG
        st.components.v1.html(
            svg_content, 
            height=950,
            scrolling=True
        )
        
        # Interactive expandable sections for Query Pattern view
        render_query_pattern_expandable_sections()
        
        # Download option
        st.download_button(
            label="⬇️ Download Query Pattern SVG",
            data=svg_content,
            file_name="query_patterns_ontology.svg",
            mime="image/svg+xml",
            help="Download this diagram as an SVG file"
        )
        
    except FileNotFoundError:
        st.error("❌ SVG file not found: query_patterns_ontology.svg")
        st.info("Please ensure the file exists in the svg/ directory")

def render_synonym_natural_language_view():
    """Render the Synonym & Natural Language Ontology slice."""
    
    # Description
    st.markdown("### Overview")
    st.write("""
    This slice focuses on the critical synonym and natural language mapping components that enable
    text-to-SQL translation. It shows how natural language terms are mapped to database entities
    through TableSynonyms, ColumnSynonyms, and Alias classes. This mapping layer is essential
    for understanding user queries in natural language and translating them to precise SQL.
    """)
    
    # SVG Content
    svg_path = os.path.join(os.path.dirname(__file__), '../../svg/synonym_natural_language_ontology.svg')
    try:
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        # Display SVG
        st.components.v1.html(
            svg_content, 
            height=1000,
            scrolling=True
        )
        
        # Interactive expandable sections for Synonym & Natural Language view
        render_synonym_natural_language_expandable_sections()
        
        # Download option
        st.download_button(
            label="⬇️ Download Synonym & NL SVG",
            data=svg_content,
            file_name="synonym_natural_language_ontology.svg",
            mime="image/svg+xml",
            help="Download this diagram as an SVG file"
        )
        
    except FileNotFoundError:
        st.error("❌ SVG file not found: synonym_natural_language_ontology.svg")
        st.info("Please ensure the file exists in the svg/ directory")

def render_business_semantic_expandable_sections():
    """Render expandable sections for Business Semantic Ontology view."""
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🏗️ SQL Structure Layer"):
            st.markdown("""
            **Core Database Classes:**
            - **Database** (`sql:Database`): BigQuery project container
            - **Schema** (`sql:Schema`): Dataset grouping related tables  
            - **Table** (`sql:Table`): Physical data tables (customers, policies, etc.)
            - **Column** (`sql:Column`): Individual fields with types and constraints
            - **ForeignKey** (`sql:ForeignKey`): Referential integrity relationships
            
            **Purpose**: Provides the physical structure and organization of data
            """)
            
        with st.expander("🧠 Business Semantics Layer"):
            st.markdown("""
            **Business Intelligence Classes:**
            - **Rule** (`business:Rule`): Business logic and segmentation rules
            - **CompositeRule** (`business:CompositeRule`): Complex multi-rule logic
            - **Concept** (`business:Concept`): Domain-specific business concepts
            - **TableSynonyms** (`business:TableSynonyms`): Alternative table names
            - **Alias** (`business:Alias`): Natural language terms for entities
            
            **Purpose**: Bridges natural language to database terminology
            """)

    with col2:
        with st.expander("🔗 Integration Relationships"):
            st.markdown("""
            **Key Integration Links:**
            - **appliesTo**: Business rules target specific tables
            - **mapsToTable**: Synonyms connect to physical tables
            - **hasAlias**: Multiple natural language terms per entity
            - **describes**: Concepts provide semantic context
            - **aliasFor**: Direct term-to-entity mappings
            
            **Data Flow**: Natural Language → Business Layer → SQL Layer
            """)
            
        with st.expander("⚡ Translation Process"):
            st.markdown("""
            **Example Query Translation:**
            
            **Input**: *"Show me all clients with high premiums"*
            
            **Step 1 - Term Resolution**:
            - "clients" → `TableSynonyms` → `insurance_customers`
            - "high premiums" → `business:Rule` → `premium > 5000`
            
            **Step 2 - SQL Generation**:
            ```sql
            SELECT * FROM insurance_customers 
            WHERE premium > 5000
            ```
            
            **Result**: Natural language successfully converted to executable SQL
            """)

    with st.expander("💡 Real-World Examples"):
        st.markdown("""
        **Example 1: Customer Segmentation**
        - Natural Language: "Find high-value customers"
        - Business Rule: `premium > 5000` (High Value Customer rule)
        - SQL Target: `insurance_customers` table
        
        **Example 2: Synonym Mapping**
        - User says: "clients", "policyholders", "insureds"
        - All map to: `insurance_customers` table
        - 12 different aliases supported per table
        
        **Example 3: Complex Business Logic**
        - Composite Rule: "High Risk Customer" 
        - Combines: Multiple claims + High claim amounts + Recent timeframe
        - Generates: Complex subquery with JOINs and aggregations
        """)
    
    with st.expander("📊 Integration Statistics"):
        st.markdown("""
        **Current Semantic Mappings:**
        - **112 Aliases**: Natural language terms for database entities
        - **22 Business Rules**: Logic for customer/policy segmentation  
        - **4 Table Synonyms**: Alternative names for main insurance tables
        - **19 Column Synonyms**: Alternative names for key fields
        - **6 Business Concepts**: Domain-specific insurance concepts
        - **3 Composite Rules**: Complex multi-condition business logic
        
        **Coverage**: 100% of insurance tables have semantic mappings
        """)

def render_query_pattern_expandable_sections():
    """Render expandable sections for Query Pattern Ontology view."""
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🎯 Query Pattern Classes"):
            st.markdown("""
            **Core Pattern Classes:**
            - **QueryGuidelines** (`sql:QueryGuidelines`): Overall query guidance for domain
            - **QueryPattern** (`sql:QueryPattern`): Reusable query templates
            - **JoinPattern** (`sql:JoinPattern`): Standardized join operations
            - **Table** (`sql:Table`): Tables involved in patterns
            
            **Purpose**: Codifies reusable database query knowledge
            """)
            
        with st.expander("🔧 Pattern Properties"):
            st.markdown("""
            **Structural Properties:**
            - **joinType**: INNER, LEFT, RIGHT, FULL, CROSS
            - **directionality**: one-to-one, one-to-many, many-to-many
            - **sqlPattern**: Complete executable SQL template
            
            **Semantic Properties:**
            - **purpose**: Business rationale for the pattern
            - **usage**: When and how to apply the pattern
            - **useCase**: Business scenario description
            """)
    
    with col2:
        with st.expander("🚀 Pattern Benefits"):
            st.markdown("""
            **Text-to-SQL Translation:**
            - Map natural language to query templates
            - Pre-validated join patterns reduce errors
            - Suggest relevant patterns to users
            
            **Knowledge Management:**
            - Codify institutional query knowledge
            - Self-documenting query patterns
            - Training data for ML-based query generation
            """)
            
        with st.expander("📋 Example Patterns"):
            st.markdown("""
            **Customer→Policy Join:**
            - Links customers to their insurance policies
            - One-to-many relationship
            - Used for customer demographics + policy analysis
            
            **Claims Analysis Pattern:**
            - Combines customer, policy, and claim data
            - Multi-table join with filtering
            - Complexity: Medium
            """)
    
    with st.expander("📈 Pattern Statistics"):
        st.markdown("""
        **Current Query Knowledge:**
        - **3 Query Patterns**: Validated template queries for common use cases
        - **6 Join Patterns**: Standardized table relationship patterns
        - **1 Query Guidelines**: Domain-specific query guidance
        - **Coverage**: All major insurance table relationships documented
        
        **Pattern Complexity**: Simple to Medium complexity patterns available
        """)

def render_synonym_natural_language_expandable_sections():
    """Render expandable sections for Synonym & Natural Language Ontology view."""
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🏷️ Synonym Classes"):
            st.markdown("""
            **Core Synonym Classes:**
            - **TableSynonyms** (`business:TableSynonyms`): Alternative names for database tables
            - **ColumnSynonyms** (`business:ColumnSynonyms`): Alternative names for table columns
            - **Alias** (`business:Alias`): Individual natural language terms with context
            
            **Purpose**: Bridge natural language terminology to database schema
            """)
            
        with st.expander("🔤 Natural Language Terms"):
            st.markdown("""
            **Table Aliases (12 for insurance_policies):**
            - "policies", "contracts", "coverage"
            - "plans", "agreements", "protection"
            - "insurance_contracts", "policyholders"
            
            **Column Aliases (6 for premium):**
            - "premium_amount", "cost", "price"
            - "payment", "rate", "amount"
            """)

    with col2:
        with st.expander("🔄 Translation Process"):
            st.markdown("""
            **4-Step Translation:**
            1. **User Query**: Natural language input
            2. **Synonym Lookup**: Map terms to database entities
            3. **SQL Generation**: Construct executable query
            4. **Execution & Results**: Return formatted results
            
            **Example**: "contracts" → insurance_policies
            """)
            
        with st.expander("🎯 Use Cases"):
            st.markdown("""
            **Business Scenarios:**
            - Customer service queries using common terms
            - Executive dashboards with business language
            - Self-service analytics for non-technical users
            - Voice interfaces and chatbots
            
            **Translation Accuracy**: Context-aware mapping
            """)
    
    with st.expander("📈 Synonym Statistics"):
        st.markdown("""
        **Current Terminology Coverage:**
        - **112 Aliases**: Total natural language terms available
        - **4 Table Synonyms**: Alternative names for main tables
        - **19 Column Synonyms**: Alternative names for key fields
        - **Multiple Contexts**: Financial, legal, colloquial terms supported
        
        **Coverage**: Comprehensive mapping for insurance domain terminology
        """)
    
    with st.expander("🔧 Technical Features"):
        st.markdown("""
        **Advanced Capabilities:**
        - **Context Awareness**: Terms understood in business context
        - **Confidence Scoring**: Reliability metrics for each mapping
        - **Multi-term Support**: Handle phrases and compound terms
        - **Bidirectional Mapping**: Natural language ↔ SQL entities
        
        **Integration**: Seamless connection to query generation pipeline
        """)

def render_ontology_views_page():
    """Main function to render the Ontology Views page with radio button selection."""
    
    st.title("📊 Ontology Visualization Dashboard")
    
    st.markdown("""
    Explore different aspects of the BigQuery Knowledge Graph ontology through interactive visualizations.
    Each view provides detailed insights into how the system maps natural language to SQL queries.
    """)
    
    # Radio button selection
    view_option = st.radio(
        "Select Ontology View:",
        ("Business Semantic Ontology", "Query Pattern Ontology", "Synonym & Natural Language Ontology"),
        horizontal=True
    )
    
    # Display selected view
    if view_option == "Business Semantic Ontology":
        render_business_semantic_view()
    elif view_option == "Query Pattern Ontology":
        render_query_pattern_view()
    elif view_option == "Synonym & Natural Language Ontology":
        render_synonym_natural_language_view()