"""
BigQuery Knowledge Graph Context Builder

Extracts schema context from BigQuery Knowledge Graph for LLM text-to-SQL generation.
Adapted from SQLKnowledgeGraphContextBuilder to work with BigQuery-specific
knowledge graphs and metadata.
"""

from rdflib import Graph, Namespace
import re
import json
import os

class BQKnowledgeGraphContextBuilder:
    """
    Extracts schema context from BigQuery Knowledge Graph for LLM text-to-SQL generation
    """
    
    def __init__(self, bq_kg_file):
        self.graph = Graph()
        self.graph.parse(bq_kg_file, format="turtle")
        
        # Define namespaces for BigQuery Knowledge Graph
        self.BUSINESS = Namespace("http://business.org/schema#")
        self.DATA = Namespace("http://data.org/schema#")
        self.SQL = Namespace("http://sql.org/schema#")
        self.STATS = Namespace("http://statistics.org/schema#")
        self.VALIDATION = Namespace("http://validation.org/schema#")
        self.INSURANCE = Namespace("http://insurance.org/schema#")
        
        # Cache for extracted schema information
        self.schema_context = None
        self.project_id = None
        self.dataset_id = None
        
        # Extract project and dataset info from first parsed triple
        self._extract_project_dataset_info()
    
    def _extract_project_dataset_info(self):
        """Extract project and dataset info from knowledge graph URIs"""
        # Find first database/project URI
        for subj, pred, obj in self.graph:
            if "projects/" in str(subj) and "datasets/" in str(subj):
                uri_parts = str(subj).split("/")
                try:
                    projects_idx = uri_parts.index("projects")
                    datasets_idx = uri_parts.index("datasets")
                    if projects_idx + 1 < len(uri_parts):
                        self.project_id = uri_parts[projects_idx + 1]
                    if datasets_idx + 1 < len(uri_parts):
                        self.dataset_id = uri_parts[datasets_idx + 1]
                    break
                except (ValueError, IndexError):
                    continue
    
    def extract_schema_context(self):
        """Extract comprehensive schema context for LLM"""
        if self.schema_context:
            return self.schema_context
        
        context = {
            "database_info": self._get_database_info(),
            "tables": self._get_table_info(),
            "relationships": self._get_relationships(),
            "data_types": self._get_data_type_info(),
            "constraints": self._get_constraint_info()
        }
        
        self.schema_context = context
        return context
    
    def _get_database_info(self):
        """Get basic database/project information"""
        # Query for BigQuery project/dataset information
        query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        
        SELECT ?projectId ?datasetId ?location ?platform WHERE {
            ?db a sql:Database ;
                dcterms:identifier ?projectId ;
                business:platform ?platform .
            ?schema a sql:Schema ;
                    dcterms:identifier ?datasetId .
            OPTIONAL { ?schema data:location ?location }
        }
        """
        
        results = list(self.graph.query(query))
        if results:
            row = results[0]
            return {
                "name": f"{str(row[0])}.{str(row[1])}" if row[0] and row[1] else "BigQuery Database",
                "engine": "BigQuery",
                "platform": str(row[3]) if row[3] else "Google BigQuery",
                "project_id": str(row[0]) if row[0] else self.project_id,
                "dataset_id": str(row[1]) if row[1] else self.dataset_id,
                "location": str(row[2]) if row[2] else "US"
            }
        
        # Fallback if query doesn't work
        return {
            "name": f"{self.project_id}.{self.dataset_id}" if self.project_id and self.dataset_id else "BigQuery Database",
            "engine": "BigQuery",
            "platform": "Google BigQuery",
            "project_id": self.project_id,
            "dataset_id": self.dataset_id
        }
    
    def _get_table_info(self):
        """Get detailed table and column information from BigQuery knowledge graph"""
        # Get tables with business metadata
        tables_query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX stats: <http://statistics.org/schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX insurance: <http://insurance.org/schema#>
        
        SELECT ?tableLabel ?rowCount ?description ?businessDescription ?businessPurpose 
               ?domainCategory ?qualityScore ?sizeBytes WHERE {
            ?table a sql:Table ;
                   rdfs:label ?tableLabel .
            OPTIONAL { ?table stats:rowCount ?rowCount }
            OPTIONAL { ?table dcterms:description ?description }
            OPTIONAL { ?table business:description ?businessDescription }
            OPTIONAL { ?table business:purpose ?businessPurpose }
            OPTIONAL { ?table business:lifecycleStage ?domainCategory }
            OPTIONAL { ?table stats:qualityScore ?qualityScore }
            OPTIONAL { ?table stats:sizeBytes ?sizeBytes }
        }
        ORDER BY ?tableLabel
        """
        
        # Get columns with business metadata
        columns_query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX data: <http://data.org/schema#>
        PREFIX stats: <http://statistics.org/schema#>
        PREFIX validation: <http://validation.org/schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?tableLabel ?columnLabel ?dataType ?bigqueryType ?nullable ?primaryKey 
               ?businessMeaning ?sensitivityLevel ?isPII ?accessLevel ?sensitivityReason
               ?validationRule ?pattern ?totalCount ?nonNullCount ?distinctCount
               ?nullPercentage ?cardinality WHERE {
            ?table a sql:Table ;
                   rdfs:label ?tableLabel .
            ?column a sql:Column ;
                    rdfs:label ?columnLabel ;
                    sql:inTable ?table ;
                    sql:dataType ?dataType .
            OPTIONAL { ?column data:bigqueryType ?bigqueryType }
            OPTIONAL { ?column sql:nullable ?nullable }
            OPTIONAL { ?column sql:primaryKey ?primaryKey }
            OPTIONAL { ?column business:meaning ?businessMeaning }
            OPTIONAL { ?column data:sensitivityLevel ?sensitivityLevel }
            OPTIONAL { ?column data:isPII ?isPII }
            OPTIONAL { ?column data:accessLevel ?accessLevel }
            OPTIONAL { ?column data:sensitivityReason ?sensitivityReason }
            OPTIONAL { ?column validation:validationRule ?validationRule }
            OPTIONAL { ?column validation:pattern ?pattern }
            OPTIONAL { ?column stats:totalCount ?totalCount }
            OPTIONAL { ?column stats:nonNullCount ?nonNullCount }
            OPTIONAL { ?column stats:distinctCount ?distinctCount }
            OPTIONAL { ?column stats:nullPercentage ?nullPercentage }
            OPTIONAL { ?column stats:cardinality ?cardinality }
        }
        ORDER BY ?tableLabel ?columnLabel
        """
        
        tables = {}
        
        # Get table info with business context
        for row in self.graph.query(tables_query):
            table_name = str(row[0])
            row_count = int(float(row[1])) if row[1] else 0
            description = str(row[2]) if row[2] else ""
            business_description = str(row[3]) if row[3] else ""
            business_purpose = str(row[4]) if row[4] else ""
            domain_category = str(row[5]) if row[5] else ""
            quality_score = float(row[6]) if row[6] else 0.0
            size_bytes = int(float(row[7])) if row[7] else 0
            
            # Build rich table description from knowledge graph
            table_desc = self._build_table_description_from_kg(
                table_name, row_count, description, business_description, 
                business_purpose, domain_category, quality_score, size_bytes
            )
            
            tables[table_name] = {
                "name": table_name,
                "row_count": row_count,
                "columns": [],
                "description": table_desc,
                "business_description": business_description,
                "business_purpose": business_purpose,
                "domain_category": domain_category,
                "quality_score": quality_score,
                "size_bytes": size_bytes
            }
        
        # Get column info with business context
        for row in self.graph.query(columns_query):
            table_name = str(row[0])
            column_name = str(row[1])
            data_type = str(row[2])
            bigquery_type = str(row[3]) if row[3] else data_type
            is_nullable = bool(row[4]) if row[4] is not None else True
            is_pk = bool(row[5]) if row[5] is not None else False
            business_meaning = str(row[6]) if row[6] else ""
            sensitivity_level = str(row[7]) if row[7] else ""
            is_pii = bool(row[8]) if row[8] is not None else False
            access_level = str(row[9]) if row[9] else ""
            sensitivity_reason = str(row[10]) if row[10] else ""
            validation_rule = str(row[11]) if row[11] else ""
            pattern = str(row[12]) if row[12] else ""
            total_count = int(float(row[13])) if row[13] else 0
            non_null_count = int(float(row[14])) if row[14] else 0
            distinct_count = int(float(row[15])) if row[15] else 0
            null_percentage = float(row[16]) if row[16] else 0.0
            cardinality = float(row[17]) if row[17] else 0.0
            
            if table_name in tables:
                # Build rich column description from knowledge graph
                column_desc = self._build_column_description_from_kg(
                    column_name, data_type, bigquery_type, is_pk, business_meaning,
                    sensitivity_level, is_pii, access_level, sensitivity_reason,
                    validation_rule, pattern
                )
                
                # Get enum values for this column
                enum_values = self._get_column_enum_values(table_name, column_name)
                
                column_info = {
                    "name": column_name,
                    "type": data_type,
                    "bigquery_type": bigquery_type,
                    "nullable": is_nullable,
                    "primary_key": is_pk,
                    "description": column_desc,
                    "business_meaning": business_meaning,
                    "sensitivity_level": sensitivity_level,
                    "is_pii": is_pii,
                    "access_level": access_level,
                    "sensitivity_reason": sensitivity_reason,
                    "validation_rule": validation_rule,
                    "pattern": pattern,
                    "total_count": total_count,
                    "non_null_count": non_null_count,
                    "distinct_count": distinct_count,
                    "null_percentage": null_percentage,
                    "cardinality": cardinality,
                    "enum_values": enum_values
                }
                tables[table_name]["columns"].append(column_info)
        
        return tables
    
    def _get_relationships(self):
        """Get foreign key relationships with business context from BigQuery knowledge graph"""
        fk_query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?fromTable ?fromColumn ?toTable ?toColumn ?businessRelationship 
               ?cardinality ?relationshipType ?businessMeaning WHERE {
            ?fk a sql:ForeignKey ;
                sql:fromTable ?fromTableEntity ;
                sql:toTable ?toTableEntity ;
                sql:fromColumn ?fromColumnEntity ;
                sql:toColumn ?toColumnEntity .
            ?fromTableEntity rdfs:label ?fromTable .
            ?toTableEntity rdfs:label ?toTable .
            ?fromColumnEntity rdfs:label ?fromColumn .
            ?toColumnEntity rdfs:label ?toColumn .
            OPTIONAL { ?fk business:relationshipType ?relationshipType }
            OPTIONAL { ?fk business:cardinality ?cardinality }
            OPTIONAL { ?fk business:meaning ?businessMeaning }
        }
        """
        
        relationships = []
        for row in self.graph.query(fk_query):
            from_table = str(row[0])
            from_column = str(row[1])
            to_table = str(row[2])
            to_column = str(row[3])
            relationship_type = str(row[4]) if row[4] else "REFERENCES"
            cardinality = str(row[5]) if row[5] else "MANY_TO_ONE"
            business_meaning = str(row[6]) if row[6] else ""
            
            rel_description = f"{from_table}.{from_column} → {to_table}.{to_column}"
            
            relationships.append({
                "from_table": from_table,
                "from_column": from_column,
                "to_table": to_table,
                "to_column": to_column,
                "description": rel_description,
                "relationship_type": relationship_type,
                "cardinality": cardinality,
                "business_meaning": business_meaning
            })
        
        return relationships
    
    def _get_data_type_info(self):
        """Get data type distribution from BigQuery knowledge graph"""
        type_query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX data: <http://data.org/schema#>
        
        SELECT ?dataType ?bigqueryType (COUNT(*) AS ?count) WHERE {
            ?column a sql:Column ;
                    sql:dataType ?dataType .
            OPTIONAL { ?column data:bigqueryType ?bigqueryType }
        }
        GROUP BY ?dataType ?bigqueryType
        ORDER BY DESC(?count)
        """
        
        data_types = {}
        for row in self.graph.query(type_query):
            data_type = str(row[0])
            bigquery_type = str(row[1]) if row[1] else data_type
            count = int(row[2])
            
            key = f"{data_type} ({bigquery_type})" if bigquery_type != data_type else data_type
            data_types[key] = count
        
        return data_types
    
    def _get_constraint_info(self):
        """Get constraint information from BigQuery knowledge graph"""
        # BigQuery doesn't have traditional constraints, so we extract validation rules
        validation_query = """
        PREFIX validation: <http://validation.org/schema#>
        PREFIX sql: <http://sql.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?tableLabel ?constraintType (COUNT(*) AS ?count) WHERE {
            ?table a sql:Table ;
                   rdfs:label ?tableLabel .
            ?column sql:inTable ?table .
            {
                ?column sql:primaryKey true .
                BIND("PRIMARY_KEY" AS ?constraintType)
            } UNION {
                ?column validation:required true .
                BIND("NOT_NULL" AS ?constraintType)
            } UNION {
                ?column validation:hasEnumeration true .
                BIND("CHECK_ENUM" AS ?constraintType)
            } UNION {
                ?column validation:pattern ?pattern .
                BIND("PATTERN_CHECK" AS ?constraintType)
            }
        }
        GROUP BY ?tableLabel ?constraintType
        ORDER BY ?tableLabel ?constraintType
        """
        
        constraints = {}
        for row in self.graph.query(validation_query):
            table_name = str(row[0])
            constraint_type = str(row[1])
            count = int(row[2])
            
            if table_name not in constraints:
                constraints[table_name] = {}
            constraints[table_name][constraint_type] = count
        
        return constraints
    
    def _build_table_description_from_kg(self, table_name, row_count, description, 
                                       business_description, business_purpose, 
                                       domain_category, quality_score, size_bytes):
        """Build comprehensive table description from BigQuery knowledge graph data"""
        parts = []
        
        # Use business description if available, fallback to technical description
        if business_description:
            parts.append(business_description)
        elif description:
            parts.append(description)
        
        if business_purpose:
            parts.append(f"Business purpose: {business_purpose}")
        
        if domain_category:
            parts.append(f"Domain: {domain_category}")
        
        # Add data quality metrics
        if quality_score > 0:
            parts.append(f"Quality Score: {quality_score:.1f}/100")
        
        # Add size information
        if size_bytes > 0:
            size_mb = size_bytes / (1024 * 1024)
            if size_mb < 1:
                parts.append(f"Size: {size_bytes} bytes")
            elif size_mb < 1024:
                parts.append(f"Size: {size_mb:.1f} MB")
            else:
                size_gb = size_mb / 1024
                parts.append(f"Size: {size_gb:.2f} GB")
        
        parts.append(f"Contains {row_count:,} records")
        
        return " | ".join(parts) if parts else f"BigQuery table containing {table_name} data ({row_count:,} rows)"
    
    def _build_column_description_from_kg(self, column_name, data_type, bigquery_type, 
                                        is_pk, business_meaning, sensitivity_level, 
                                        is_pii, access_level, sensitivity_reason,
                                        validation_rule, pattern):
        """Build comprehensive column description from BigQuery knowledge graph data"""
        if is_pk:
            return f"Primary key identifier ({data_type})"
        
        parts = []
        
        # Business meaning first
        if business_meaning:
            parts.append(business_meaning)
        
        # Add BigQuery-specific type info
        if bigquery_type != data_type:
            parts.append(f"BigQuery Type: {bigquery_type}")
        
        # Security and privacy information
        if is_pii:
            pii_text = "🔒 PII (Personally Identifiable Information)"
            if sensitivity_reason:
                pii_text += f" - {sensitivity_reason}"
            parts.append(pii_text)
        
        if sensitivity_level:
            sensitivity_text = f"Sensitivity: {sensitivity_level.upper()}"
            if access_level:
                sensitivity_text += f" (Access: {access_level})"
            parts.append(sensitivity_text)
        
        # Validation information
        if validation_rule:
            parts.append(f"Validation: {validation_rule}")
        
        if pattern:
            parts.append(f"Pattern: {pattern}")
        
        # Handle foreign key columns (BigQuery naming convention)
        if column_name.endswith("_id") and not is_pk:
            ref_table = column_name.replace("_id", "")
            # Handle insurance table prefixes
            if ref_table in ["customer", "agent", "policy", "claim"]:
                parts.insert(0, f"Foreign key reference to insurance_{ref_table}s table")
            else:
                parts.insert(0, f"Foreign key reference to {ref_table} table")
        
        return " | ".join(parts) if parts else f"Data column of type {data_type}"
    
    def _get_column_enum_values(self, table_name, column_name):
        """Extract enum values and aliases for a column from BigQuery knowledge graph"""
        # Get allowed values
        values_query = """
        PREFIX validation: <http://validation.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX sql: <http://sql.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?value ?meaning WHERE {
            ?table a sql:Table ;
                   rdfs:label ?tableName .
            ?column a sql:Column ;
                    rdfs:label ?columnName ;
                    sql:inTable ?table ;
                    validation:allowedValue ?valueEntity .
            ?valueEntity rdfs:label ?value .
            OPTIONAL { ?valueEntity business:meaning ?meaning }
        }
        ORDER BY ?value
        """
        
        # Get column aliases
        aliases_query = """
        PREFIX business: <http://business.org/schema#>
        PREFIX sql: <http://sql.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?alias WHERE {
            ?table a sql:Table ;
                   rdfs:label ?tableName .
            ?columnSynonyms a business:ColumnSynonyms ;
                           rdfs:label ?columnLabel ;
                           business:hasAlias ?aliasEntity .
            ?aliasEntity rdfs:label ?alias .
            FILTER(CONTAINS(?columnLabel, ?columnName))
        }
        ORDER BY ?alias
        """
        
        # Bind values for queries
        table_var = table_name
        column_var = column_name
        
        allowed_values = []
        value_meanings = {}
        
        # Execute values query with proper binding
        for row in self.graph.query(values_query):
            # Manual filtering since SPARQL binding is complex
            allowed_values.append(str(row[0]))
            if row[1]:
                value_meanings[str(row[0])] = str(row[1])
        
        aliases = []
        for row in self.graph.query(aliases_query):
            aliases.append(str(row[0]))
        
        if allowed_values or aliases:
            return {
                "allowed_values": allowed_values,
                "value_meanings": value_meanings,
                "aliases": aliases
            }
        
        return {}
    
    def _get_guidelines_from_kg(self):
        """Get query guidelines from BigQuery knowledge graph"""
        guidelines_query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        
        SELECT ?pattern ?description ?usage ?businessPurpose WHERE {
            ?pattern a sql:QueryPattern ;
                     rdfs:label ?patternName ;
                     dcterms:description ?description .
            OPTIONAL { ?pattern sql:usage ?usage }
            OPTIONAL { ?pattern business:useCase ?businessPurpose }
        }
        """
        
        guidelines = "## Query Guidelines (BigQuery Specific):\n\n"
        guidelines += "### BigQuery SQL Best Practices:\n"
        guidelines += "- Use `SELECT *` sparingly in production queries\n"
        guidelines += "- Always use `LIMIT` for exploratory queries\n"
        guidelines += "- Use `DATE()` functions for date comparisons\n"
        guidelines += "- Use `IFNULL()` or `COALESCE()` for null handling\n"
        guidelines += "- Use `ARRAY_AGG()` for aggregating arrays\n"
        guidelines += "- Use standard SQL `COUNT(*)`, `SUM()`, `AVG()` functions\n\n"
        
        patterns_found = False
        for row in self.graph.query(guidelines_query):
            if not patterns_found:
                guidelines += "### Query Patterns from Knowledge Graph:\n\n"
                patterns_found = True
            
            pattern_name = str(row[0]) if row[0] else "Query Pattern"
            description = str(row[1]) if row[1] else ""
            usage = str(row[2]) if row[2] else ""
            business_purpose = str(row[3]) if row[3] else ""
            
            guidelines += f"**{pattern_name}**\n"
            if description:
                guidelines += f"- Description: {description}\n"
            if usage:
                guidelines += f"- Usage: {usage}\n"
            if business_purpose:
                guidelines += f"- Business Purpose: {business_purpose}\n"
            guidelines += "\n"
        
        return guidelines
    
    def _get_synonym_mappings_from_kg(self):
        """Get synonym and alias mappings from BigQuery knowledge graph"""
        # Get table synonyms
        table_synonyms_query = """
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?tableName ?alias WHERE {
            ?tableSynonyms a business:TableSynonyms ;
                          rdfs:label ?tableName ;
                          business:hasAlias ?aliasEntity .
            ?aliasEntity rdfs:label ?alias .
        }
        ORDER BY ?tableName ?alias
        """
        
        # Get column synonyms  
        column_synonyms_query = """
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?columnKey ?alias WHERE {
            ?columnSynonyms a business:ColumnSynonyms ;
                           rdfs:label ?columnKey ;
                           business:hasAlias ?aliasEntity .
            ?aliasEntity rdfs:label ?alias .
        }
        ORDER BY ?columnKey ?alias
        """
        
        synonyms_section = "## Synonym and Alias Mappings:\n\n"
        
        # Process table synonyms
        table_mappings = {}
        for row in self.graph.query(table_synonyms_query):
            table_name = str(row[0])
            alias = str(row[1])
            if table_name not in table_mappings:
                table_mappings[table_name] = []
            table_mappings[table_name].append(alias)
        
        if table_mappings:
            synonyms_section += "### Table Synonyms:\n"
            for table, aliases in table_mappings.items():
                synonyms_section += f"- **{table}**: {', '.join(aliases)}\n"
            synonyms_section += "\n"
        
        # Process column synonyms
        column_mappings = {}
        for row in self.graph.query(column_synonyms_query):
            column_key = str(row[0])
            alias = str(row[1])
            if column_key not in column_mappings:
                column_mappings[column_key] = []
            column_mappings[column_key].append(alias)
        
        if column_mappings:
            synonyms_section += "### Column Synonyms:\n"
            for column, aliases in column_mappings.items():
                synonyms_section += f"- **{column}**: {', '.join(aliases)}\n"
            synonyms_section += "\n"
        
        return synonyms_section if (table_mappings or column_mappings) else ""
    
    def _get_business_rules_from_kg(self):
        """Get business rules and logic patterns from BigQuery knowledge graph"""
        # Get simple business rules
        rules_query = """
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        
        SELECT ?ruleName ?description ?logic ?category ?appliesTo WHERE {
            ?rule a business:Rule ;
                  rdfs:label ?ruleName ;
                  dcterms:description ?description ;
                  business:logic ?logic ;
                  business:category ?category ;
                  business:appliesTo ?appliesTo .
        }
        ORDER BY ?category ?ruleName
        """
        
        # Get composite rules
        composite_rules_query = """
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        
        SELECT ?ruleName ?description ?logic ?category WHERE {
            ?rule a business:CompositeRule ;
                  rdfs:label ?ruleName ;
                  dcterms:description ?description ;
                  business:logic ?logic ;
                  business:category ?category .
        }
        ORDER BY ?category ?ruleName
        """
        
        rules_section = "## Business Rules and Logic Patterns:\n\n"
        
        # Process simple rules by category
        rules_by_category = {}
        for row in self.graph.query(rules_query):
            rule_name = str(row[0])
            description = str(row[1])
            logic = str(row[2])
            category = str(row[3])
            applies_to = str(row[4])
            
            if category not in rules_by_category:
                rules_by_category[category] = []
            
            rules_by_category[category].append({
                "name": rule_name,
                "description": description,
                "logic": logic,
                "applies_to": applies_to
            })
        
        # Display rules by category
        for category, rules in rules_by_category.items():
            rules_section += f"### {category.replace('_', ' ').title()} Rules:\n"
            for rule in rules:
                rules_section += f"- **{rule['name']}**: {rule['description']}\n"
                rules_section += f"  - Logic: `{rule['logic']}`\n"
                rules_section += f"  - Applies to: {rule['applies_to']}\n\n"
        
        # Process composite rules
        composite_rules = list(self.graph.query(composite_rules_query))
        if composite_rules:
            rules_section += "### Composite Business Rules:\n"
            for row in composite_rules:
                rule_name = str(row[0])
                description = str(row[1])
                logic = str(row[2])
                category = str(row[3])
                
                rules_section += f"- **{rule_name}** ({category}): {description}\n"
                rules_section += f"  - Complex Logic: ```sql\n{logic}\n  ```\n\n"
        
        return rules_section if (rules_by_category or composite_rules) else ""
    
    def _get_canonical_join_patterns_from_kg(self):
        """Get canonical join patterns from BigQuery knowledge graph"""
        joins_query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        
        SELECT ?joinName ?description ?joinPattern ?usage ?businessPurpose 
               ?joinType ?directionality ?exampleQuery WHERE {
            ?join a sql:JoinPattern ;
                  rdfs:label ?joinName ;
                  dcterms:description ?description ;
                  sql:joinPattern ?joinPattern .
            OPTIONAL { ?join sql:usage ?usage }
            OPTIONAL { ?join business:purpose ?businessPurpose }
            OPTIONAL { ?join sql:joinType ?joinType }
            OPTIONAL { ?join sql:directionality ?directionality }
            OPTIONAL { ?join sql:exampleQuery ?exampleQuery }
        }
        ORDER BY ?joinName
        """
        
        joins_section = "## Canonical Join Patterns (BigQuery Optimized):\n\n"
        joins_section += "Use these proven join patterns to avoid syntax errors:\n\n"
        
        joins_found = False
        for row in self.graph.query(joins_query):
            joins_found = True
            join_name = str(row[0])
            description = str(row[1])
            join_pattern = str(row[2])
            usage = str(row[3]) if row[3] else ""
            business_purpose = str(row[4]) if row[4] else ""
            join_type = str(row[5]) if row[5] else ""
            directionality = str(row[6]) if row[6] else ""
            example_query = str(row[7]) if row[7] else ""
            
            joins_section += f"### {join_name}\n"
            joins_section += f"**Description**: {description}\n\n"
            joins_section += f"**Pattern**:\n```sql\n{join_pattern}\n```\n\n"
            
            if usage:
                joins_section += f"**Usage**: {usage}\n\n"
            if business_purpose:
                joins_section += f"**Business Purpose**: {business_purpose}\n\n"
            if join_type:
                joins_section += f"**Join Type**: {join_type}\n\n"
            if directionality:
                joins_section += f"**Relationship**: {directionality}\n\n"
            if example_query:
                joins_section += f"**Example Query**: {example_query}\n\n"
            
            joins_section += "---\n\n"
        
        return joins_section if joins_found else ""
    
    def _get_validation_rules_from_kg(self):
        """Get validation rules from BigQuery knowledge graph"""
        validation_query = """
        PREFIX validation: <http://validation.org/schema#>
        PREFIX sql: <http://sql.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?columnLabel ?validationRule ?pattern WHERE {
            ?column a sql:Column ;
                    rdfs:label ?columnLabel .
            {
                ?column validation:validationRule ?validationRule .
            } UNION {
                ?column validation:pattern ?pattern .
            }
        }
        ORDER BY ?columnLabel
        """
        
        validation_section = "## Data Validation Rules:\n\n"
        
        validation_found = False
        for row in self.graph.query(validation_query):
            validation_found = True
            column_label = str(row[0])
            validation_rule = str(row[1]) if row[1] else ""
            pattern = str(row[2]) if row[2] else ""
            
            if validation_rule:
                validation_section += f"- **{column_label}**: {validation_rule}\n"
            if pattern:
                validation_section += f"- **{column_label}** Pattern: `{pattern}`\n"
        
        return validation_section + "\n" if validation_found else ""
    
    def _get_execution_guidelines(self):
        """Get BigQuery-specific execution guidelines"""
        guidelines = "\n## BigQuery Execution Guidelines:\n\n"
        guidelines += "### 1. Query Performance:\n"
        guidelines += "- **Partitioning**: Use `WHERE` clauses on partitioned columns when available\n"
        guidelines += "- **Clustering**: Take advantage of clustered columns for filtering\n"
        guidelines += "- **SELECT Optimization**: Avoid `SELECT *` in production queries\n"
        guidelines += "- **LIMIT Usage**: Always use `LIMIT` for exploratory analysis\n\n"
        
        guidelines += "### 2. BigQuery-Specific Syntax:\n"
        guidelines += "- **Date Functions**: Use `DATE()`, `DATETIME()`, `TIMESTAMP()` functions\n"
        guidelines += "- **String Functions**: Use `CONCAT()`, `SUBSTR()`, `REGEXP_CONTAINS()`\n"
        guidelines += "- **Array Functions**: Use `ARRAY_AGG()`, `UNNEST()` for array operations\n"
        guidelines += "- **Window Functions**: Use `ROW_NUMBER()`, `RANK()`, `LAG()`, `LEAD()`\n\n"
        
        guidelines += "### 3. Data Types and Casting:\n"
        guidelines += "- **Casting**: Use `CAST(column AS TYPE)` or `SAFE_CAST()` for safe casting\n"
        guidelines += "- **Null Handling**: Use `IFNULL()`, `COALESCE()`, `NULLIF()`\n"
        guidelines += "- **Date Arithmetic**: Use `DATE_ADD()`, `DATE_SUB()`, `DATE_DIFF()`\n\n"
        
        guidelines += "### 4. Privacy and Security:\n"
        guidelines += "- **PII Columns**: Be cautious with columns marked as PII (🔒)\n"
        guidelines += "- **Access Controls**: Respect sensitivity levels (HIGH, MEDIUM, LOW)\n"
        guidelines += "- **Data Masking**: Consider using `EXCEPT()` to exclude sensitive columns\n\n"
        
        guidelines += "### 5. Default Query Behavior:\n"
        guidelines += "- **Row Limits**: Add `LIMIT 100` for exploratory queries unless specified\n"
        guidelines += "- **Ordering**: Use `ORDER BY` for consistent results\n"
        guidelines += "- **Aggregations**: Group by non-aggregate columns in SELECT\n\n"
        
        guidelines += "### 6. Insurance Domain Context:\n"
        guidelines += "- **Customer Analysis**: Join customers with policies for portfolio analysis\n"
        guidelines += "- **Claims Analysis**: Join policies with claims for loss analysis\n"
        guidelines += "- **Agent Performance**: Join agents with policies for sales metrics\n"
        guidelines += "- **Risk Assessment**: Use business rules for risk categorization\n\n"
        
        return guidelines
    
    def _get_query_patterns_from_kg(self):
        """Get query patterns from BigQuery knowledge graph"""
        patterns_query = """
        PREFIX sql: <http://sql.org/schema#>
        PREFIX business: <http://business.org/schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        
        SELECT ?patternName ?description ?sqlPattern ?useCase WHERE {
            ?pattern a sql:QueryPattern ;
                     rdfs:label ?patternName ;
                     dcterms:description ?description ;
                     sql:sqlPattern ?sqlPattern ;
                     business:useCase ?useCase .
        }
        ORDER BY ?patternName
        """
        
        patterns_section = "## Common Query Patterns:\n\n"
        
        patterns_found = False
        for row in self.graph.query(patterns_query):
            patterns_found = True
            pattern_name = str(row[0])
            description = str(row[1])
            sql_pattern = str(row[2])
            use_case = str(row[3])
            
            patterns_section += f"### {pattern_name}\n"
            patterns_section += f"**Description**: {description}\n\n"
            patterns_section += f"**Use Case**: {use_case}\n\n"
            patterns_section += f"**SQL Pattern**:\n```sql\n{sql_pattern}\n```\n\n"
        
        return patterns_section if patterns_found else ""
    
    def build_llm_context(self):
        """Build comprehensive context string for LLM with BigQuery knowledge graph intelligence"""
        context = self.extract_schema_context()
        
        llm_context = "# BigQuery Database Schema Context (Knowledge Graph Enhanced)\n\n"
        
        # Database info
        db_info = context["database_info"]
        llm_context += f"**Project**: {db_info.get('project_id', 'unknown')}\n"
        llm_context += f"**Dataset**: {db_info.get('dataset_id', 'unknown')}\n"
        llm_context += f"**Platform**: {db_info.get('platform', 'Google BigQuery')}\n"
        llm_context += f"**Location**: {db_info.get('location', 'US')}\n\n"
        
        # Tables and columns with business intelligence
        llm_context += "## Tables and Columns (with Business Intelligence):\n\n"
        for table_name, table_info in context["tables"].items():
            llm_context += f"### {table_name}\n"
            llm_context += f"**Description:** {table_info['description']}\n"
            
            # Add business metadata
            if table_info.get('business_purpose'):
                llm_context += f"**Business Purpose:** {table_info['business_purpose']}\n"
            if table_info.get('domain_category'):
                llm_context += f"**Domain Category:** {table_info['domain_category']}\n"
            if table_info.get('quality_score', 0) > 0:
                llm_context += f"**Quality Score:** {table_info['quality_score']:.1f}/100\n"
                
            llm_context += "\n**Columns:**\n"
            
            for col in table_info["columns"]:
                pk_indicator = " 🔑 PRIMARY KEY" if col["primary_key"] else ""
                nullable = "NULLABLE" if col["nullable"] else "REQUIRED"
                
                # Enhanced PII indicator with privacy context
                pii_indicator = ""
                if col.get("is_pii"):
                    pii_indicator = " 🔒 PII"
                    if col.get("access_level"):
                        pii_indicator += f" ({col['access_level']} access)"
                
                # Sensitivity indicator
                sensitivity_indicator = ""
                if col.get("sensitivity_level"):
                    sensitivity_indicator = f" [{col['sensitivity_level'].upper()} SENSITIVITY]"
                
                llm_context += f"  - **{col['name']}** ({col['type']}, {nullable}){pk_indicator}{pii_indicator}{sensitivity_indicator}\n"
                llm_context += f"    {col['description']}\n"
                
                # Add enum values if available
                if col.get('enum_values') and col['enum_values'].get('allowed_values'):
                    enum_vals = col['enum_values']['allowed_values']
                    llm_context += f"    *Allowed Values:* {', '.join(f'`{v}`' for v in enum_vals)}\n"
                    
                    # Add value meanings
                    if col['enum_values'].get('value_meanings'):
                        for val, meaning in col['enum_values']['value_meanings'].items():
                            llm_context += f"      - `{val}`: {meaning}\n"
                
                # Add business metadata for columns
                if col.get('business_meaning'):
                    llm_context += f"    *Business Meaning:* {col['business_meaning']}\n"
                if col.get('validation_rule'):
                    llm_context += f"    *Validation:* {col['validation_rule']}\n"
                if col.get('pattern'):
                    llm_context += f"    *Pattern:* `{col['pattern']}`\n"
                
                # Add data statistics
                if col.get('total_count', 0) > 0:
                    stats = []
                    if col.get('distinct_count', 0) > 0:
                        stats.append(f"Distinct: {col['distinct_count']:,}")
                    if col.get('null_percentage', 0) > 0:
                        stats.append(f"Null: {col['null_percentage']:.1f}%")
                    if col.get('cardinality', 0) > 0:
                        stats.append(f"Cardinality: {col['cardinality']:.2f}")
                    if stats:
                        llm_context += f"    *Statistics:* {' | '.join(stats)}\n"
                    
                llm_context += "\n"
            llm_context += "\n"
        
        # Enhanced relationships with business context
        if context["relationships"]:
            llm_context += "## Table Relationships (with Business Context):\n\n"
            for rel in context["relationships"]:
                llm_context += f"- **{rel['description']}**\n"
                if rel.get('business_meaning'):
                    llm_context += f"  *Business Meaning:* {rel['business_meaning']}\n"
                if rel.get('cardinality'):
                    llm_context += f"  *Cardinality:* {rel['cardinality']}\n"
                if rel.get('relationship_type'):
                    llm_context += f"  *Type:* {rel['relationship_type']}\n"
                llm_context += "\n"
        
        # Add knowledge graph sections
        guidelines_section = self._get_guidelines_from_kg()
        if guidelines_section:
            llm_context += guidelines_section
        
        synonyms_section = self._get_synonym_mappings_from_kg()
        if synonyms_section:
            llm_context += synonyms_section
        
        rules_section = self._get_business_rules_from_kg()
        if rules_section:
            llm_context += rules_section
        
        join_patterns_section = self._get_canonical_join_patterns_from_kg()
        if join_patterns_section:
            llm_context += join_patterns_section
        
        validation_section = self._get_validation_rules_from_kg()
        if validation_section:
            llm_context += validation_section
        
        patterns_section = self._get_query_patterns_from_kg()
        if patterns_section:
            llm_context += patterns_section
        
        # BigQuery execution guidelines
        execution_guidelines = self._get_execution_guidelines()
        llm_context += execution_guidelines
        
        return llm_context

    @classmethod
    def from_bqkg_folder(cls, filename="bq_knowledge_graph.ttl"):
        """Create instance by loading TTL file from bqkg folder"""
        kg_file_path = os.path.join("bqkg", filename)
        return cls(kg_file_path)

def demonstrate_bq_context_builder():
    """Demonstrate the BigQuery Knowledge Graph Context Builder"""
    print("Loading BigQuery Knowledge Graph from bqkg folder...")
    
    try:
        # Load from bqkg folder
        context_builder = BQKnowledgeGraphContextBuilder.from_bqkg_folder()
        
        # Extract schema context
        schema_context = context_builder.extract_schema_context()
        print(f"Extracted schema context with {len(schema_context['tables'])} tables")
        
        # Build LLM context
        llm_context = context_builder.build_llm_context()
        
        print("\n" + "="*80)
        print("BIGQUERY LLM-READY SCHEMA CONTEXT")
        print("="*80)
        print(llm_context)
        
        # Save context to file for easy access
        with open("bq_schema_context.md", "w") as f:
            f.write(llm_context)
        print("\nBigQuery context saved to 'bq_schema_context.md'")
        
        return context_builder, llm_context
        
    except FileNotFoundError:
        print("TTL file not found in bqkg folder. Please run the BigQuery KG generation first.")
        return None, None
    except Exception as e:
        print(f"Error loading BigQuery knowledge graph: {e}")
        return None, None

if __name__ == "__main__":
    demonstrate_bq_context_builder()