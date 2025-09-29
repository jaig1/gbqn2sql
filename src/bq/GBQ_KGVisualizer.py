"""
BigQuery Knowledge Graph Visualizer for SQL Knowledge Graph
Provides interactive visualization capabilities using streamlit-agraph for BigQuery
"""

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import rdflib
from rdflib import Graph, Namespace, URIRef
import networkx as nx
from typing import List, Dict, Tuple, Set
import colorsys
import random

class GBQ_KGVisualizer:
    def __init__(self, kg_file_path: str = "bqkg/bq_knowledge_graph.ttl"):
        """Initialize the BigQuery KG visualizer with RDF knowledge graph file"""
        self.kg_file_path = kg_file_path
        self.graph = Graph()
        self.graph.parse(kg_file_path, format="turtle")
        
        # Define namespaces for BigQuery knowledge graph
        self.sqldata = Namespace("http://example.com/sql-kg/data#")
        self.sqlkg = Namespace("http://example.com/sql-kg/ontology#")
        
        # BigQuery-specific namespaces (actual structure)
        self.sql = Namespace("http://sql.org/schema#")
        self.business = Namespace("http://business.org/schema#")
        self.data = Namespace("http://data.org/schema#")
        self.insurance = Namespace("http://insurance.org/schema#")
        self.stats = Namespace("http://statistics.org/schema#")
        
        # Color palette for different node types (BigQuery optimized)
        self.node_colors = {
            'Table': '#4285F4',           # Google Blue for BigQuery tables
            'Column': '#34A853',          # Google Green for columns
            'BusinessRule': '#EA4335',    # Google Red for business rules
            'BusinessConcept': '#FF9800', # Orange for business concepts
            'CanonicalJoinPattern': '#FBBC04',  # Google Yellow for join patterns
            'Constraint': '#9AA0A6',      # Google Grey for constraints
            'Dataset': '#673AB7',         # Purple for BigQuery datasets
            'Default': '#DDA0DD'          # Plum for unknown types
        }

        # Size mapping for different node types
        self.node_sizes = {
            'Table': 25,
            'Column': 15,
            'BusinessRule': 20,
            'BusinessConcept': 18,
            'CanonicalJoinPattern': 22,
            'Constraint': 12,
            'Dataset': 30,
            'Default': 18
        }

    def _sanitize_title(self, title: str) -> str:
        """Sanitize title to avoid streamlit-agraph path interpretation issues"""
        # Replace colons with dashes and other problematic characters
        return title.replace(':', ' -').replace('/', '_').replace('\\', '_')
    
    def get_node_type(self, node_uri: str) -> str:
        """Determine the type of a node based on its RDF type"""
        node_ref = URIRef(node_uri)
        
        # Query for the type with BigQuery-specific types
        for _, _, obj in self.graph.triples((node_ref, rdflib.RDF.type, None)):
            type_name = str(obj).split('#')[-1]
            
            # Map BigQuery RDF types to our expected types
            type_mapping = {
                'Table': 'Table',
                'Column': 'Column', 
                'Rule': 'BusinessRule',  # business:Rule -> BusinessRule
                'Concept': 'BusinessConcept',  # business:Concept -> BusinessConcept
                'JoinPattern': 'CanonicalJoinPattern',  # sql:JoinPattern -> CanonicalJoinPattern
                'Constraint': 'Constraint'
            }
            
            return type_mapping.get(type_name, type_name)
        
        # Fallback to URI analysis for BigQuery-specific patterns
        if '/tables/' in node_uri and 'insurance_' in node_uri:
            return 'Table'
        elif '/columns/' in node_uri:
            return 'Column'
        elif '/business-rules/' in node_uri:
            return 'BusinessRule'
        elif '/business-concepts/' in node_uri:
            return 'BusinessConcept'
        elif 'canonical_join_' in node_uri:
            return 'CanonicalJoinPattern'
        elif 'pk_' in node_uri or 'fk_' in node_uri:
            return 'Constraint'
        elif '/datasets/' in node_uri or 'insurance_analytics' in node_uri:
            return 'Dataset'
        else:
            return 'Default'
    
    def get_node_label(self, node_uri: str) -> str:
        """Get a human-readable label for a node"""
        node_ref = URIRef(node_uri)
        
        # Try to get various label properties
        label_props = [
            self.sqlkg.hasName,
            rdflib.RDFS.label,
            self.business.label,
            self.sql.name
        ]
        
        for prop in label_props:
            for _, _, obj in self.graph.triples((node_ref, prop, None)):
                return str(obj)
        
        # Fallback to URI analysis for BigQuery-specific patterns
        if '/tables/' in node_uri:
            # Extract table name from BigQuery URI
            parts = node_uri.split('/')
            if 'insurance_' in node_uri:
                table_name = parts[-1].replace('insurance_', '')
                return table_name.capitalize()
        elif '/columns/' in node_uri:
            # Extract column name
            parts = node_uri.split('/')
            return parts[-1]
        elif '/business-rules/' in node_uri:
            # Extract rule name
            parts = node_uri.split('/')
            rule_id = parts[-1]
            return f"Rule {rule_id.replace('rule_', '')}"
        elif '/business-concepts/' in node_uri:
            # Extract concept name
            parts = node_uri.split('/')
            return parts[-1].replace('_', ' ').title()
        
        # Generic fallback
        label = node_uri.split('#')[-1].split('/')[-1].replace('_', ' ').title()
        
        # Special handling for BigQuery table names
        if label.lower() in ['agents', 'customers', 'policies', 'claims']:
            return label.capitalize()
        
        return label
    
    def get_node_description(self, node_uri: str) -> str:
        """Get description for tooltip"""
        node_ref = URIRef(node_uri)
        
        # Try different description properties
        description_props = [
            self.sqlkg.description,
            self.sqlkg.ruleDescription,
            self.sqlkg.businessPurpose,
            self.sqlkg.bigqueryContext  # BigQuery-specific context
        ]
        
        for prop in description_props:
            for _, _, obj in self.graph.triples((node_ref, prop, None)):
                return str(obj)
        
        return f"BigQuery {self.get_node_type(node_uri)}: {self.get_node_label(node_uri)}"
    
    def find_node_uri_by_id(self, node_id: str) -> str:
        """Find the full URI for a node based on its ID/label"""
        # First, try to find by exact URI match
        if node_id.startswith('http'):
            return node_id
            
        # Search through all subjects in the graph to find a matching node
        for subject in self.graph.subjects():
            subject_str = str(subject)
            
            # Check if the node_id appears in the URI
            if node_id in subject_str:
                return subject_str
                
            # Check if the node_id matches a label
            label = self.get_node_label(subject_str)
            if label.lower() == node_id.lower():
                return subject_str
                
            # Check for partial matches in cleaned labels
            cleaned_label = label.replace(' ', '_').lower()
            if cleaned_label == node_id.lower() or node_id.lower() in cleaned_label:
                return subject_str
        
        # If no match found, return the original ID
        return node_id

    def get_node_details_by_uri(self, node_uri: str) -> Dict:
        """Get detailed information about a node by its URI"""
        try:
            # First try to resolve the node URI if it's just an ID
            full_uri = self.find_node_uri_by_id(node_uri)
            node_ref = URIRef(full_uri)
            
            details = {
                'uri': full_uri,
                'label': self.get_node_label(full_uri),
                'type': self.get_node_type(full_uri),
                'description': self.get_node_description(full_uri),
                'properties': {}
            }
            
            # Get all properties for this node
            for predicate, obj in self.graph.predicate_objects(node_ref):
                prop_name = str(predicate).split('#')[-1].split('/')[-1]
                details['properties'][prop_name] = str(obj)
            
            return details
        except Exception as e:
            return {'error': f"Could not get details for {node_uri}: {str(e)}"}
    
    def create_filtered_graph(self, node_types: List[str], max_nodes: int = 50) -> Tuple[List[Node], List[Edge]]:
        """Create nodes and edges for specific node types"""
        nodes = []
        edges = []
        added_nodes = set()
        
        # BigQuery RDF type mappings
        type_queries = {
            'Table': self.sql.Table,
            'Column': self.sql.Column,
            'BusinessRule': self.business.Rule,
            'BusinessConcept': self.business.Concept,
            'CanonicalJoinPattern': self.sql.JoinPattern,  # BigQuery join patterns
            'Constraint': self.sqlkg.Constraint,  # Fallback
            'Dataset': self.sqlkg.Dataset  # Fallback
        }
        
        node_count = 0
        
        for node_type in node_types:
            if node_type in type_queries and node_count < max_nodes:
                # Query for nodes of this type
                for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, type_queries[node_type])):
                    if node_count >= max_nodes:
                        break
                    
                    node_uri = str(subj)
                    if node_uri not in added_nodes:
                        label = self.get_node_label(node_uri)
                        description = self.get_node_description(node_uri)
                        
                        node = Node(
                            id=node_uri,
                            label=label,
                            size=self.node_sizes.get(node_type, 18),
                            color=self.node_colors.get(node_type, '#DDA0DD'),
                            title=description
                        )
                        nodes.append(node)
                        added_nodes.add(node_uri)
                        node_count += 1
        
        # Create edges between related nodes (BigQuery-specific properties)
        edge_properties = [
            self.sql.inTable,        # Column to Table relationship
            self.sql.hasColumn,      # Table to Column relationship
            self.business.appliesTo, # Rule/Concept to Table relationship
            self.sqlkg.hasColumn,    # Fallback
            self.sqlkg.hasConstraint,
            self.sqlkg.appliesTo,
            self.sqlkg.relatedTo,
            self.sqlkg.referencesTable,
            self.sqlkg.referencesColumn,
            self.sqlkg.partOf  # BigQuery-specific: table part of dataset
        ]
        
        for prop in edge_properties:
            for subj, _, obj in self.graph.triples((None, prop, None)):
                subj_uri = str(subj)
                obj_uri = str(obj)
                
                if subj_uri in added_nodes and obj_uri in added_nodes:
                    edge_label = str(prop).split('#')[-1].replace('_', ' ')
                    edge = Edge(
                        source=subj_uri,
                        target=obj_uri,
                        label=edge_label,
                        color='#gray'
                    )
                    edges.append(edge)
        
        return nodes, edges
    
    def create_bigquery_dataset_view(self) -> Tuple[List[Node], List[Edge]]:
        """Create a BigQuery-specific view showing dataset structure"""
        nodes = []
        edges = []
        
        # Create dataset node (insurance_analytics)
        dataset_node = Node(
            id="insurance_analytics_dataset",
            label="insurance_analytics",
            size=35,
            color=self.node_colors['Dataset'],
            title=self._sanitize_title("BigQuery Dataset: insurance_analytics - Contains all insurance-related tables and data")
        )
        nodes.append(dataset_node)
        
        # Get all tables and create the dataset hierarchy
        for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.sql.Table)):
            table_uri = str(subj)
            table_label = self.get_node_label(table_uri)
            table_desc = self.get_node_description(table_uri)
            
            # Create table node
            table_node = Node(
                id=table_uri,
                label=table_label,
                size=30,
                color=self.node_colors['Table'],
                title=self._sanitize_title(f"BigQuery Table: {table_label} - {table_desc}")
            )
            nodes.append(table_node)
            
            # Create edge from dataset to table
            dataset_edge = Edge(
                source="insurance_analytics_dataset",
                target=table_uri,
                label="contains",
                color='#4285F4'
            )
            edges.append(dataset_edge)
            
            # Get columns for this table (columns have sql:inTable pointing to this table)
            for col_subj, _, _ in self.graph.triples((None, self.sql.inTable, URIRef(table_uri))):
                col_uri = str(col_subj)
                col_label = self.get_node_label(col_uri)
                col_desc = self.get_node_description(col_uri)
                
                # Create column node
                col_node = Node(
                    id=col_uri,
                    label=col_label,
                    size=15,
                    color=self.node_colors['Column'],
                    title=self._sanitize_title(f"BigQuery Column: {col_label} - {col_desc}")
                )
                nodes.append(col_node)
                
                # Create edge from table to column
                edge = Edge(
                    source=table_uri,
                    target=col_uri,
                    label="has column",
                    color='#34A853'
                )
                edges.append(edge)
        
        return nodes, edges
    
    def create_table_schema_view(self) -> Tuple[List[Node], List[Edge]]:
        """Create a focused view of BigQuery tables and their columns"""
        nodes = []
        edges = []
        
        # Get all tables using BigQuery RDF structure
        for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.sql.Table)):
            table_uri = str(subj)
            table_label = self.get_node_label(table_uri)
            table_desc = self.get_node_description(table_uri)
            
            # Create table node with BigQuery styling
            table_node = Node(
                id=table_uri,
                label=table_label,
                size=30,
                color=self.node_colors['Table'],
                title=self._sanitize_title(f"BigQuery Table: {table_label} - {table_desc}")
            )
            nodes.append(table_node)
            
            # Get columns for this table using BigQuery structure
            # In BigQuery KG, columns reference tables with sql:inTable
            for col_subj, _, _ in self.graph.triples((None, self.sql.inTable, URIRef(table_uri))):
                col_uri = str(col_subj)
                col_label = self.get_node_label(col_uri)
                col_desc = self.get_node_description(col_uri)
                
                # Create column node
                col_node = Node(
                    id=col_uri,
                    label=col_label,
                    size=15,
                    color=self.node_colors['Column'],
                    title=self._sanitize_title(f"BigQuery Column: {col_label} - {col_desc}")
                )
                nodes.append(col_node)
                
                # Create edge from table to column
                edge = Edge(
                    source=table_uri,
                    target=col_uri,
                    label="has column",
                    color='#666666'
                )
                edges.append(edge)
        
        return nodes, edges
    
    def create_business_intelligence_dashboard(self, focus_filter: str = "Business Rules") -> Tuple[List[Node], List[Edge]]:
        """Create a focused Business Intelligence Dashboard view"""
        nodes = []
        edges = []
        added_nodes = set()
        
        # Enhanced color palette for business intelligence
        bi_colors = {
            'agent_performance': '#FF6B6B',     # Coral for agent metrics
            'claims_severity': '#4ECDC4',       # Teal for claim analysis  
            'claims_status': '#45B7D1',         # Blue for status tracking
            'claims_timing': '#96CEB4',         # Mint for time-based rules
            'claims_value': '#FFEAA7',          # Yellow for value thresholds
            'financial_metrics': '#DDA0DD',     # Plum for financial rules
            'premium_segmentation': '#98D8C8',  # Aqua for segmentation
            'demographics': '#F7DC6F',          # Gold for customer demographics
            'data_quality': '#BB8FCE',          # Lavender for quality rules
            'BusinessConcept': '#FF9800',       # Orange for concepts
            'Table': '#4285F4',                 # Google Blue for tables
            'Column_PII': '#E74C3C',            # Red for PII columns
            'Column_Sensitive': '#F39C12',      # Orange for sensitive columns
            'Column_Regular': '#27AE60'         # Green for regular columns
        }
        
        # 1. CREATE BUSINESS RULES LAYER (filtered by focus)
        rule_categories = {}
        if focus_filter in ["All", "Business Rules"]:
            for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.business.Rule)):
                rule_uri = str(subj)
                rule_label = self.get_node_label(rule_uri)
                rule_desc = self.get_node_description(rule_uri)
                
                # Get rule category and logic
                category = "general"
                logic = ""
                for _, _, cat_obj in self.graph.triples((URIRef(rule_uri), self.business.category, None)):
                    category = str(cat_obj)
                    break
                
                for _, _, logic_obj in self.graph.triples((URIRef(rule_uri), self.business.logic, None)):
                    logic = str(logic_obj)
                    break
                
                # Track categories for grouping and apply category filter
                if category not in rule_categories:
                    rule_categories[category] = []
                rule_categories[category].append(rule_uri)
                

                
                # Create enhanced rule node with business intelligence
                rule_color = bi_colors.get(category, '#EA4335')
                rule_size = 25 if 'subquery' in logic.lower() else 20  # Larger for complex rules
                
                # Enhanced tooltip with business context
                tooltip = f"📋 Business Rule: {rule_label}\\n"
                tooltip += f"🏷️ Category: {category.replace('_', ' ').title()}\\n"
                tooltip += f"⚙️ Logic: {logic[:100]}{'...' if len(logic) > 100 else ''}\\n"
                tooltip += f"📖 Description: {rule_desc}"
                
                rule_node = Node(
                    id=rule_uri,
                    label=f"{rule_label}\\n[{category.replace('_', ' ')}]",
                    size=rule_size,
                    color=rule_color,
                    title=self._sanitize_title(tooltip),
                    shape="box"  # Rectangular for business rules
                )
                nodes.append(rule_node)
                added_nodes.add(rule_uri)
        
        # 2. CREATE BUSINESS CONCEPTS LAYER (filtered by focus) 
        concept_cluster = {}
        if focus_filter in ["All", "Business Domain Vocabulary", "Synonym Networks"]:
            for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.business.Concept)):
                concept_uri = str(subj)
                concept_label = self.get_node_label(concept_uri)
                concept_desc = self.get_node_description(concept_uri)
                
                # Create central concept node
                concept_tooltip = f"💡 Business Concept: {concept_label}\\n📖 {concept_desc}"
                concept_node = Node(
                    id=concept_uri,
                    label=f"💡 {concept_label}",
                    size=30,
                    color=bi_colors['BusinessConcept'],
                    title=self._sanitize_title(concept_tooltip),
                    shape="diamond"  # Diamond for concepts
                )
                nodes.append(concept_node)
                added_nodes.add(concept_uri)
                
                # Get related terms (synonyms) and create satellite nodes
                related_terms = []
                for _, _, term_obj in self.graph.triples((URIRef(concept_uri), self.business.relatedTerm, None)):
                    related_terms.append(str(term_obj))
            
                # Create synonym nodes (only for Synonym Networks focus)
                if focus_filter in ["All", "Synonym Networks"]:
                    for i, term in enumerate(related_terms):
                        synonym_id = f"{concept_uri}_synonym_{i}"
                        synonym_node = Node(
                            id=synonym_id,
                            label=term,
                            size=12,
                            color="#FFE0B2",  # Light orange for synonyms
                            title=self._sanitize_title(f"📝 Synonym for {concept_label}: {term}"),
                            shape="ellipse"
                        )
                        nodes.append(synonym_node)
                        
                        # Connect synonym to concept
                        synonym_edge = Edge(
                            source=concept_uri,
                            target=synonym_id,
                            label="synonym",
                            color="#FFA726",
                            dashes=True  # Dashed for synonym relationships
                        )
                        edges.append(synonym_edge)
        
        # 3. CREATE TABLE LAYER (filtered by focus)
        if focus_filter in ["All", "Business Rules", "Data Governance"]:
            for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.sql.Table)):
                table_uri = str(subj)
                table_label = self.get_node_label(table_uri)
                
                # Get business concept for table
                table_concept = ""
                quality_score = ""
                for _, _, concept_obj in self.graph.triples((URIRef(table_uri), self.insurance.concept, None)):
                    table_concept = str(concept_obj).replace('_', ' ').title()
                    break
                
                for _, _, quality_obj in self.graph.triples((URIRef(table_uri), self.stats.qualityScore, None)):
                    quality_score = f" (Q: {float(str(quality_obj)):.1f}%)"
                    break
                
                # Enhanced table tooltip
                table_tooltip = f"🗃️ Table: {table_label}\\n"
                if table_concept:
                    table_tooltip += f"💼 Business Concept: {table_concept}\\n"
                if quality_score:
                    table_tooltip += f"📊 Data Quality: {quality_score.strip(' (Q: )')}"
                
                table_node = Node(
                    id=table_uri,
                    label=f"{table_label}{quality_score}",
                    size=35,
                    color=bi_colors['Table'],
                    title=self._sanitize_title(table_tooltip),
                    shape="box"
                )
                nodes.append(table_node)
                added_nodes.add(table_uri)        # 4. CREATE COLUMN LAYER (only for Data Governance focus)
        pii_columns = []
        sensitive_columns = []
        regular_columns = []
        
        if focus_filter in ["All", "Data Governance"]:
            for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.sql.Column)):
                col_uri = str(subj)
                col_label = self.get_node_label(col_uri)
                
                # Determine PII status and sensitivity
                is_pii = False
                sensitivity_level = "low"
                sensitivity_reason = ""
                business_meaning = ""
                data_type = ""
                
                for _, _, pii_obj in self.graph.triples((URIRef(col_uri), self.data.isPII, None)):
                    is_pii = str(pii_obj).lower() == 'true'
                    break
                    
                for _, _, sens_obj in self.graph.triples((URIRef(col_uri), self.data.sensitivityLevel, None)):
                    sensitivity_level = str(sens_obj)
                    break
                    
                for _, _, reason_obj in self.graph.triples((URIRef(col_uri), self.data.sensitivityReason, None)):
                    sensitivity_reason = str(reason_obj)
                    break
                    
                for _, _, meaning_obj in self.graph.triples((URIRef(col_uri), self.business.meaning, None)):
                    business_meaning = str(meaning_obj)
                    break
                    
                for _, _, type_obj in self.graph.triples((URIRef(col_uri), self.data.bigqueryType, None)):
                    data_type = str(type_obj)
                    break
                
                # For data governance focus, only show PII and sensitive columns
                if not (is_pii or sensitivity_level in ["high", "medium"]):
                    continue
                
                # Categorize column for coloring
                if is_pii:
                    col_color = bi_colors['Column_PII']
                    col_icon = "🔒"
                    pii_columns.append(col_uri)
                elif sensitivity_level == "high":
                    col_color = bi_colors['Column_Sensitive'] 
                    col_icon = "⚠️"
                    sensitive_columns.append(col_uri)
                else:
                    col_color = bi_colors['Column_Regular']
                    col_icon = "📊"
                    regular_columns.append(col_uri)
                
                # Enhanced column tooltip with business intelligence
                col_tooltip = f"{col_icon} Column: {col_label} ({data_type})\\n"
                if is_pii:
                    col_tooltip += "🔒 Contains PII (Personally Identifiable Information)\\n"
                if sensitivity_level != "low":
                    col_tooltip += f"⚠️ Sensitivity: {sensitivity_level.title()}\\n"
                if sensitivity_reason:
                    col_tooltip += f"📋 Reason: {sensitivity_reason}\\n"
                if business_meaning:
                    col_tooltip += f"💼 Business Meaning: {business_meaning}"
                
                col_size = 18 if is_pii else (15 if sensitivity_level == "high" else 12)
                
                col_node = Node(
                    id=col_uri,
                    label=f"{col_icon} {col_label}",
                    size=col_size,
                    color=col_color,
                    title=self._sanitize_title(col_tooltip)
                )
                nodes.append(col_node)
                added_nodes.add(col_uri)
        
        # 5. CREATE INTELLIGENT RELATIONSHIPS
        
        # Business rules to tables relationships
        for subj, _, obj in self.graph.triples((None, self.business.appliesTo, None)):
            rule_uri = str(subj)
            table_uri = str(obj)
            if rule_uri in added_nodes and table_uri in added_nodes:
                edge = Edge(
                    source=rule_uri,
                    target=table_uri,
                    label="applies to",
                    color="#E74C3C",
                    width=3
                )
                edges.append(edge)
        
        # Table to column relationships with PII highlighting
        for subj, _, obj in self.graph.triples((None, self.sql.inTable, None)):
            col_uri = str(subj)
            table_uri = str(obj)
            if col_uri in added_nodes and table_uri in added_nodes:
                # Different edge styles for different column types
                if col_uri in pii_columns:
                    edge_color = "#E74C3C"  # Red for PII
                    edge_width = 3
                elif col_uri in sensitive_columns:
                    edge_color = "#F39C12"  # Orange for sensitive
                    edge_width = 2
                else:
                    edge_color = "#27AE60"  # Green for regular
                    edge_width = 1
                
                edge = Edge(
                    source=table_uri,
                    target=col_uri,
                    label="contains",
                    color=edge_color,
                    width=edge_width
                )
                edges.append(edge)
        
        return nodes, edges
    
    def create_business_rules_view(self) -> Tuple[List[Node], List[Edge]]:
        """Create a view focused on BigQuery business rules and their relationships"""
        nodes = []
        edges = []
        
        # Get all business rules using BigQuery RDF structure
        for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.business.Rule)):
            rule_uri = str(subj)
            rule_label = self.get_node_label(rule_uri)
            rule_desc = self.get_node_description(rule_uri)
            
            # Create business rule node
            rule_node = Node(
                id=rule_uri,
                label=rule_label,
                size=20,
                color=self.node_colors['BusinessRule'],
                title=self._sanitize_title(f"BigQuery Business Rule: {rule_label} - {rule_desc}")
            )
            nodes.append(rule_node)
            
            # Get tables this rule applies to (try both business and sqlkg namespaces)
            for _, _, table_obj in self.graph.triples((URIRef(rule_uri), self.business.appliesTo, None)):
                table_uri = str(table_obj)
                table_label = self.get_node_label(table_uri)
                table_desc = self.get_node_description(table_uri)
                
                # Create table node if not already added
                if not any(node.id == table_uri for node in nodes):
                    table_node = Node(
                        id=table_uri,
                        label=table_label,
                        size=25,
                        color=self.node_colors['Table'],
                        title=self._sanitize_title(f"BigQuery Table: {table_label} - {table_desc}")
                    )
                    nodes.append(table_node)
                
                # Create edge from rule to table
                edge = Edge(
                    source=rule_uri,
                    target=table_uri,
                    label="applies to",
                    color='#EA4335'
                )
                edges.append(edge)
        
        # Also get business concepts
        for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.business.Concept)):
            concept_uri = str(subj)
            concept_label = self.get_node_label(concept_uri)
            concept_desc = self.get_node_description(concept_uri)
            
            # Create business concept node
            concept_node = Node(
                id=concept_uri,
                label=concept_label,
                size=18,
                color=self.node_colors['BusinessConcept'],
                title=self._sanitize_title(f"BigQuery Business Concept: {concept_label} - {concept_desc}")
            )
            nodes.append(concept_node)
        
        return nodes, edges
    
    def create_join_patterns_view(self) -> Tuple[List[Node], List[Edge]]:
        """Create a view focused on BigQuery join patterns"""
        nodes = []
        edges = []
        
        # Get all join patterns using BigQuery RDF structure
        for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.sql.JoinPattern)):
            join_uri = str(subj)
            join_label = self.get_node_label(join_uri)
            join_desc = self.get_node_description(join_uri)
            
            # Create join pattern node
            join_node = Node(
                id=join_uri,
                label=join_label,
                size=22,
                color=self.node_colors.get('CanonicalJoinPattern', '#FBBC04'),
                title=self._sanitize_title(f"BigQuery Join Pattern: {join_label} - {join_desc}")
            )
            nodes.append(join_node)
            
            # Try to get join pattern details and extract table references
            for _, _, pattern_obj in self.graph.triples((URIRef(join_uri), self.sql.joinPattern, None)):
                pattern_text = str(pattern_obj)
                # Extract table names from join pattern text (simplified approach)
                if 'insurance_' in pattern_text:
                    # This is a basic approach - could be enhanced with regex
                    pass
        
        return nodes, edges

    def create_privacy_aware_view(self) -> Tuple[List[Node], List[Edge]]:
        """Create a view focused on BigQuery data privacy and PII detection"""
        nodes = []
        edges = []
        
        # Get all tables using BigQuery RDF structure
        for subj, _, _ in self.graph.triples((None, rdflib.RDF.type, self.sql.Table)):
            table_uri = str(subj)
            table_label = self.get_node_label(table_uri)
            
            # Create table node
            table_node = Node(
                id=table_uri,
                label=table_label,
                size=25,
                color=self.node_colors['Table'],
                title=self._sanitize_title(f"BigQuery Table: {table_label}")
            )
            nodes.append(table_node)
            
            # Get columns for this table using BigQuery structure
            for col_subj, _, _ in self.graph.triples((None, self.sql.inTable, URIRef(table_uri))):
                col_uri = str(col_subj)
                col_label = self.get_node_label(col_uri)
                
                # Check for PII using BigQuery knowledge graph properties
                is_pii = False
                sensitivity_level = "Low"
                sensitivity_reason = ""
                
                # Check data:isPII property
                for _, _, pii_value in self.graph.triples((URIRef(col_uri), self.data.isPII, None)):
                    if str(pii_value).lower() == 'true':
                        is_pii = True
                        break
                
                # Get sensitivity level
                for _, _, sens_obj in self.graph.triples((URIRef(col_uri), self.data.sensitivity, None)):
                    sensitivity_level = str(sens_obj)
                    break
                
                # Get sensitivity reason
                for _, _, reason_obj in self.graph.triples((URIRef(col_uri), self.data.sensitivityReason, None)):
                    sensitivity_reason = str(reason_obj)
                    break
                
                # Color code based on PII status and sensitivity
                if is_pii:
                    if sensitivity_level == "High":
                        col_color = '#FF4444'  # Red for high sensitivity PII
                        size = 20
                    else:
                        col_color = '#FF8844'  # Orange for medium sensitivity PII
                        size = 16
                    col_title = f"⚠️ PII: {col_label} (Sensitivity: {sensitivity_level}) - {sensitivity_reason}"
                else:
                    col_color = '#44AA44'  # Green for safe data
                    size = 12
                    col_title = f"✅ Safe: {col_label} (Non-PII BigQuery Column)"
                
                col_node = Node(
                    id=col_uri,
                    label=col_label,
                    size=size,
                    color=col_color,
                    title=col_title
                )
                nodes.append(col_node)
                
                # Create edge with privacy indicator
                if is_pii:
                    edge_label = f"⚠️ {sensitivity_level} PII"
                    edge_color = '#FF6666' if sensitivity_level == "High" else '#FF8866'
                else:
                    edge_label = "Safe Column"
                    edge_color = '#66AA66'
                
                edge = Edge(
                    source=table_uri,
                    target=col_uri,
                    label=edge_label,
                    color=edge_color
                )
                edges.append(edge)

        return nodes, edges

    def execute_sparql_query(self, sparql_query: str) -> List[Dict]:
        """Execute a custom SPARQL query on the BigQuery knowledge graph"""
        try:
            # Execute the SPARQL query
            results = self.graph.query(sparql_query)
            
            # Convert results to list of dictionaries
            result_list = []
            for row in results:
                row_dict = {}
                for i, var in enumerate(results.vars):
                    value = row[i]
                    # Convert URIRef and Literal to string
                    if value is not None:
                        row_dict[str(var)] = str(value)
                    else:
                        row_dict[str(var)] = None
                result_list.append(row_dict)
            
            return result_list
            
        except Exception as e:
            st.error(f"BigQuery SPARQL Query Error: {str(e)}")
            return []

    def get_bigquery_example_sparql_queries(self) -> Dict[str, Dict[str, str]]:
        """Get BigQuery-specific example SPARQL queries with plain English descriptions"""
        return {
            "Show all BigQuery tables and their business purposes": {
                "description": "Lists all tables in the BigQuery insurance_analytics dataset with their business purposes and descriptions",
                "sparql": """
                    PREFIX sql: <http://sql.org/schema#>
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    
                    SELECT ?table ?description WHERE {
                        ?table a sql:Table .
                        OPTIONAL { ?table dcterms:description ?description }
                    }
                """
            },
            
            "Find all BigQuery business rules organized by category": {
                "description": "Shows all business rules grouped by their categories (premium_segmentation, agent_performance, etc.) for BigQuery analytics",
                "sparql": """
                    PREFIX business: <http://business.org/schema#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT ?rule ?label ?category WHERE {
                        ?rule a business:Rule .
                        OPTIONAL { ?rule rdfs:label ?label }
                        OPTIONAL { ?rule business:category ?category }
                    }
                    ORDER BY ?category ?label
                """
            },
            
            "Discover all BigQuery columns with their data types": {
                "description": "Lists all columns across all BigQuery tables with their data types and which table they belong to",
                "sparql": """
                    PREFIX sql: <http://sql.org/schema#>
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    
                    SELECT ?column ?identifier ?datatype ?table WHERE {
                        ?column sql:inTable ?table .
                        OPTIONAL { ?column dcterms:identifier ?identifier }
                        OPTIONAL { ?column sql:dataType ?datatype }
                    }
                """
            },
            
            "Identify all sensitive BigQuery data and PII columns": {
                "description": "Finds columns containing personally identifiable information (PII) with sensitivity levels for BigQuery compliance",
                "sparql": """
                    PREFIX sql: <http://sql.org/schema#>
                    PREFIX data: <http://data.org/schema#>
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    
                    SELECT ?column ?identifier ?sensitivity ?reason WHERE {
                        ?column data:isPII true .
                        OPTIONAL { ?column dcterms:identifier ?identifier }
                        OPTIONAL { ?column data:sensitivity ?sensitivity }
                        OPTIONAL { ?column data:sensitivityReason ?reason }
                    }
                """
            },
            
            "Explore BigQuery table relationships and join patterns": {
                "description": "Shows canonical join patterns that define how BigQuery tables are typically connected for optimal performance",
                "sparql": """
                    PREFIX sql: <http://sql.org/schema#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    
                    SELECT ?pattern ?name ?description ?joinType WHERE {
                        ?pattern a sql:JoinPattern .
                        OPTIONAL { ?pattern rdfs:label ?name }
                        OPTIONAL { ?pattern dcterms:description ?description }
                        OPTIONAL { ?pattern sql:joinType ?joinType }
                    }
                """
            },
            
            "Show all BigQuery premium segmentation business rules": {
                "description": "Displays business rules specifically related to premium segmentation (Budget, Standard, Premium) for BigQuery analytics",
                "sparql": """
                    PREFIX business: <http://business.org/schema#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    
                    SELECT ?rule ?label ?condition ?description WHERE {
                        ?rule a business:Rule .
                        ?rule business:category "premium_segmentation" .
                        OPTIONAL { ?rule rdfs:label ?label }
                        OPTIONAL { ?rule business:condition ?condition }
                        OPTIONAL { ?rule dcterms:description ?description }
                    }
                """
            },
            
            "Find all BigQuery agent performance evaluation rules": {
                "description": "Lists business rules that define agent performance categories (Top Performer, New Agent, etc.) for BigQuery reporting",
                "sparql": """
                    PREFIX business: <http://business.org/schema#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    
                    SELECT ?rule ?label ?condition ?description WHERE {
                        ?rule a business:Rule .
                        ?rule business:category "agent_performance" .
                        OPTIONAL { ?rule rdfs:label ?label }
                        OPTIONAL { ?rule business:condition ?condition }
                        OPTIONAL { ?rule dcterms:description ?description }
                    }
                """
            },
            
            "Analyze BigQuery data quality and completeness rules": {
                "description": "Shows business rules that define data quality standards and completeness requirements for BigQuery datasets",
                "sparql": """
                    PREFIX business: <http://business.org/schema#>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX dcterms: <http://purl.org/dc/terms/>
                    
                    SELECT ?rule ?label ?condition ?description WHERE {
                        ?rule a business:Rule .
                        ?rule business:category "data_quality" .
                        OPTIONAL { ?rule rdfs:label ?label }
                        OPTIONAL { ?rule business:condition ?condition }
                        OPTIONAL { ?rule dcterms:description ?description }
                    }
                """
            }
        }

    def create_query_result_visualization(self, results: List[Dict], query_type: str = "general") -> Tuple[List[Node], List[Edge]]:
        """Create a visualization from BigQuery SPARQL query results"""
        nodes = []
        edges = []
        
        if not results:
            return nodes, edges
        
        # Analyze the query results to determine visualization strategy
        result_keys = list(results[0].keys()) if results else []
        
        # Strategy 1: If results contain subject-predicate-object triples
        if 'subject' in result_keys and 'predicate' in result_keys and 'object' in result_keys:
            added_nodes = set()
            
            for result in results:
                subject = result.get('subject', '')
                predicate = result.get('predicate', '')
                obj = result.get('object', '')
                
                # Create subject node
                if subject and subject not in added_nodes:
                    subj_label = subject.split('#')[-1].replace('_', ' ').title()
                    subj_node = Node(
                        id=subject,
                        label=subj_label,
                        size=20,
                        color=self._get_node_color_from_uri(subject),
                        title=self._sanitize_title(f"BigQuery Subject: {subj_label}")
                    )
                    nodes.append(subj_node)
                    added_nodes.add(subject)
                
                # Create object node
                if obj and obj not in added_nodes and obj.startswith('http'):
                    obj_label = obj.split('#')[-1].replace('_', ' ').title()
                    obj_node = Node(
                        id=obj,
                        label=obj_label,
                        size=20,
                        color=self._get_node_color_from_uri(obj),
                        title=self._sanitize_title(f"BigQuery Object: {obj_label}")
                    )
                    nodes.append(obj_node)
                    added_nodes.add(obj)
                
                # Create edge
                if subject and obj and obj.startswith('http'):
                    pred_label = predicate.split('#')[-1].replace('_', ' ')
                    edge = Edge(
                        source=subject,
                        target=obj,
                        label=pred_label,
                        color='#666666'
                    )
                    edges.append(edge)
        
        # Strategy 2: If results contain entities (tables, rules, columns)
        elif any(key in result_keys for key in ['table', 'rule', 'column', 'pattern']):
            for i, result in enumerate(results):
                # Determine entity type and create appropriate node
                entity_uri = None
                entity_label = ""
                entity_type = "Default"
                
                if 'table' in result:
                    entity_uri = result['table']
                    entity_type = "Table"
                    entity_label = result.get('purpose', result.get('description', 'BigQuery Table')).split('/')[-1]
                elif 'rule' in result:
                    entity_uri = result['rule']
                    entity_type = "BusinessRule"
                    entity_label = result.get('name', result.get('description', 'BigQuery Rule')).split('/')[-1]
                elif 'column' in result:
                    entity_uri = result['column']
                    entity_type = "Column"
                    entity_label = result.get('name', 'BigQuery Column').split('/')[-1]
                elif 'pattern' in result:
                    entity_uri = result['pattern']
                    entity_type = "CanonicalJoinPattern"
                    entity_label = result.get('name', 'BigQuery Pattern').split('/')[-1]
                
                if entity_uri:
                    node = Node(
                        id=entity_uri,
                        label=entity_label[:20] + "..." if len(entity_label) > 20 else entity_label,
                        size=self.node_sizes.get(entity_type, 18),
                        color=self.node_colors.get(entity_type, '#DDA0DD'),
                        title=self._sanitize_title(f"BigQuery {entity_type}: {str(result)}")
                    )
                    nodes.append(node)
        
        # Strategy 3: Create simple nodes for any other results
        else:
            for i, result in enumerate(results):
                # Create a node for each result
                label = str(list(result.values())[0])[:20] if result.values() else f"BigQuery Result {i+1}"
                node = Node(
                    id=f"bq_result_{i}",
                    label=label,
                    size=15,
                    color='#4285F4',  # Google Blue for BigQuery results
                    title=self._sanitize_title(f"BigQuery Result: {str(result)}")
                )
                nodes.append(node)
        
        return nodes, edges

    def _get_node_color_from_uri(self, uri: str) -> str:
        """Get appropriate color for a BigQuery node based on its URI"""
        if 'table_' in uri or any(table in uri for table in ['agents', 'customers', 'policies', 'claims']):
            return self.node_colors['Table']
        elif 'column_' in uri:
            return self.node_colors['Column']
        elif 'business_rule_' in uri:
            return self.node_colors['BusinessRule']
        elif 'canonical_join_' in uri:
            return self.node_colors['CanonicalJoinPattern']
        elif 'dataset_' in uri or 'insurance_analytics' in uri:
            return self.node_colors['Dataset']
        else:
            return self.node_colors['Default']


def generate_ai_explanation_for_sparql_results(sparql_query: str, results: List[Dict], selected_query_name: str = None) -> str:
    """Generate AI explanation for SPARQL query results using Gemini"""
    try:
        # Import Gemini here to avoid circular imports
        import os
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        # Initialize Vertex AI
        project_id = os.getenv('GCP_PROJECT_ID', 'your-project-id')
        location = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
        model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash-lite')
        
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name)
        
        # Prepare the context for the AI
        results_summary = f"Query returned {len(results)} results"
        sample_results = str(results[:3]) if len(results) > 3 else str(results)
        
        query_context = f"Query Name: {selected_query_name}" if selected_query_name else "Custom SPARQL Query"
        
        prompt = f"""
You are an AI assistant analyzing BigQuery Knowledge Graph SPARQL query results. Please provide a clear, human-readable explanation of what these results show.

{query_context}

SPARQL Query:
{sparql_query}

Results Summary: {results_summary}
Sample Results: {sample_results}

Please explain:
1. What type of information this query was seeking
2. What the results tell us about the BigQuery insurance_analytics dataset
3. Any interesting patterns or insights from the data
4. The business relevance of these findings

Keep your explanation conversational, informative, and accessible to non-technical users. Focus on the business meaning rather than technical details.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Unable to generate AI explanation: {str(e)}. The query returned {len(results)} results from the BigQuery knowledge graph."


def render_kg_visualization_page():
    """Render the BigQuery Knowledge Graph Visualization page in Streamlit"""
    st.title("🕸️ Knowledge Graph Visualizer")
    st.markdown("Interactive exploration of the BigQuery SQL Knowledge Graph generated for the insurance_analytics dataset.")
    
    # Focus dropdown in main page
    bi_focus = st.selectbox(
        "Select Focus Area",
        [
            "All",
            "Business Rules",
            "Business Domain Vocabulary", 
            "Data Governance",
            "Synonym Networks"
        ],
        help="Choose which aspect of business intelligence to visualize"
    )
    
    st.divider()  # Visual separator
    
    # Initialize BigQuery visualizer
    kg_file_path = "bqkg/bq_knowledge_graph.ttl"
    
    try:
        visualizer = GBQ_KGVisualizer(kg_file_path)
        
        # Hardcoded configuration for Business Intelligence Dashboard
        config = Config(
            width=1200,  # Wider for dashboard
            height=800,  # Taller for dashboard
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#FF6B6B",  # Coral highlight for business focus
            collapsible=False,
            # Optimized physics for business intelligence layout
            barnesHut_gravitationalConstant=-3000,
            barnesHut_centralGravity=0.1,
            barnesHut_springLength=200,
            barnesHut_springConstant=0.05,
            barnesHut_damping=0.4
        )
        
        # Generate Business Intelligence Dashboard visualization
        st.subheader(f"🎯 Business Intelligence Dashboard - {bi_focus}")
        st.markdown(f"**Focus: {bi_focus} - Targeted view of business domain intelligence**")
        
        # Generate focused view
        nodes, edges = visualizer.create_business_intelligence_dashboard(focus_filter=bi_focus)
        
        # Render the graph
        if nodes:
            st.markdown("🔍 **Hover** over nodes for details • 🖱️ **Drag** to rearrange • 🔍 **Zoom** with mouse wheel")
            
            return_value = agraph(nodes=nodes, edges=edges, config=config)
            
            # Display selected node information
            if return_value:
                st.subheader("🔍 Selected Business Entity Details")
                
                try:
                    # Handle the return value safely - could be URI, node ID, or other format
                    if isinstance(return_value, str):
                        # Enhanced display for BI Dashboard
                        st.text(f"🎯 Selected Entity: {return_value}")
                        
                        # Determine entity type and show relevant business context
                        if "/business-rules/" in return_value:
                            st.success("📋 Business Rule Selected")
                        elif "/business-concepts/" in return_value:
                            st.success("💡 Business Concept Selected")
                        elif "/tables/" in return_value:
                            st.success("🗃️ Database Table Selected")
                        elif "/columns/" in return_value:
                            st.success("📊 Database Column Selected")

                        
                        # Try to get details for this node from the visualizer
                        node_details = visualizer.get_node_details_by_uri(return_value)
                        if node_details and 'error' not in node_details:
                            # Show enhanced details for BI Dashboard
                            with st.expander("📖 Detailed Business Intelligence", expanded=True):
                                st.json(node_details)
                        elif 'error' in node_details:
                            st.warning(f"Could not find detailed information for this node: {node_details['error']}")
                        else:
                            st.info("No additional details available for this node.")
                    elif isinstance(return_value, (dict, list)):
                        st.json(return_value)
                    else:
                        st.text(f"Selected: {str(return_value)}")
                except Exception as e:
                    st.error(f"Error displaying node details: {str(e)}")
                    st.text(f"Raw return value: {str(return_value)}")
        else:
            st.warning("No BigQuery nodes found for the selected criteria. Try adjusting your filters.")
        
        # BigQuery SPARQL Query Interface
        st.divider()
        st.header("🔍 BigQuery Custom SPARQL Query Interface")
        st.markdown("Execute custom SPARQL queries on the BigQuery knowledge graph with both text and visual results.")
        
        # Get BigQuery example queries with descriptions
        example_queries = visualizer.get_bigquery_example_sparql_queries()
        
        # Query selection and help
        col1, col2 = st.columns([3, 1])
        with col1:
            # Plain English dropdown
            query_options = ["Write Custom Query"] + list(example_queries.keys())
            selected_query = st.selectbox(
                "🗣️ Choose a BigQuery query in plain English:",
                query_options,
                help="Select a predefined BigQuery query described in plain English, or choose 'Write Custom Query'"
            )
        
        with col2:
            with st.expander("📚 BigQuery SPARQL Help"):
                st.markdown("""
                **Common Prefixes:**
                ```sparql
                PREFIX sqlkg: <http://example.com/sql-kg/ontology#>
                PREFIX sqldata: <http://example.com/sql-kg/data#>
                ```
                
                **BigQuery Entity Types:**
                - `sqlkg:Dataset` - BigQuery datasets
                - `sqlkg:Table` - BigQuery tables
                - `sqlkg:Column` - Table columns  
                - `sqlkg:BusinessRule` - Business rules
                - `sqlkg:CanonicalJoinPattern` - Join patterns
                
                **Properties:**
                - `sqlkg:hasName` - Entity name
                - `sqlkg:description` - Description
                - `sqlkg:businessPurpose` - Business purpose
                - `sqlkg:ruleCategory` - Rule category
                - `sqlkg:appliesTo` - Rule applies to table
                - `sqlkg:partOf` - Table part of dataset
                """)
        
        # Show description for selected query
        if selected_query != "Write Custom Query":
            st.info(f"📝 **BigQuery Query Description:** {example_queries[selected_query]['description']}")
        
        # SPARQL query text area
        if selected_query == "Write Custom Query":
            sparql_query = st.text_area(
                "✏️ Enter your BigQuery SPARQL query:",
                height=200,
                placeholder="""PREFIX sqlkg: <http://example.com/sql-kg/ontology#>

SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object
}
LIMIT 10""",
                help="Write your custom SPARQL query for BigQuery knowledge graph here"
            )
        else:
            # Pre-populate with selected query
            sparql_query = st.text_area(
                f"📝 BigQuery SPARQL Query: {selected_query}",
                value=example_queries[selected_query]['sparql'].strip(),
                height=200,
                help="You can modify this BigQuery query or use it as-is"
            )
        
        # Execute query button
        col1, col2 = st.columns([1, 3])
        with col1:
            execute_query = st.button("🚀 Execute BigQuery Query", type="primary")
        
        # Auto-execute for sample queries or manual execution for custom queries
        should_execute = False
        if selected_query != "Write Custom Query":
            # Auto-execute sample queries
            should_execute = True
            st.info("🚀 Auto-executing sample query...")
        elif execute_query and sparql_query.strip():
            # Manual execution for custom queries
            should_execute = True
        
        # Execute the query
        if should_execute and sparql_query.strip():
            with st.spinner("🔄 Executing BigQuery SPARQL query..."):
                results = visualizer.execute_sparql_query(sparql_query)
            
            if results:
                st.success(f"✅ BigQuery query executed successfully! Found **{len(results)}** results.")
                
                # Generate AI explanation for the results
                with st.spinner("🤖 Generating AI explanation of results..."):
                    explanation = generate_ai_explanation_for_sparql_results(
                        sparql_query, 
                        results, 
                        selected_query if selected_query != "Write Custom Query" else None
                    )
                
                # Display AI explanation
                st.subheader("🤖 AI Analysis of BigQuery Knowledge Graph Results")
                st.markdown(explanation)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("� Total Results", len(results))
                with col2:
                    st.metric("📋 Fields", len(results[0].keys()) if results else 0)
                with col3:
                    st.metric("💾 Data Size", f"{len(str(results))} chars")
                
                # Raw results in expandable section
                with st.expander("� View Raw BigQuery JSON Results"):
                    st.json(results)
                
            else:
                st.warning("❌ No BigQuery results found or query execution failed.")
                st.info("💡 **Tips for better BigQuery results:**\n- Check your SPARQL syntax\n- Ensure the prefixes are correct\n- Try one of the example BigQuery queries first")
        
        elif execute_query and not sparql_query.strip():
            st.error("⚠️ Please enter a BigQuery SPARQL query before executing.")
    
    except FileNotFoundError:
        st.error(f"BigQuery knowledge graph file not found: {kg_file_path}")
        st.info("Please ensure the BigQuery knowledge graph has been generated first by running the BigQuery text-to-SQL system.")
    
    except Exception as e:
        st.error(f"Error loading BigQuery knowledge graph: {str(e)}")
        st.info("Please check that the BigQuery knowledge graph file is valid and accessible.")

if __name__ == "__main__":
    render_kg_visualization_page()