"""
Gemini Access Test

This module tests access to Google Cloud Vertex AI Gemini models using
default gcloud authentication (Application Default Credentials).

Usage:
    python GeminiAccessTest.py

Requirements:
    - gcloud CLI installed and authenticated
    - Default application credentials configured
    - Access to specified GCP project and Vertex AI
"""

import os
import sys
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeminiAccessTest:
    """
    Test class for validating Gemini model access using default gcloud authentication
    """
    
    def __init__(self):
        """Initialize the test with configuration from environment variables"""
        self.project_id = None
        self.location = None
        self.model_name = None
        self.model = None
        
    def load_configuration(self) -> bool:
        """
        Load configuration from .env file
        
        Returns:
            bool: True if configuration loaded successfully, False otherwise
        """
        try:
            # Load environment variables from .env file
            load_dotenv()
            
            # Get required configuration
            self.project_id = os.getenv('GCP_PROJECT_ID')
            self.location = os.getenv('VERTEX_AI_LOCATION')
            self.model_name = os.getenv('GEMINI_MODEL_NAME')
            
            # Validate required variables
            missing_vars = []
            if not self.project_id:
                missing_vars.append('GCP_PROJECT_ID')
            if not self.location:
                missing_vars.append('VERTEX_AI_LOCATION')
            if not self.model_name:
                missing_vars.append('GEMINI_MODEL_NAME')
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                return False
            
            logger.info(f"Configuration loaded successfully:")
            logger.info(f"  Project ID: {self.project_id}")
            logger.info(f"  Location: {self.location}")
            logger.info(f"  Model Name: {self.model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def test_authentication(self) -> bool:
        """
        Test default gcloud authentication
        
        Returns:
            bool: True if authentication is working, False otherwise
        """
        try:
            logger.info("Testing default authentication...")
            
            # Get default credentials
            credentials, project = default()
            
            if credentials:
                logger.info("✓ Default credentials found")
                if project:
                    logger.info(f"  Default project: {project}")
                else:
                    logger.info("  No default project set in credentials")
                return True
            else:
                logger.error("✗ No default credentials found")
                return False
                
        except DefaultCredentialsError as e:
            logger.error(f"✗ Default credentials error: {e}")
            logger.error("Please run: gcloud auth application-default login")
            return False
        except Exception as e:
            logger.error(f"✗ Authentication test failed: {e}")
            return False
    
    def initialize_vertex_ai(self) -> bool:
        """
        Initialize Vertex AI with the specified project and location
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Vertex AI...")
            
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            logger.info("✓ Vertex AI initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize Vertex AI: {e}")
            return False
    
    def test_model_access(self) -> bool:
        """
        Test access to the specified Gemini model
        
        Returns:
            bool: True if model access successful, False otherwise
        """
        try:
            logger.info(f"Testing access to model: {self.model_name}")
            
            # Initialize the model
            self.model = GenerativeModel(self.model_name)
            
            logger.info("✓ Model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to access model: {e}")
            return False
    
    def test_model_response(self) -> bool:
        """
        Test model response with a simple prompt
        
        Returns:
            bool: True if model responds successfully, False otherwise
        """
        try:
            logger.info("Testing model response...")
            
            # Simple test prompt
            test_prompt = "Hello! Please respond with 'Access test successful' to confirm you are working."
            
            logger.info(f"Sending test prompt: {test_prompt}")
            
            # Generate response
            response = self.model.generate_content(test_prompt)
            
            if response and response.text:
                logger.info("✓ Model responded successfully")
                logger.info(f"Response: {response.text.strip()}")
                return True
            else:
                logger.error("✗ Model response was empty")
                return False
                
        except Exception as e:
            logger.error(f"✗ Model response test failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """
        Run all tests in sequence
        
        Returns:
            bool: True if all tests pass, False if any test fails
        """
        print("=" * 60)
        print("GEMINI ACCESS TEST - COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        tests = [
            ("Configuration Loading", self.load_configuration),
            ("Authentication Test", self.test_authentication),
            ("Vertex AI Initialization", self.initialize_vertex_ai),
            ("Model Access Test", self.test_model_access),
            ("Model Response Test", self.test_model_response)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            
            try:
                result = test_func()
                if result:
                    print(f"✓ {test_name}: PASSED")
                else:
                    print(f"✗ {test_name}: FAILED")
                    all_passed = False
                    break  # Stop on first failure
                    
            except Exception as e:
                print(f"✗ {test_name}: ERROR - {e}")
                all_passed = False
                break
        
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 ALL TESTS PASSED - Gemini access is working correctly!")
            print("✓ Authentication configured properly")
            print("✓ Vertex AI access confirmed")
            print("✓ Gemini model is accessible and responsive")
        else:
            print("❌ TESTS FAILED - Please check the errors above")
            print("\nTroubleshooting steps:")
            print("1. Ensure gcloud CLI is installed: gcloud --version")
            print("2. Login with application default credentials: gcloud auth application-default login")
            print("3. Verify project access: gcloud config get-value project")
            print("4. Check Vertex AI API is enabled in your project")
            print("5. Verify your account has Vertex AI permissions")
        
        print("=" * 60)
        return all_passed


def main():
    """Main function to run the Gemini access test"""
    tester = GeminiAccessTest()
    success = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()