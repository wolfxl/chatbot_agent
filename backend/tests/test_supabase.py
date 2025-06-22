"""
Simple test script for Supabase connection
"""
import asyncio
import os
import sys

# Add parent directory to path so we can import from database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from database.supabase_client import create_supabase_client

async def test_supabase_connection():
    """Test Supabase connection and basic operations"""
    
    # Check if environment variables are set
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        print("Please set these in your .env file or environment")
        return
    
    try:
        # Create client
        print("🔌 Creating Supabase client...")
        client = create_supabase_client(supabase_url, supabase_key)
        
        # Test connection
        print("🧪 Testing connection...")
        connection_result = await client.test_connection()
        print(f"Connection status: {connection_result['status']}")
        print(f"Message: {connection_result['message']}")
        
        if connection_result['status'] == 'connected':
            print("✅ Connection successful!")
            
            # Test basic operations
            print("\n📊 Testing basic operations...")
            
            # Get categories
            categories = await client.get_all_categories()
            print(f"Categories found: {len(categories)}")
            
            # Get organizations
            organizations = await client.get_organizations()
            print(f"Organizations found: {len(organizations)}")
            
            # Get locations
            locations = await client.get_all_locations()
            print(f"Locations found: {len(locations)}")
            
            # Get camps (limit to first few for testing)
            camps = await client.get_all_camps()
            print(f"Camps found: {len(camps)}")
            
            if camps:
                print(f"Sample camp: {camps[0].get('camp_name', 'Unknown')}")
            
        else:
            print("❌ Connection failed!")
            print(f"Error: {connection_result['message']}")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(test_supabase_connection())
