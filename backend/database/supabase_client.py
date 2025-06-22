"""
Standalone Supabase client for database operations
"""
import os
from typing import List, Dict, Optional, Any
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Supabase database client for summer camp data"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize Supabase client
        
        Args:
            supabase_url: Supabase project URL (defaults to environment variable)
            supabase_key: Supabase API key (defaults to environment variable)
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key must be provided or set as environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test database connection
        
        Returns:
            Connection status information
        """
        try:
            # Simple query to test connection
            result = self.client.table("organizations").select("id").limit(1).execute()
            return {
                "status": "connected",
                "message": "Successfully connected to Supabase",
                "data_count": len(result.data) if result.data else 0
            }
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}"
            }
    
    async def get_all_camps(self) -> List[Dict[str, Any]]:
        """
        Get all camps with related data
        
        Returns:
            List of camps with organization and session information
        """
        try:
            result = self.client.table('camps').select("""
                *,
                organizations(name, email, contact),
                camp_sessions(
                    *,
                    locations(name, city, state, address, formatted_address, latitude, longitude)
                ),
                camp_categories(
                    categories(name)
                )
            """).execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Error getting all camps: {e}")
            return []
    
    async def get_camp_by_id(self, camp_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific camp by ID
        
        Args:
            camp_id: Camp ID to retrieve
            
        Returns:
            Camp data or None if not found
        """
        try:
            result = self.client.table('camps').select("""
                *,
                organizations(name, email, contact),
                camp_sessions(
                    *,
                    locations(name, city, state, address, formatted_address, latitude, longitude)
                ),
                camp_categories(
                    categories(name)
                )
            """).eq('id', camp_id).execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting camp by ID {camp_id}: {e}")
            return None
    
    async def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        Get all camp categories
        
        Returns:
            List of categories
        """
        try:
            result = self.client.table('categories').select('*').execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    async def get_all_locations(self) -> List[Dict[str, Any]]:
        """
        Get all camp locations
        
        Returns:
            List of locations
        """
        try:
            result = self.client.table('locations').select('*').execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting locations: {e}")
            return []
    
    async def get_organizations(self) -> List[Dict[str, Any]]:
        """
        Get all organizations
        
        Returns:
            List of organizations
        """
        try:
            result = self.client.table('organizations').select('*').execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting organizations: {e}")
            return []
    
    async def search_camps_basic(self, **filters) -> List[Dict[str, Any]]:
        """
        Basic camp search with simple filters
        
        Args:
            **filters: Filter parameters (min_grade, max_grade, min_price, max_price, etc.)
            
        Returns:
            List of matching camps
        """
        try:
            query = self.client.table('camps').select("""
                *,
                organizations(name, email, contact),
                camp_sessions(
                    *,
                    locations(name, city, state, address, formatted_address, latitude, longitude)
                ),
                camp_categories(
                    categories(name)
                )
            """)
            
            # Apply filters
            if filters.get('min_grade') is not None:
                query = query.gte('min_grade', filters['min_grade'])
            if filters.get('max_grade') is not None:
                query = query.lte('max_grade', filters['max_grade'])
            if filters.get('min_price') is not None:
                query = query.gte('price', filters['min_price'])
            if filters.get('max_price') is not None:
                query = query.lte('price', filters['max_price'])
            
            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"Error in basic camp search: {e}")
            return []


# Global Supabase client instance
def create_supabase_client(supabase_url: str = None, supabase_key: str = None) -> SupabaseClient:
    """
    Factory function to create Supabase client
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        
    Returns:
        Configured SupabaseClient instance
    """
    return SupabaseClient(supabase_url, supabase_key)


# Default client instance (uses environment variables)
try:
    supabase_client = SupabaseClient()
except ValueError as e:
    logger.warning(f"Could not create default Supabase client: {e}")
    supabase_client = None
