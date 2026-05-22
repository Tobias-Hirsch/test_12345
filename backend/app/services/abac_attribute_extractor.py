from datetime import datetime
from typing import Dict, Any, Optional
import redis
from sqlalchemy.orm import Session
from app.models.database import User, Policy # Import Policy model
from app.core.config import settings # Assuming settings contains LDAP_AUTH_ENABLE
 
class ABACAttributeExtractor:
    def __init__(self, db: Session, redis_client: redis.Redis):
        self.db = db
        self.redis_client = redis_client

    def get_user_attributes(self, user: User) -> Dict[str, Any]:
        """
        Extracts attributes from the user object.
        """
        # Ensure roles are loaded. This might be redundant if already eager-loaded, but it's safe.
        if not hasattr(user, 'roles'):
             # This is a fallback, ideally the user object should come with roles pre-loaded.
            user = self.db.query(User).filter_by(id=user.id).one()

        user_attributes = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "phone": user.phone,
            "department": user.department,
            "is_active": user.is_active,
            "security_level": user.security_level,
            "roles": [role.name for role in user.roles],
        }
        return user_attributes

    def get_resource_attributes(self, resource_obj: Any, resource_type: Optional[str] = None, resource_id: Optional[Any] = None) -> Dict[str, Any]:
        """
        Extracts attributes from a resource object or from explicit type/id.
        """
        if resource_obj:
            res_type = resource_obj.__class__.__name__.lower()
            res_id = getattr(resource_obj, 'id', None)
            owner_id = getattr(resource_obj, 'owner_id', None)
        else:
            res_type = resource_type
            res_id = resource_id
            owner_id = None # Cannot determine owner without the object

        return {
            "type": res_type,
            "id": res_id,
            "owner_id": owner_id
        }

    def get_action_attributes(self, action_type: str) -> Dict[str, Any]:
        """
        Hinweis
        """
        return {"type": action_type}

    def get_environment_attributes(self) -> Dict[str, Any]:
        """
        Extracts environmental attributes.
        """
        return {"current_time": datetime.now()}

    def get_all_attributes(
        self,
        user: User,
        action_type: str,
        resource_obj: Optional[Any] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Gathers all attributes for policy evaluation.
        """
        attributes = {
            "user": self.get_user_attributes(user),
            "resource": self.get_resource_attributes(resource_obj, resource_type, resource_id),
            "action": self.get_action_attributes(action_type),
            "environment": self.get_environment_attributes(),
        }
        return attributes