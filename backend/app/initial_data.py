import asyncio
import json
from sqlalchemy.orm import Session
from app.models.database import Role, Policy, User
from app.services import auth
from app.schemas import schemas
from app.core.config import settings

# Define initial roles
INITIAL_ROLES = [
    {"name": "admin", "description": "Super administrator with all system permissions."},
    {"name": "content_manager", "description": "Responsible for creating and managing RAG knowledge base content."},
    {"name": "standard_user", "description": "Standard user of the system with basic read-only and query permissions."},
]

# Define initial policies based on the new ABAC model
INITIAL_POLICIES = [
    # Admin Policy
    {
      "name": "Allow Admin Full Access",
      "description": "Grants administrators unrestricted access to all resources and actions.",
      "effect": "allow",
      "subjects": [{ "key": "user.roles", "operator": "in", "value": ["admin"] }],
      "actions": ["*"],
      "resources": [{ "key": "resource.type", "operator": "in", "value": ["*"] }],
      "query_conditions": []
    },
    # Content Manager Policies
    {
      "name": "Allow Content Manager to Manage Content (RAG, Files)",
      "effect": "allow",
      "subjects": [{ "key": "user.roles", "operator": "in", "value": ["content_manager"] }],
      "actions": ["create", "read", "update", "delete", "manage", "read_list", "read_files", "upload_file", "delete_file", "preview_file", "query"],
      "resources": [{ "key": "resource.type", "operator": "in", "value": ["rag_data", "rag_file"] }],
      "query_conditions": []
    },
    {
      "name": "Allow Content Manager Read-Only Access to Core Data Lists",
      "effect": "allow",
      "subjects": [{ "key": "user.roles", "operator": "in", "value": ["content_manager"] }],
      "actions": ["read_list"],
      "resources": [{ "key": "resource.type", "operator": "in", "value": ["user", "role", "policy"] }],
      "query_conditions": []
    },
    # Standard User Policies
    {
      "name": "Allow Standard User to Use Chat",
      "effect": "allow",
      "subjects": [{ "key": "user.roles", "operator": "in", "value": ["standard_user"] }],
      "actions": ["create", "read"],
      "resources": [{ "key": "resource.type", "operator": "in", "value": ["chat"] }],
      "query_conditions": []
    },
    {
      "name": "Allow User to Manage Their Own Profile",
      "description": "Allows any user to view and update their own user object.",
      "effect": "allow",
      "subjects": [],
      "actions": ["read", "update"],
      "resources": [{ "key": "resource.type", "operator": "in", "value": ["user"] }],
      "query_conditions": [
        {
          "resource_attribute": "owner_id",
          "operator": "eq",
          "subject_attribute": "id"
        }
      ]
    },
    {
      "name": "Allow User to Manage Their Own Files (List & CRUD)",
      "effect": "allow",
      "subjects": [],
      "actions": ["create", "read", "update", "delete", "read_list"],
      "resources": [{ "key": "resource.type", "operator": "in", "value": ["file"] }],
      "query_conditions": [
          {
            "resource_attribute": "user_id",
            "operator": "eq",
            "subject_attribute": "id"
          }
      ]
    }
]

async def initialize_data(db: Session):
    """
    Initializes the database with default roles, policies, and an admin user
    if they do not already exist.
    """
    # Initialize Roles
    for role_data in INITIAL_ROLES:
        role = db.query(Role).filter(Role.name == role_data["name"]).first()
        if not role:
            new_role = Role(**role_data)
            db.add(new_role)
            print(f"Created role: {new_role.name}")


    # Synchronize policies: Update existing policies or create them if they don't exist
    print("Synchronizing policies...")
    for policy_data in INITIAL_POLICIES:
        policy = db.query(Policy).filter(Policy.name == policy_data["name"]).first()
        # To ensure data integrity and handle schema changes, we adopt a delete-and-recreate strategy.
        if policy:
            db.delete(policy)
            # Flush the session to ensure the delete operation is executed before the add.
            # This prevents IntegrityError for unique constraints.
            db.flush()
            print(f"Deleted existing policy: {policy.name} to be recreated.")
        
        # Create the policy from the latest definition
        new_policy = Policy(**policy_data)
        db.add(new_policy)
        print(f"Created/Recreated policy: {policy_data['name']}")
            
    print("Policy synchronization complete.")
    db.commit()  # Commit policy changes first to isolate them

    # Create a default admin user if the setting is enabled
    if settings.CREATE_DEFAULT_ADMIN:
        # Use a separate query for the user to ensure the session is fresh after commit
        admin_user = db.query(User).filter(User.username == settings.DEFAULT_ADMIN_USERNAME).first()
        admin_role = db.query(Role).filter(Role.name == "admin").first()

        if not admin_user:
            if admin_role:
                user_in = schemas.UserCreate(
                    username=settings.DEFAULT_ADMIN_USERNAME,
                    email=settings.DEFAULT_ADMIN_EMAIL,
                    password=settings.DEFAULT_ADMIN_PASSWORD,
                    department="IT", # Default department
                    phone="0000000000" # Default phone
                )
                # Note: auth.create_user handles password hashing and may already assign
                # the admin role automatically when this is the first user.
                new_admin_user = auth.create_user(db, user_create=user_in, is_active=True)

                # Make role assignment idempotent. Without this guard, the app can try to
                # insert the same (user_id, role_id) pair twice into user_role_association.
                if not any(role.id == admin_role.id for role in new_admin_user.roles):
                    new_admin_user.roles.append(admin_role)
                    db.add(new_admin_user)
                    db.commit()

                print(f"Created default admin user: {new_admin_user.username}")
        else:
            # If the admin user already exists, make sure they have the admin role.
            # This keeps startup safe after a partially successful previous initialization.
            if admin_role and not any(role.id == admin_role.id for role in admin_user.roles):
                admin_user.roles.append(admin_role)
                db.add(admin_user)
                db.commit()
                print(f"Assigned admin role to existing default admin user: {admin_user.username}")

if __name__ == "__main__":
    from app.models.database import get_db, create_database_tables
    
    print("Running data initialization script...")
    # Create tables if they don't exist
    create_database_tables()
    
    # Get a database session
    db_session = next(get_db())
    
    try:
        asyncio.run(initialize_data(db_session))
        print("Data initialization complete.")
    finally:
        db_session.close()