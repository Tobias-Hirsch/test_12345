import os
from ldap3 import Server, Connection, SAFE_SYNC, SYNC, ALL, SUBTREE
from ldap3.core.exceptions import LDAPException, LDAPBindError, LDAPSocketOpenError
import logging
from ..core.config import settings  # Global import for configuration settings

logger = logging.getLogger(__name__)

class LDAPService:
    def __init__(self):
        self.server_uri = settings.LDAP_SERVER
        self.base_dn = settings.LDAP_BASE_DN
        self.bind_dn = settings.LDAP_BIND_DN
        self.bind_password = settings.LDAP_BIND_PASSWORD
        self.user_search_base = settings.LDAP_USER_SEARCH_BASE
        self.user_search_filter = settings.LDAP_USER_SEARCH_FILTER
        self.group_search_base = settings.LDAP_GROUP_SEARCH_BASE
        self.group_search_filter = settings.LDAP_GROUP_SEARCH_FILTER
        self.user_id_attr = settings.LDAP_USER_ID_ATTRIBUTE
        self.user_display_name_attr = settings.LDAP_USER_DISPLAY_NAME_ATTRIBUTE
        self.user_email_attr = settings.LDAP_USER_EMAIL_ATTRIBUTE
        self.user_department_attr = settings.LDAP_USER_DEPARTMENT_ATTRIBUTE
        self.user_security_level_attr = settings.LDAP_USER_SECURITY_LEVEL_ATTRIBUTE
        self.user_account_control_attr = settings.LDAP_USER_ACCOUNT_CONTROL_ATTRIBUTE
        self.use_ssl = settings.LDAP_USE_SSL
        self.validate_cert = settings.LDAP_VALIDATE_CERT

        self.server = Server(
            self.server_uri,
            use_ssl=self.use_ssl,
            get_info=ALL,
            tls=None if not self.use_ssl else self._get_tls_config()
        )
        # For simplicity, we'll manage a single connection for now.
        # For production, consider a connection pool (e.g., using a queue or a dedicated pool library).
        self._connection = None

    def _get_tls_config(self):
        # In a real application, you would load your CA certificates here.
        # For now, we'll use a basic TLS config.
        # If validate_cert is False, it means we are not validating the server certificate.
        # This is NOT recommended for production.
        from ldap3 import Tls
        return Tls(validate=self.validate_cert)

    def get_connection(self):
        if self._connection is None or not self._connection.bound:
            try:
                self._connection = Connection(
                    self.server,
                    user=self.bind_dn,
                    password=self.bind_password,
                    auto_bind=True,
                    client_strategy=SAFE_SYNC,
                    read_only=True
                )
                logger.info("LDAP connection established and bound.")
            except LDAPSocketOpenError as e:
                logger.error(f"Failed to open LDAP socket: {e}")
                raise ConnectionError(f"Fehler bei der Verarbeitung{e}")
            except LDAPBindError as e:
                logger.error(f"Failed to bind to LDAP server with provided credentials: {e}")
                raise ConnectionError(f"LDAP Fehler bei der Verarbeitung{e}")
            except LDAPException as e:
                logger.error(f"LDAP connection or binding error: {e}")
                raise ConnectionError(f"LDAP Fehler bei der Verarbeitung{e}")
        return self._connection

    def authenticate_user(self, username: str, password: str) -> bool:
        conn = None
        try:
            # Attempt to bind as the user to authenticate
            user_dn = self._get_user_dn(username)
            if not user_dn:
                logger.warning(f"User DN not found for username: {username}")
                return False

            conn = Connection(self.server, user=user_dn, password=password, auto_bind=True)
            if conn.bind():
                logger.info(f"User '{username}' authenticated successfully via LDAP.")
                return True
            else:
                logger.warning(f"User '{username}' LDAP authentication failed. Error: {conn.result}")
                return False
        except LDAPBindError as e:
            logger.warning(f"LDAP bind error for user '{username}': {e}")
            return False
        except LDAPException as e:
            logger.error(f"LDAP authentication error for user '{username}': {e}")
            return False
        finally:
            if conn and conn.bound:
                conn.unbind()

    def _get_user_dn(self, username: str) -> str | None:
        conn = self.get_connection()
        search_filter = self.user_search_filter.format(username=username)
        conn.search(
            search_base=self.user_search_base,
            search_filter=search_filter,
            search_scope=SUBTREE,
            attributes=[self.user_id_attr]
        )
        if conn.entries:
            return conn.entries[0].entry_dn
        return None

    def get_user_attributes(self, username: str) -> dict | None:
        conn = self.get_connection()
        search_filter = self.user_search_filter.format(username=username)
        attributes = [
            self.user_id_attr,
            self.user_display_name_attr,
            self.user_email_attr,
            self.user_department_attr,
            self.user_security_level_attr,
            self.user_account_control_attr,
            'memberOf' # To get group memberships
        ]
        conn.search(
            search_base=self.user_search_base,
            search_filter=search_filter,
            search_scope=SUBTREE,
            attributes=attributes
        )
        if conn.entries:
            entry = conn.entries[0]
            user_attrs = {
                "username": str(getattr(entry, self.user_id_attr, [''])[0]),
                "display_name": str(getattr(entry, self.user_display_name_attr, [''])[0]),
                "email": str(getattr(entry, self.user_email_attr, [''])[0]),
                "department": str(getattr(entry, self.user_department_attr, [''])[0]),
                "security_level": int(getattr(entry, self.user_security_level_attr, [0])[0]),
                "user_account_control": int(getattr(entry, self.user_account_control_attr, [0])[0]),
                "member_of": [str(group) for group in getattr(entry, 'memberOf', [])]
            }
            return user_attrs
        return None

    def parse_user_account_control(self, control_value: int) -> dict:
        """
        Parses the userAccountControl attribute to determine account status.
        Common flags:
        - ACCOUNTDISABLE (0x0002): Account is disabled.
        - NORMAL_ACCOUNT (0x0200): Typical user account.
        - DONT_EXPIRE_PASSWORD (0x10000): Password never expires.
        """
        is_disabled = bool(control_value & 0x0002)
        return {
            "is_disabled": is_disabled,
            "is_active": not is_disabled
        }

    # Placeholder for group members if needed, though memberOf is usually sufficient
    def get_group_members(self, group_dn: str) -> list[str]:
        conn = self.get_connection()
        conn.search(
            search_base=group_dn,
            search_filter='(objectClass=*)', # Search for any object within the group DN
            search_scope=SUBTREE,
            attributes=['member'] # Get members of the group
        )
        members = []
        if conn.entries:
            for entry in conn.entries:
                if 'member' in entry:
                    members.extend([str(m) for m in entry.member])
        return members

# Example usage (for testing purposes, not part of the main app flow)
# if __name__ == "__main__":
#     # Ensure .env is loaded for testing
#     load_dotenv()

#     ldap_service = LDAPService()

#     # Test connection and bind
#     try:
#         conn = ldap_service.get_connection()
#         print("Successfully connected and bound to LDAP server.")
#         conn.unbind() # Unbind after test
#     except ConnectionError as e:
#         print(f"Connection test failed: {e}")

#     # Test user authentication
#     test_username = "your_test_ad_username" # Replace with a valid AD username
#     test_password = "your_test_ad_password" # Replace with the valid password

#     print(f"\nAttempting to authenticate user: {test_username}")
#     if ldap_service.authenticate_user(test_username, test_password):
#         print(f"User '{test_username}' authenticated successfully.")
#     else:
#         print(f"User '{test_username}' authentication failed.")

#     # Test getting user attributes
#     print(f"\nAttempting to get attributes for user: {test_username}")
#     user_attrs = ldap_service.get_user_attributes(test_username)
#     if user_attrs:
#         print("User Attributes:")
#         for key, value in user_attrs.items():
#             print(f"  {key}: {value}")
        
#         account_status = ldap_service.parse_user_account_control(user_attrs.get(ldap_service.user_account_control_attr, 0))
#         print(f"  Account Status (parsed): {account_status}")
#     else:
#         print(f"Could not retrieve attributes for user: {test_username}")

    # Test getting group members (example, replace with a real group DN)
    # test_group_dn = "CN=YourGroup,OU=Groups,DC=yourdomain,DC=com"
    # print(f"\nAttempting to get members of group: {test_group_dn}")
    # group_members = ldap_service.get_group_members(test_group_dn)
    # if group_members:
    #     print("Group Members:")
    #     for member in group_members:
    #         print(f"  {member}")
    # else:
    #     print(f"Could not retrieve members for group: {test_group_dn}")