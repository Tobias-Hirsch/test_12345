from typing import Dict, Any, Callable, List
from app.models.database import Policy
from app.services import abac_functions # Hinweis

class ABACPolicyEvaluator:
    def __init__(self):
        self.function_registry: Dict[str, Callable[..., bool]] = {
            "is_resource_owner": abac_functions.is_resource_owner,
            "is_within_working_hours": abac_functions.is_within_working_hours,
            # RegistrierenKommentar
        }

    def _evaluate_condition(self, condition: Dict[str, Any], attributes: Dict[str, Any]) -> bool:
        """
        Hinweis
        """
        if "function" in condition:
            func_name = condition["function"]
            if func_name not in self.function_registry:
                raise ValueError(f"Unknown ABAC function: {func_name}")
            func = self.function_registry[func_name]
            args = [self._resolve_attribute_path(attr_path, attributes) for attr_path in condition.get("args", [])]
            return func(*args)
        else:
            attribute_path = condition["attribute"]
            operator = condition["operator"]
            value = condition["value"]

            actual_value = self._resolve_attribute_path(attribute_path, attributes)
            print(f"DEBUG: Evaluating condition: attribute_path='{attribute_path}', operator='{operator}', value='{value}', actual_value='{actual_value}'")

            if operator == "equals":
                return actual_value == value
            elif operator == "in":
                # Ensure actual_value is iterable for 'in' operator
                if not isinstance(value, list):
                    print(f"WARNING: 'in' operator expects a list for 'value', but got {type(value)}")
                    return False
                # Check if any item in 'value' list is present in 'actual_value' list
                # This assumes actual_value is also a list or iterable
                if not isinstance(actual_value, (list, set, tuple)):
                    print(f"WARNING: 'in' operator expects 'actual_value' to be iterable, but got {type(actual_value)}")
                    return False
                return any(item in actual_value for item in value)
            elif operator == "greater_than":
                return actual_value > value
            elif operator == "greater_than_or_equal":
                return actual_value >= value
            elif operator == "less_than":
                return actual_value < value
            elif operator == "less_than_or_equal":
                return actual_value <= value
            elif operator == "not_equals":
                return actual_value != value
            # Kommentar
            else:
                raise ValueError(f"Unknown ABAC operator: {operator}")

    def _resolve_attribute_path(self, path: str, attributes: Dict[str, Any]) -> Any:
        """
        Resolves an attribute value from a nested dictionary based on a dot-separated path.
        Handles lists of objects by extracting the attribute from each object in the list.
        e.g., "user.roles.name" with attributes {'user': {'roles': [{'name': 'admin'}, {'name': 'user'}]}}
        will return ['admin', 'user'].
        """
        parts = path.split('.')
        current_value = attributes
        for i, part in enumerate(parts):
            if isinstance(current_value, dict):
                current_value = current_value.get(part)
            elif isinstance(current_value, list):
                # If we encounter a list, we try to resolve the rest of the path for each item in the list.
                remaining_path = '.'.join(parts[i:])
                # This list comprehension recursively calls the resolver for each item in the list
                # with the remaining part of the path.
                resolved_values = [self._resolve_attribute_path(remaining_path, item) for item in current_value if isinstance(item, dict)]
                # We filter out None values which indicate the path was not found in an item.
                final_values = [v for v in resolved_values if v is not None]
                # If the values themselves are lists (from deeper nesting), flatten them.
                if any(isinstance(i, list) for i in final_values):
                    return [item for sublist in final_values for item in sublist]
                return final_values if final_values else None

            if current_value is None:
                print(f"DEBUG: Attribute path '{path}' not found at part '{part}'. Returning None.")
                return None
        
        print(f"DEBUG: Resolved attribute path '{path}' to value: {current_value}")
        return current_value

    def _evaluate_rules(self, rules_logic: Dict[str, Any], attributes: Dict[str, Any]) -> bool:
        """
        Hinweis
        """
        operator = rules_logic.get("operator", "AND")
        rules = rules_logic.get("rules", [])

        if operator == "AND":
            for rule in rules:
                if not isinstance(rule, dict):
                    print(f"WARNING: Invalid rule format in AND block. Expected a dictionary, but got {type(rule)}. Rule: {rule}")
                    return False  # Fail closed for safety

                if "rules" in rule:  # Nested logic
                    if not self._evaluate_rules(rule, attributes):
                        return False
                elif "function" in rule:  # Function call
                    if not self._evaluate_condition(rule, attributes):
                        return False
                else:  # Simple condition
                    if not self._evaluate_condition(rule, attributes):
                        return False
            return True
        elif operator == "OR":
            for rule in rules:
                if not isinstance(rule, dict):
                    print(f"WARNING: Invalid rule format in OR block. Expected a dictionary, but got {type(rule)}. Rule: {rule}")
                    continue  # In OR, we can skip invalid rules

                if "rules" in rule:  # Nested logic
                    if self._evaluate_rules(rule, attributes):
                        return True
                elif "function" in rule:  # Function call
                    if self._evaluate_condition(rule, attributes):
                        return True
                else:  # Simple condition
                    if self._evaluate_condition(rule, attributes):
                        return True
            return False
        else:
            raise ValueError(f"Unknown logical operator: {operator}")

    def _check_policy_match(self, policy: Policy, attributes: Dict[str, Any], action: str, resource_type: str, resource_id: str) -> bool:
        """
        Checks if a single policy matches the given attributes, action, and resource.
        This logic is now aligned with QueryFilterService._is_policy_applicable.
        """
        print(f"\n--- Evaluating Policy: '{policy.name}' for action '{action}' on resource '{resource_type}' ---")

        # 1. Check Actions
        action_match = "*" in policy.actions or action in policy.actions
        if not action_match:
            print(f"    [FAIL] Action Match: action '{action}' not in {policy.actions}")
            return False
        print(f"    [PASS] Action Match: action '{action}' in {policy.actions}")

        # 2. Check Resources
        resource_match = False
        for res in policy.resources:
            if not isinstance(res, dict): continue
            res_key, res_op, res_val = res.get("key"), res.get("operator"), res.get("value", [])
            if res_key == "resource.type" and res_op == "in" and ("*" in res_val or resource_type in res_val):
                resource_match = True
                break
            if res_key == "resource.type" and res_op == "eq" and resource_type == res_val:
                resource_match = True
                break
        if not resource_match:
            print(f"    [FAIL] Resource Match: resource_type '{resource_type}' not in {policy.resources}")
            return False
        print(f"    [PASS] Resource Match: resource_type '{resource_type}' in {policy.resources}")

        # 3. Check Subjects
        subject_match = False
        if not policy.subjects:
            subject_match = True
        else:
            for sub in policy.subjects:
                if not isinstance(sub, dict): continue
                key, operator, value = sub.get("key"), sub.get("operator"), sub.get("value")
                
                user_val = self._resolve_attribute_path(key, attributes)
                print(f"    - Subject Eval: key='{key}', operator='{operator}', required_value='{value}', user_value='{user_val}'")

                if user_val is None:
                    continue

                if operator == "eq" and user_val == value:
                    subject_match = True
                    break
                if operator == "in" and isinstance(user_val, list) and isinstance(value, list) and any(item in value for item in user_val):
                    subject_match = True
                    break
                if operator == "contains" and isinstance(user_val, list) and value in user_val:
                    subject_match = True
                    break
        
        if not subject_match:
            print(f"    [FAIL] Subject Match: User attributes do not match policy subjects {policy.subjects}")
            return False
        print(f"    [PASS] Subject Match: User attributes match policy subjects.")

        # 4. Check Conditions
        if policy.query_conditions:
            for cond in policy.query_conditions:
                resource_attr_name = cond.get("resource_attribute")
                subject_attr_name = cond.get("subject_attribute")
                resource_attr_value = self._resolve_attribute_path(f"resource.{resource_attr_name}", attributes)
                subject_attr_value = self._resolve_attribute_path(f"user.{subject_attr_name}", attributes)

                print(f"    - Condition Eval: resource_attr='{resource_attr_value}', subject_attr='{subject_attr_value}'")

                if resource_attr_value is None or subject_attr_value is None:
                    print(f"    [FAIL] Condition Match: Could not resolve attributes for condition.")
                    return False
                
                if cond.get("operator") == "eq" and resource_attr_value != subject_attr_value:
                    print(f"    [FAIL] Condition Match: {resource_attr_value} != {subject_attr_value}")
                    return False
            print(f"    [PASS] Condition Match: All conditions met.")

        print(f"--- POLICY '{policy.name}' MATCHED ---")
        return True

    def evaluate(self, policies: List[Policy], attributes: Dict[str, Any], action: str, resource_type: str, resource_id: Any = None) -> bool:
        """
        Evaluates if an action on a resource is permitted, based on a set of policies and attributes.
        It follows a "deny-overrides" model.
        """
        # Filter active policies
        active_policies = [p for p in policies if p.is_active]

        # 1. Evaluate DENY policies. If any single deny policy matches, access is forbidden immediately.
        for policy in active_policies:
            if policy.effect == 'deny':
                try:
                    if self._check_policy_match(policy, attributes, action, resource_type, resource_id):
                        print(f"DEBUG: DENY policy '{policy.name}' matched. Access DENIED.")
                        return False # Deny overrides everything
                except Exception as e:
                    print(f"ERROR evaluating DENY policy '{policy.name}': {e}")
                    continue # Skip malformed policy

        # 2. If no deny policies matched, evaluate ALLOW policies. At least one allow policy must match.
        for policy in active_policies:
            if policy.effect == 'allow':
                try:
                    if self._check_policy_match(policy, attributes, action, resource_type, resource_id):
                        print(f"DEBUG: ALLOW policy '{policy.name}' matched. Access GRANTED.")
                        return True # Found a matching allow policy
                except Exception as e:
                    print(f"ERROR evaluating ALLOW policy '{policy.name}': {e}")
                    continue # Skip malformed policy
        
        # If we get here, it means no deny policies matched and no allow policies matched.
        print(f"DEBUG: No matching ALLOW policy found. Access DENIED by default.")
        return False