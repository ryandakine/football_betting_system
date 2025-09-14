#!/usr/bin/env python3
"""
N8N Workflow JSON Fixer
Fixes common issues that cause "propertyValues[itemName] is not iterable" errors
"""

import json
import os
import sys
from typing import Any, Dict, List


def fix_n8n_workflow(workflow_data: dict[str, Any]) -> dict[str, Any]:
    """Fix common n8n workflow issues"""

    # Remove problematic metadata
    problematic_keys = [
        "pinData",
        "staticData",
        "tags",
        "triggerCount",
        "updatedAt",
        "versionId",
        "meta",
        "id",
    ]

    for key in problematic_keys:
        if key in workflow_data:
            del workflow_data[key]

    # Ensure nodes have proper structure
    if "nodes" in workflow_data:
        for i, node in enumerate(workflow_data["nodes"]):
            # Add unique ID if missing
            if "id" not in node:
                node["id"] = f"node-{i}"

            # Fix parameters structure
            if "parameters" in node:
                # Remove any problematic nested structures
                if "value" in node["parameters"]:
                    if isinstance(node["parameters"]["value"], dict):
                        # Clean up value object
                        clean_value = {}
                        for k, v in node["parameters"]["value"].items():
                            if isinstance(v, str) and v.strip():
                                clean_value[k] = v
                        node["parameters"]["value"] = clean_value

    # Fix connections structure
    if "connections" in workflow_data:
        for node_name, connection_data in workflow_data["connections"].items():
            if "main" in connection_data:
                # Ensure main is a list of lists
                if isinstance(connection_data["main"], list):
                    for i, connection_list in enumerate(connection_data["main"]):
                        if isinstance(connection_list, list):
                            # Clean each connection
                            clean_connections = []
                            for conn in connection_list:
                                if isinstance(conn, dict) and "node" in conn:
                                    clean_connections.append(conn)
                            connection_data["main"][i] = clean_connections

    # Ensure settings exist
    if "settings" not in workflow_data:
        workflow_data["settings"] = {"executionOrder": "v1"}

    return workflow_data


def validate_workflow(workflow_data: dict[str, Any]) -> list[str]:
    """Validate workflow structure and return any issues"""
    issues = []

    # Check required fields
    required_fields = ["name", "nodes", "connections"]
    for field in required_fields:
        if field not in workflow_data:
            issues.append(f"Missing required field: {field}")

    # Check nodes structure
    if "nodes" in workflow_data:
        for i, node in enumerate(workflow_data["nodes"]):
            if not isinstance(node, dict):
                issues.append(f"Node {i} is not a dictionary")
                continue

            required_node_fields = ["name", "type", "position"]
            for field in required_node_fields:
                if field not in node:
                    issues.append(f"Node {i} missing required field: {field}")

    # Check connections structure
    if "connections" in workflow_data:
        for node_name, connection_data in workflow_data["connections"].items():
            if not isinstance(connection_data, dict):
                issues.append(f"Connection for {node_name} is not a dictionary")
                continue

            if "main" not in connection_data:
                issues.append(f"Connection for {node_name} missing 'main' field")

    return issues


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_n8n_workflow.py <workflow_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".json", "_fixed.json")

    try:
        # Read the workflow file
        with open(input_file, encoding="utf-8") as f:
            workflow_data = json.load(f)

        print(f"📖 Loaded workflow: {workflow_data.get('name', 'Unknown')}")

        # Validate before fixing
        issues = validate_workflow(workflow_data)
        if issues:
            print("⚠️  Issues found before fixing:")
            for issue in issues:
                print(f"   - {issue}")

        # Fix the workflow
        fixed_workflow = fix_n8n_workflow(workflow_data)

        # Validate after fixing
        issues_after = validate_workflow(fixed_workflow)
        if issues_after:
            print("❌ Issues remain after fixing:")
            for issue in issues_after:
                print(f"   - {issue}")
        else:
            print("✅ Workflow validation passed!")

        # Save the fixed workflow
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(fixed_workflow, f, indent=2, ensure_ascii=False)

        print(f"💾 Fixed workflow saved to: {output_file}")
        print(f"📊 Original size: {os.path.getsize(input_file)} bytes")
        print(f"📊 Fixed size: {os.path.getsize(output_file)} bytes")

        # Show what was fixed
        print("\n🔧 Fixes applied:")
        print("   - Removed problematic metadata (pinData, staticData, tags, etc.)")
        print("   - Added unique IDs to all nodes")
        print("   - Cleaned up parameter structures")
        print("   - Fixed connection arrays")
        print("   - Ensured proper settings structure")

    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
