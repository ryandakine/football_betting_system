#!/usr/bin/env python3
"""
Aggressive N8N Workflow JSON Fixer
Specifically targets "propertyValues[itemName] is not iterable" errors
"""

import json
import os
import sys
from typing import Any, Dict, List


def deep_clean_node(node: dict[str, Any]) -> dict[str, Any]:
    """Deep clean a node to remove all problematic structures"""

    # Create a clean copy
    clean_node = {
        "name": node.get("name", "Unknown Node"),
        "type": node.get("type", "n8n-nodes-base.function"),
        "typeVersion": node.get("typeVersion", 1),
        "position": node.get("position", [0, 0]),
        "id": node.get("id", f"node-{hash(node.get('name', 'unknown'))}"),
    }

    # Clean parameters - this is where the error usually occurs
    if "parameters" in node:
        clean_params = {}

        for key, value in node["parameters"].items():
            if key == "functionCode" and isinstance(value, str):
                # Keep function code as-is
                clean_params[key] = value
            elif key == "rule" and isinstance(value, dict):
                # Keep cron rules
                clean_params[key] = value
            elif key == "url" and isinstance(value, str):
                # Keep URLs
                clean_params[key] = value
            elif key == "method" and isinstance(value, str):
                # Keep HTTP methods
                clean_params[key] = value
            elif key == "responseFormat" and isinstance(value, str):
                # Keep response format
                clean_params[key] = value
            elif key == "queryParameters" and isinstance(value, dict):
                # Clean query parameters
                if "parameters" in value and isinstance(value["parameters"], list):
                    clean_params_list = []
                    for param in value["parameters"]:
                        if (
                            isinstance(param, dict)
                            and "name" in param
                            and "value" in param
                        ):
                            clean_params_list.append(
                                {
                                    "name": str(param["name"]),
                                    "value": str(param["value"]),
                                }
                            )
                    clean_params[key] = {"parameters": clean_params_list}
            elif key == "sendHeaders" and isinstance(value, bool):
                clean_params[key] = value
            elif key == "sendBody" and isinstance(value, bool):
                clean_params[key] = value
            elif key == "headerParameters" and isinstance(value, dict):
                # Clean header parameters
                if "parameters" in value and isinstance(value["parameters"], list):
                    clean_headers = []
                    for header in value["parameters"]:
                        if (
                            isinstance(header, dict)
                            and "name" in header
                            and "value" in header
                        ):
                            clean_headers.append(
                                {
                                    "name": str(header["name"]),
                                    "value": str(header["value"]),
                                }
                            )
                    clean_params[key] = {"parameters": clean_headers}
            elif key == "bodyParameters" and isinstance(value, dict):
                # Clean body parameters
                if "parameters" in value and isinstance(value["parameters"], list):
                    clean_body = []
                    for body_param in value["parameters"]:
                        if (
                            isinstance(body_param, dict)
                            and "name" in body_param
                            and "value" in body_param
                        ):
                            clean_body.append(
                                {
                                    "name": str(body_param["name"]),
                                    "value": str(body_param["value"]),
                                }
                            )
                    clean_params[key] = {"parameters": clean_body}
            elif key == "jsonBody" and isinstance(value, str):
                clean_params[key] = value
            elif key == "operation" and isinstance(value, str):
                clean_params[key] = value
            elif key == "table" and isinstance(value, str):
                clean_params[key] = value
            elif key == "columns" and isinstance(value, dict):
                # Clean columns structure
                if "mappingMode" in value and "value" in value:
                    clean_columns = {
                        "mappingMode": str(value["mappingMode"]),
                        "value": {},
                    }
                    if isinstance(value["value"], dict):
                        for col_key, col_value in value["value"].items():
                            if isinstance(col_value, str):
                                clean_columns["value"][col_key] = col_value
                    clean_params[key] = clean_columns
            elif key == "conditions" and isinstance(value, dict):
                # Clean conditions structure
                if "options" in value and "conditions" in value:
                    clean_conditions = {
                        "options": {
                            "caseSensitive": value["options"].get(
                                "caseSensitive", True
                            ),
                            "leftValue": "",
                            "typeValidation": "strict",
                        },
                        "conditions": [],
                    }
                    if isinstance(value["conditions"], list):
                        for condition in value["conditions"]:
                            if isinstance(condition, dict) and "id" in condition:
                                clean_condition = {
                                    "id": str(condition["id"]),
                                    "leftValue": str(condition.get("leftValue", "")),
                                    "rightValue": condition.get("rightValue", True),
                                    "operator": {"type": "boolean"},
                                }
                                clean_conditions["conditions"].append(clean_condition)
                    clean_conditions["combinator"] = "and"
                    clean_params[key] = clean_conditions
            elif key == "channel" and isinstance(value, str):
                clean_params[key] = value
            elif key == "text" and isinstance(value, str):
                clean_params[key] = value
            elif key == "options" and isinstance(value, dict):
                # Keep simple options
                clean_params[key] = {}

        if clean_params:
            clean_node["parameters"] = clean_params

    return clean_node


def fix_n8n_workflow_aggressive(workflow_data: dict[str, Any]) -> dict[str, Any]:
    """Aggressively fix n8n workflow issues"""

    # Create a completely clean workflow structure
    clean_workflow = {
        "name": workflow_data.get("name", "Fixed Workflow"),
        "nodes": [],
        "connections": {},
        "settings": {"executionOrder": "v1"},
    }

    # Clean each node
    if "nodes" in workflow_data:
        for i, node in enumerate(workflow_data["nodes"]):
            if isinstance(node, dict):
                clean_node = deep_clean_node(node)
                clean_workflow["nodes"].append(clean_node)

    # Clean connections - this is critical
    if "connections" in workflow_data:
        for node_name, connection_data in workflow_data["connections"].items():
            if isinstance(connection_data, dict) and "main" in connection_data:
                if isinstance(connection_data["main"], list):
                    clean_connections = []
                    for connection_list in connection_data["main"]:
                        if isinstance(connection_list, list):
                            clean_connection_list = []
                            for conn in connection_list:
                                if isinstance(conn, dict) and "node" in conn:
                                    clean_conn = {
                                        "node": str(conn["node"]),
                                        "type": "main",
                                        "index": 0,
                                    }
                                    clean_connection_list.append(clean_conn)
                            if clean_connection_list:
                                clean_connections.append(clean_connection_list)
                    if clean_connections:
                        clean_workflow["connections"][node_name] = {
                            "main": clean_connections
                        }

    return clean_workflow


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_n8n_workflow_aggressive.py <workflow_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".json", "_aggressive_fixed.json")

    try:
        # Read the workflow file
        with open(input_file, encoding="utf-8") as f:
            workflow_data = json.load(f)

        print(f"📖 Loaded workflow: {workflow_data.get('name', 'Unknown')}")
        print(f"📊 Original nodes: {len(workflow_data.get('nodes', []))}")

        # Apply aggressive fix
        fixed_workflow = fix_n8n_workflow_aggressive(workflow_data)

        print(f"📊 Fixed nodes: {len(fixed_workflow.get('nodes', []))}")

        # Save the fixed workflow
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(fixed_workflow, f, indent=2, ensure_ascii=False)

        print(f"💾 Aggressively fixed workflow saved to: {output_file}")
        print(f"📊 Original size: {os.path.getsize(input_file)} bytes")
        print(f"📊 Fixed size: {os.path.getsize(output_file)} bytes")

        # Show what was fixed
        print("\n🔧 Aggressive fixes applied:")
        print("   - Completely restructured all nodes")
        print("   - Removed ALL problematic metadata")
        print("   - Cleaned ALL parameter structures")
        print("   - Fixed ALL connection arrays")
        print("   - Simplified complex nested objects")
        print("   - Ensured all values are strings or simple types")

        print(f"\n🎯 Try importing: {output_file}")

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
