#!/usr/bin/env python3
"""
Recover Workflow Logic - Extract all custom logic from original workflow
"""

import json
import os
import sys
from typing import Any, Dict, List


def extract_function_code(workflow_data: dict[str, Any]) -> dict[str, str]:
    """Extract all function code from the workflow"""
    function_code = {}

    if "nodes" in workflow_data:
        for node in workflow_data["nodes"]:
            if isinstance(node, dict) and "parameters" in node:
                if "functionCode" in node["parameters"]:
                    node_name = node.get("name", "Unknown")
                    function_code[node_name] = node["parameters"]["functionCode"]

    return function_code


def create_compatible_workflow(
    workflow_data: dict[str, Any], function_code: dict[str, str]
) -> dict[str, Any]:
    """Create a new compatible workflow with all original logic"""

    # Create a clean workflow structure
    new_workflow = {
        "name": workflow_data.get("name", "Recovered Workflow"),
        "nodes": [],
        "connections": {},
        "settings": {"executionOrder": "v1"},
    }

    # Recreate nodes with clean structure
    if "nodes" in workflow_data:
        for i, node in enumerate(workflow_data["nodes"]):
            if isinstance(node, dict):
                # Create clean node
                clean_node = {
                    "name": node.get("name", f"Node {i}"),
                    "type": node.get("type", "n8n-nodes-base.function"),
                    "typeVersion": node.get("typeVersion", 1),
                    "position": node.get("position", [i * 200, 300]),
                    "id": f"node-{i}-{hash(node.get('name', 'unknown'))}",
                }

                # Add parameters
                if "parameters" in node:
                    clean_params = {}

                    # Handle different node types
                    if node.get("type") == "n8n-nodes-base.cron":
                        if "rule" in node["parameters"]:
                            clean_params["rule"] = node["parameters"]["rule"]

                    elif node.get("type") == "n8n-nodes-base.httpRequest":
                        # Copy HTTP request parameters
                        for key in ["url", "method", "responseFormat"]:
                            if key in node["parameters"]:
                                clean_params[key] = node["parameters"][key]

                        # Copy query parameters
                        if "queryParameters" in node["parameters"]:
                            clean_params["queryParameters"] = node["parameters"][
                                "queryParameters"
                            ]

                        # Copy body parameters
                        if "bodyParameters" in node["parameters"]:
                            clean_params["bodyParameters"] = node["parameters"][
                                "bodyParameters"
                            ]

                        # Copy header parameters
                        if "headerParameters" in node["parameters"]:
                            clean_params["headerParameters"] = node["parameters"][
                                "headerParameters"
                            ]

                        # Copy JSON body
                        if "jsonBody" in node["parameters"]:
                            clean_params["jsonBody"] = node["parameters"]["jsonBody"]

                    elif node.get("type") == "n8n-nodes-base.function":
                        # Restore function code
                        node_name = node.get("name", "Unknown")
                        if node_name in function_code:
                            clean_params["functionCode"] = function_code[node_name]

                    elif node.get("type") == "n8n-nodes-base.set":
                        # Copy set parameters
                        if "operation" in node["parameters"]:
                            clean_params["operation"] = node["parameters"]["operation"]
                        if "values" in node["parameters"]:
                            clean_params["values"] = node["parameters"]["values"]
                        if "options" in node["parameters"]:
                            clean_params["options"] = node["parameters"]["options"]

                    elif node.get("type") == "n8n-nodes-base.if":
                        # Copy if conditions
                        if "conditions" in node["parameters"]:
                            clean_params["conditions"] = node["parameters"][
                                "conditions"
                            ]
                        if "options" in node["parameters"]:
                            clean_params["options"] = node["parameters"]["options"]

                    elif node.get("type") == "n8n-nodes-base.slack":
                        # Copy Slack parameters
                        for key in ["channel", "text"]:
                            if key in node["parameters"]:
                                clean_params[key] = node["parameters"][key]

                    elif node.get("type") == "n8n-nodes-base.supabase":
                        # Copy Supabase parameters
                        for key in ["operation", "table", "columns"]:
                            if key in node["parameters"]:
                                clean_params[key] = node["parameters"][key]

                    if clean_params:
                        clean_node["parameters"] = clean_params

                new_workflow["nodes"].append(clean_node)

    # Recreate connections
    if "connections" in workflow_data:
        for node_name, connection_data in workflow_data["connections"].items():
            if isinstance(connection_data, dict) and "main" in connection_data:
                if isinstance(connection_data["main"], list):
                    clean_main = []
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
                                clean_main.append(clean_connection_list)
                    if clean_main:
                        new_workflow["connections"][node_name] = {"main": clean_main}

    return new_workflow


def main():
    if len(sys.argv) != 2:
        print("Usage: python recover_workflow_logic.py <original_workflow_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".json", "_recovered.json")

    try:
        # Read the original workflow
        with open(input_file, encoding="utf-8") as f:
            workflow_data = json.load(f)

        print(f"📖 Recovering workflow: {workflow_data.get('name', 'Unknown')}")

        # Extract all function code
        function_code = extract_function_code(workflow_data)
        print(f"📊 Found {len(function_code)} function nodes with custom code")

        # List the functions found
        for func_name in function_code.keys():
            print(f"   - {func_name}")

        # Create new compatible workflow
        recovered_workflow = create_compatible_workflow(workflow_data, function_code)

        # Save the recovered workflow
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(recovered_workflow, f, indent=2, ensure_ascii=False)

        print(f"💾 Recovered workflow saved to: {output_file}")
        print(f"📊 Original nodes: {len(workflow_data.get('nodes', []))}")
        print(f"📊 Recovered nodes: {len(recovered_workflow.get('nodes', []))}")

        print(f"\n🎯 Try importing: {output_file}")
        print("This preserves ALL your original logic and function code!")

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
