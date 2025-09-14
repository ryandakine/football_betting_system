"""
Test script for all MLB betting system services
"""

import subprocess
import sys
import time

import requests


def test_mlflow():
    """Test MLflow server"""
    print("🧪 Testing MLflow...")
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow server is running at http://localhost:5000")
            return True
        else:
            print(f"❌ MLflow server returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MLflow server test failed: {e}")
        return False


def test_prometheus():
    """Test Prometheus"""
    print("📊 Testing Prometheus...")
    try:
        response = requests.get("http://localhost:9090/-/healthy", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus is running at http://localhost:9090")
            return True
        else:
            print(f"❌ Prometheus returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus test failed: {e}")
        return False


def test_grafana():
    """Test Grafana"""
    print("📈 Testing Grafana...")
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Grafana is running at http://localhost:3000")
            return True
        else:
            print(f"❌ Grafana returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Grafana test failed: {e}")
        return False


def test_alertmanager():
    """Test AlertManager"""
    print("🔔 Testing AlertManager...")
    try:
        response = requests.get("http://localhost:9093/-/healthy", timeout=5)
        if response.status_code == 200:
            print("✅ AlertManager is running at http://localhost:9093")
            return True
        else:
            print(f"❌ AlertManager returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ AlertManager test failed: {e}")
        return False


def test_node_exporter():
    """Test Node Exporter"""
    print("🖥️ Testing Node Exporter...")
    try:
        response = requests.get("http://localhost:9100/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Node Exporter is running at http://localhost:9100")
            return True
        else:
            print(f"❌ Node Exporter returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Node Exporter test failed: {e}")
        return False


def test_cadvisor():
    """Test cAdvisor"""
    print("🐳 Testing cAdvisor...")
    try:
        response = requests.get("http://localhost:8080/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ cAdvisor is running at http://localhost:8080")
            return True
        else:
            print(f"❌ cAdvisor returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ cAdvisor test failed: {e}")
        return False


def test_docker_containers():
    """Test Docker containers"""
    print("🐳 Testing Docker containers...")
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=mlb-",
                "--format",
                "table {{.Names}}\t{{.Status}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("✅ Docker containers status:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Docker command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker test failed: {e}")
        return False


def test_mlflow_integration():
    """Test MLflow Python integration"""
    print("🐍 Testing MLflow Python integration...")
    try:
        import mlflow

        mlflow.set_tracking_uri("http://localhost:5000")

        # Test creating an experiment
        experiment_name = "test-experiment"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="test-run") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.85)
            print(f"✅ MLflow integration successful! Run ID: {run.info.run_id}")
            return True

    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 MLB Betting System - Service Test")
    print("=" * 50)

    tests = [
        ("MLflow Server", test_mlflow),
        ("Prometheus", test_prometheus),
        ("Grafana", test_grafana),
        ("AlertManager", test_alertmanager),
        ("Node Exporter", test_node_exporter),
        ("cAdvisor", test_cadvisor),
        ("Docker Containers", test_docker_containers),
        ("MLflow Integration", test_mlflow_integration),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All services are running successfully!")
        print("\n📋 Access URLs:")
        print("- MLflow: http://localhost:5000")
        print("- Prometheus: http://localhost:9090")
        print("- Grafana: http://localhost:3000 (admin/admin)")
        print("- AlertManager: http://localhost:9093")
        print("- Node Exporter: http://localhost:9100")
        print("- cAdvisor: http://localhost:8080")
    else:
        print("⚠️ Some services need attention. Check the logs above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
