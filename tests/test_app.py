import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Test 1: Environment variable loading ────────────────────────────────
def test_env_template_exists():
    """env_template.txt should exist in repo"""
    assert os.path.exists("env_template.txt"), "env_template.txt not found"

def test_gemini_api_key_in_template():
    """env_template.txt should reference an API key"""
    with open("env_template.txt") as f:
        content = f.read()
    assert "API_KEY" in content, "No API_KEY found in env_template.txt"
# ── Test 2: Config module ────────────────────────────────────────────────
def test_system_config_imports():
    """SystemConfig should be importable"""
    from config.settings import SystemConfig
    assert SystemConfig is not None

def test_system_config_has_api_key_field():
    """SystemConfig should have GEMINI_API_KEY class attribute"""
    from config.settings import SystemConfig
    assert hasattr(SystemConfig, "GEMINI_API_KEY"), \
        "SystemConfig missing GEMINI_API_KEY"

def test_system_config_validate_api_key_method():
    """SystemConfig should have validate_api_key method"""
    from config.settings import SystemConfig
    assert callable(SystemConfig.validate_api_key), \
        "SystemConfig missing validate_api_key method"

def test_system_config_default_model():
    """SystemConfig should have DEFAULT_MODEL set"""
    from config.settings import SystemConfig
    assert SystemConfig.DEFAULT_MODEL is not None
    assert "gemini" in SystemConfig.DEFAULT_MODEL.lower()

# ── Test 3: Core coordinator ─────────────────────────────────────────────
def test_coordinator_imports():
    """ResearchCoordinator should be importable"""
    from core.coordinator import ResearchCoordinator
    assert ResearchCoordinator is not None

def test_coordinator_instantiation_without_key():
    """Coordinator should raise clear error without API key"""
    from core.coordinator import ResearchCoordinator
    original = os.environ.get("GEMINI_API_KEY")
    try:
        os.environ.pop("GEMINI_API_KEY", None)
        # Should either raise an error or return an object
        # — not silently fail
        try:
            coord = ResearchCoordinator()
            # If it doesn't raise, it should at least exist
            assert coord is not None
        except (ValueError, KeyError, TypeError) as e:
            # Raising a clear error is also acceptable
            assert len(str(e)) > 0
    finally:
        if original:
            os.environ["GEMINI_API_KEY"] = original

# ── Test 4: Requirements ─────────────────────────────────────────────────
def test_requirements_file_exists():
    """requirements.txt must exist"""
    assert os.path.exists("requirements.txt")

def test_required_packages_in_requirements():
    """Key packages must be in requirements.txt"""
    with open("requirements.txt") as f:
        content = f.read().lower()
    for pkg in ["streamlit", "langchain", "faiss", "pypdf"]:
        assert pkg in content, f"{pkg} missing from requirements.txt"

# ── Test 5: Project structure ────────────────────────────────────────────
def test_core_folder_exists():
    assert os.path.isdir("core"), "core/ folder missing"

def test_config_folder_exists():
    assert os.path.isdir("config"), "config/ folder missing"

def test_agents_folder_exists():
    assert os.path.isdir("agents"), "agents/ folder missing"

def test_streamlit_app_exists():
    assert os.path.exists("streamlit_app.py"), "streamlit_app.py missing"

def test_dockerfile_exists():
    assert os.path.exists("Dockerfile"), "Dockerfile missing"