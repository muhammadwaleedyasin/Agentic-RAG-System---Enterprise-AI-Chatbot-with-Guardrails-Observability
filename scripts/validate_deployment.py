#!/usr/bin/env python3
"""
Deployment Validation Script for Agentic RAG System
Validates that the project is ready for deployment to GitHub and production.
"""

import os
import sys
from pathlib import Path

def check_required_files():
    """Check that all required files exist."""
    required_files = [
        'README.md',
        '.gitignore',
        '.env.example',
        'requirements.txt',
        'pyproject.toml',
        'deploy/compose/docker-compose.yml',
        'deploy/compose/.env.example',
        'frontend/package.json',
        'frontend/.env.example',
        'frontend/Dockerfile',
        'deploy/compose/Dockerfile.app',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_env_files():
    """Verify environment files are properly templated."""
    env_files = [
        '.env.example',
        'deploy/compose/.env.example',
        'frontend/.env.example'
    ]
    
    issues = []
    for env_file in env_files:
        if Path(env_file).exists():
            with open(env_file, 'r') as f:
                content = f.read()
                # Check for actual secrets that shouldn't be there
                if 'sk-' in content or 'xoxb-' in content or 'ghp_' in content:
                    issues.append(f"{env_file} contains what appears to be real API keys")
        else:
            issues.append(f"{env_file} does not exist")
    
    if issues:
        print("❌ Environment file issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Environment files properly templated")
        return True

def check_sensitive_files():
    """Check that no sensitive files are present."""
    sensitive_patterns = [
        '.env',
        '*.key',
        '*.pem',
        '*.p12',
        '*.jks',
        'debug-*.png',
        'debug-*.jpg',
    ]
    
    sensitive_found = []
    for pattern in sensitive_patterns:
        for file_path in Path('.').glob(pattern):
            if file_path.name != '.env.example':
                sensitive_found.append(str(file_path))
    
    if sensitive_found:
        print("❌ Sensitive files found (should be in .gitignore):")
        for file_path in sensitive_found:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ No sensitive files found in repository")
        return True

def check_docker_configs():
    """Validate Docker configurations are production-ready."""
    docker_compose_path = Path('deploy/compose/docker-compose.yml')
    
    if not docker_compose_path.exists():
        print("❌ Docker compose file not found")
        return False
    
    with open(docker_compose_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for hardcoded localhost references
    if 'localhost:' in content and 'localhost:3000' not in content:
        issues.append("Docker compose contains hardcoded localhost references")
    
    # Check that environment variables are used
    if '${' not in content:
        issues.append("Docker compose should use environment variables")
    
    if issues:
        print("❌ Docker configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Docker configurations look good")
        return True

def check_documentation():
    """Check that documentation is complete."""
    readme_path = Path('README.md')
    
    if not readme_path.exists():
        print("❌ README.md not found")
        return False
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    required_sections = [
        '# Agentic RAG',
        'Quick Start',
        'Docker',
        'Configuration',
        'Environment',
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print("❌ README.md missing sections:")
        for section in missing_sections:
            print(f"  - {section}")
        return False
    else:
        print("✅ Documentation appears complete")
        return True

def main():
    """Run all validation checks."""
    print("🚀 Validating deployment readiness...\n")
    
    checks = [
        check_required_files(),
        check_env_files(),
        check_sensitive_files(),
        check_docker_configs(),
        check_documentation(),
    ]
    
    print(f"\n{'='*50}")
    
    if all(checks):
        print("🎉 All checks passed! Project is ready for deployment.")
        return 0
    else:
        print("⚠️  Some issues found. Please resolve them before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
