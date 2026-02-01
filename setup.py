"""Setup configuration for Enterprise RAG Chatbot."""

from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="enterprise-rag-chatbot",
    version="1.0.0",
    author="Backend Developer Agent",
    author_email="backend@enterprise-rag.com",
    description="Complete RAG pipeline with pluggable components for enterprise use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enterprise/rag-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "gpu": [
            "torch[cuda]>=2.1.1",
            "accelerate>=0.24.0",
        ],
        "all": [
            "torch[cuda]>=2.1.1",
            "accelerate>=0.24.0",
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1", 
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag-server=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.yaml", "*.yml", "*.json"],
    },
)