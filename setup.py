from setuptools import setup, find_packages

setup(
    name="astrag",
    version="0.1.0",
    description="AST-based code parsing, slicing, and graph mapping tool",
    author="Jiajun Lin",
    # Automatically find the 'astrag' folder and its submodules
    packages=find_packages(include=["astrag", "astrag.*"]),
    
    # Add any external pip packages astrag needs to run (e.g., tree-sitter, networkx)
    install_requires=[
        # "networkx>=2.0",
        # "tree-sitter>=0.20.0",
    ],
    
    # This makes 'astrag' runnable from the terminal if your __main__.py has a main() function
    entry_points={
        "console_scripts": [
            "astrag=astrag.__main__:main",
        ],
    },
    python_requires=">=3.8",
)