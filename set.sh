# Give pyasn1-modules the pyasn1 it wants
pip install "pyasn1<0.5.0,>=0.4.6"

# Bring pillow up to ~10.4 for crawl4ai & exo
pip install "pillow==10.4.0"

# Upgrade psutil and pydantic and rich for crawl4ai
pip install "psutil>=6.1.1" "pydantic>=2.10" "rich>=13.9.4"

# Downgrade protobuf for streamlit (<5.0)
pip install "protobuf<5,>=3.20"

# Pin Jinja2, numpy, and transformers for exo
pip install "Jinja2==3.1.4" "numpy==2.0.0" "transformers==4.46.3"

