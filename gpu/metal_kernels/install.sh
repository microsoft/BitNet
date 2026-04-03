#!/bin/bash
# Build and install BitNet Metal backend

set -e

echo "Building BitNet Metal Backend..."
echo "================================"

# Check prerequisites
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

if ! python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null; then
    echo "Error: PyTorch not found. Please install PyTorch first:"
    echo "  pip install torch"
    exit 1
fi

# Check Metal availability
if ! python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "Warning: Metal/MPS not available on this system"
    echo "The implementation will fall back to CPU"
fi

cd "$(dirname "$0")"

# Create build directory
mkdir -p build
cd build

# Try to build the Metal extension
echo ""
echo "Attempting to build Metal extension..."
echo "--------------------------------------"

# Note: Full build requires proper PyTorch C++ extension setup
# For now, we'll install the Python components
cd ..

# Install Python packages
echo ""
echo "Installing Python components..."
echo "------------------------------"

# Add metal_kernels to path
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
METAL_DIR="$SITE_PACKAGES/bitnet_metal"

echo "Installing to: $METAL_DIR"

# Create package directory
mkdir -p "$METAL_DIR"
cp metal_kernels/model.py "$METAL_DIR/"
cp metal_kernels/__init__.py "$METAL_DIR/" 2>/dev/null || echo "# Metal backend package" > "$METAL_DIR/__init__.py"

# Copy Metal shaders
mkdir -p "$METAL_DIR/shaders"
cp metal_kernels/*.metal "$METAL_DIR/shaders/" 2>/dev/null || echo "Note: No .metal files found"

echo ""
echo "Installation complete!"
echo "======================"
echo ""
echo "To use the Metal backend:"
echo "  from bitnet_metal.model import Transformer, ModelArgs"
echo ""
echo "To profile performance:"
echo "  python utils/profile_inference.py --backend metal"
echo ""
echo "Note: Full Metal kernel acceleration requires building the C++ extension:"
echo "  cd gpu/metal_kernels && python setup.py build_ext --inplace"
