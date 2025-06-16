#!/bin/bash

# Swift Qwen3 Fine-tuning Test Runner
# This script sets up the environment and runs all tests for the Swift fine-tuning pipeline.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Swift Qwen3 Fine-tuning Test Runner${NC}"
echo "========================================"

# Function to print colored output
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log_info "Script directory: $SCRIPT_DIR"
log_info "Project root: $PROJECT_ROOT"

# Step 1: Environment Setup
log_info "Setting up environment..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    log_success "Virtual environment detected: $VIRTUAL_ENV"
else
    log_warning "No virtual environment detected"
    
    # Check if .venv exists
    if [[ -d "$PROJECT_ROOT/.venv" ]]; then
        log_info "Activating .venv..."
        source "$PROJECT_ROOT/.venv/bin/activate"
        log_success "Activated .venv"
    else
        log_warning "No .venv found. Consider creating one with: uv venv"
    fi
fi

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
log_info "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
log_info "Set PYTHONPATH=$PYTHONPATH"

# Step 2: Check Dependencies
log_info "Checking dependencies..."

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_success "Python available: $PYTHON_VERSION"
else
    log_error "Python3 not found"
    exit 1
fi

# Check if uv is available (recommended)
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    log_success "uv available: $UV_VERSION"
    PACKAGE_MANAGER="uv"
else
    log_warning "uv not found, using pip"
    PACKAGE_MANAGER="pip"
fi

# Step 3: Install Required Dependencies
log_info "Installing required dependencies..."

if [[ "$PACKAGE_MANAGER" == "uv" ]]; then
    uv pip install -e "$PROJECT_ROOT" || log_warning "Failed to install project in editable mode"
    uv pip install pytest pytest-cov || log_warning "Failed to install testing dependencies"
else
    pip install -e "$PROJECT_ROOT" || log_warning "Failed to install project in editable mode"
    pip install pytest pytest-cov || log_warning "Failed to install testing dependencies"
fi

# Step 4: Validate Project Structure
log_info "Validating project structure..."

# Check if main finetune script exists
if [[ -f "$PROJECT_ROOT/finetune_swift.py" ]]; then
    log_success "Found finetune_swift.py"
else
    log_error "finetune_swift.py not found in project root"
    exit 1
fi

# Check if configuration file exists
if [[ -f "$PROJECT_ROOT/finetune.yaml" ]]; then
    log_success "Found finetune.yaml"
else
    log_warning "finetune.yaml not found - tests will use defaults"
fi

# Step 5: Run Unit Tests
log_info "Running unit tests..."

cd "$SCRIPT_DIR"

if python3 test_swift_finetune.py; then
    log_success "Unit tests passed"
    UNIT_TEST_RESULT=0
else
    log_error "Unit tests failed"
    UNIT_TEST_RESULT=1
fi

# Step 6: Run Integration Tests
log_info "Running integration tests..."

if python3 test_integration.py; then
    log_success "Integration tests passed"
    INTEGRATION_TEST_RESULT=0
else
    log_error "Integration tests failed"
    INTEGRATION_TEST_RESULT=1
fi

# Step 7: Run Configuration Validation Test
log_info "Running configuration validation test..."

if python3 -c "
import sys
sys.path.append('$PROJECT_ROOT')
from finetune_swift import SwiftQwen3FineTuner
try:
    fine_tuner = SwiftQwen3FineTuner()
    print('✅ Configuration validation passed')
    sys.exit(0)
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
"; then
    log_success "Configuration validation passed"
    CONFIG_TEST_RESULT=0
else
    log_error "Configuration validation failed"
    CONFIG_TEST_RESULT=1
fi

# Step 8: Test Script Execution (Dry Run)
log_info "Testing script execution (dry run)..."

cd "$PROJECT_ROOT"

if python3 finetune_swift.py --help &> /dev/null; then
    log_success "Script help command works"
    SCRIPT_TEST_RESULT=0
else
    log_error "Script help command failed"
    SCRIPT_TEST_RESULT=1
fi

# Step 9: Summary
echo ""
echo "========================================"
log_info "TEST SUMMARY"
echo "========================================"

TOTAL_TESTS=4
PASSED_TESTS=0

if [[ $UNIT_TEST_RESULT -eq 0 ]]; then
    log_success "Unit Tests: PASSED"
    ((PASSED_TESTS++))
else
    log_error "Unit Tests: FAILED"
fi

if [[ $INTEGRATION_TEST_RESULT -eq 0 ]]; then
    log_success "Integration Tests: PASSED"
    ((PASSED_TESTS++))
else
    log_error "Integration Tests: FAILED"
fi

if [[ $CONFIG_TEST_RESULT -eq 0 ]]; then
    log_success "Configuration Tests: PASSED"
    ((PASSED_TESTS++))
else
    log_error "Configuration Tests: FAILED"
fi

if [[ $SCRIPT_TEST_RESULT -eq 0 ]]; then
    log_success "Script Execution Tests: PASSED"
    ((PASSED_TESTS++))
else
    log_error "Script Execution Tests: FAILED"
fi

echo ""
echo "Tests passed: $PASSED_TESTS/$TOTAL_TESTS"

if [[ $PASSED_TESTS -eq $TOTAL_TESTS ]]; then
    log_success "All tests passed! 🎉"
    echo ""
    log_info "You can now run Swift fine-tuning with:"
    echo "  cd $PROJECT_ROOT"
    echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3"
    echo "  source .venv/bin/activate"
    echo "  python3 finetune_swift.py --mode full"
    exit 0
else
    log_error "Some tests failed. Please check the output above."
    exit 1
fi 