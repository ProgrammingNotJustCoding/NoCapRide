#!/bin/bash

# Colors for terminal output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

# API base URL
API_BASE_URL="http://localhost:8888/api"

# Function to print colored output
print_info() {
  echo -e "${BLUE}INFO:${NC} $1"
}

print_success() {
  echo -e "${GREEN}SUCCESS:${NC} $1"
}

print_error() {
  echo -e "${RED}ERROR:${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}WARNING:${NC} $1"
}

# Function to run a curl command and display the result
run_curl() {
  local description=$1
  local command=$2
  local timeout=$3
  
  print_info "$description"
  echo "Command: $command"
  
  # Run the command with timeout
  if [ -n "$timeout" ]; then
    timeout $timeout bash -c "$command" && print_success "Request completed successfully" || print_error "Request failed or timed out"
  else
    eval "$command" && print_success "Request completed successfully" || print_error "Request failed"
  fi
  
  echo ""
}

# Test health endpoint
run_curl "Testing health endpoint" "curl -s $API_BASE_URL/health | jq ."

# Test regions endpoint with different filters
run_curl "Testing regions endpoint (all regions)" "curl -s \"$API_BASE_URL/regions?type=ward\" | jq ."
run_curl "Testing regions endpoint (simple numbers)" "curl -s \"$API_BASE_URL/regions?type=ward&filter=simple\" | jq ."
run_curl "Testing regions endpoint (b_ prefixed)" "curl -s \"$API_BASE_URL/regions?type=ward&filter=b\" | jq ."
run_curl "Testing regions endpoint (w_ prefixed)" "curl -s \"$API_BASE_URL/regions?type=ward&filter=w\" | jq ."

# Test historical data endpoint with different region types
run_curl "Testing historical data with simple region (1)" "curl -s \"$API_BASE_URL/historical?type=ward&hours=12&region=1\" | jq ."

# Get a b_ prefixed region if available
B_REGION=$(curl -s "$API_BASE_URL/regions?type=ward&filter=b" | jq -r '.regions[0]')
if [ "$B_REGION" != "null" ] && [ -n "$B_REGION" ]; then
  run_curl "Testing historical data with b_ region ($B_REGION)" "curl -s \"$API_BASE_URL/historical?type=ward&hours=12&region=$B_REGION\" | jq ."
else
  print_warning "No b_ prefixed regions available to test"
fi

# Get a w_ prefixed region if available
W_REGION=$(curl -s "$API_BASE_URL/regions?type=ward&filter=w" | jq -r '.regions[0]')
if [ "$W_REGION" != "null" ] && [ -n "$W_REGION" ]; then
  run_curl "Testing historical data with w_ region ($W_REGION)" "curl -s \"$API_BASE_URL/historical?type=ward&hours=12&region=$W_REGION\" | jq ."
else
  print_warning "No w_ prefixed regions available to test"
fi

# Test forecast endpoint with different region types
run_curl "Testing forecast with simple region (1)" "curl -s \"$API_BASE_URL/forecast?type=ward&hours=12&region=1\" | jq ." "60"

# Test forecast with b_ region if available
if [ "$B_REGION" != "null" ] && [ -n "$B_REGION" ]; then
  run_curl "Testing forecast with b_ region ($B_REGION)" "curl -s \"$API_BASE_URL/forecast?type=ward&hours=12&region=$B_REGION\" | jq ." "60"
else
  print_warning "No b_ prefixed regions available to test forecast"
fi

# Test forecast with w_ region if available
if [ "$W_REGION" != "null" ] && [ -n "$W_REGION" ]; then
  run_curl "Testing forecast with w_ region ($W_REGION)" "curl -s \"$API_BASE_URL/forecast?type=ward&hours=12&region=$W_REGION\" | jq ." "60"
else
  print_warning "No w_ prefixed regions available to test forecast"
fi

# Test forecast/all endpoint with different filters and worker counts
run_curl "Testing forecast/all endpoint (all regions, 4 workers)" "curl -s \"$API_BASE_URL/forecast/all?type=ward&hours=12&workers=4\" | jq ." "90"
run_curl "Testing forecast/all endpoint (simple regions, 2 workers)" "curl -s \"$API_BASE_URL/forecast/all?type=ward&hours=12&filter=simple&workers=2\" | jq ." "90"

# Test with maximum workers for better performance
run_curl "Testing forecast/all endpoint (all regions, 8 workers)" "curl -s \"$API_BASE_URL/forecast/all?type=ward&hours=12&workers=8\" | jq ." "90"

# If we have b_ regions, test with that filter
if [ "$B_REGION" != "null" ] && [ -n "$B_REGION" ]; then
  run_curl "Testing forecast/all endpoint (b_ regions, 4 workers)" "curl -s \"$API_BASE_URL/forecast/all?type=ward&hours=12&filter=b&workers=4\" | jq ." "90"
fi

# If we have w_ regions, test with that filter
if [ "$W_REGION" != "null" ] && [ -n "$W_REGION" ]; then
  run_curl "Testing forecast/all endpoint (w_ regions, 4 workers)" "curl -s \"$API_BASE_URL/forecast/all?type=ward&hours=12&filter=w&workers=4\" | jq ." "90"
fi

# Test model info endpoint
run_curl "Testing model info endpoint" "curl -s \"$API_BASE_URL/model/info?type=ward\" | jq ."

# Test cache clear endpoint
run_curl "Testing cache clear endpoint" "curl -s -X POST \"$API_BASE_URL/cache/clear\" | jq ."

print_info "All tests completed!" 