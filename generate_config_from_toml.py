import toml
import json
import os

TOML_PATH = ".streamlit/secrets.toml"
UNIFIED_CONFIG_PATH = "data/keys/azure_config.json"

if not os.path.exists(TOML_PATH):
    raise FileNotFoundError(f"❌ Could not find {TOML_PATH}")

secrets = toml.load(TOML_PATH)

print("🔍 Loaded secrets:")
print(json.dumps(secrets, indent=2))

# Initialize the output dictionary
output_config = {}

# --- Add all model configs (contain api_key and api_base) ---
for key, block in secrets.items():
    if isinstance(block, dict) and "api_key" in block:
        output_config[key] = block
        print(f"✅ Included block: [{key}]")

# --- Include Azure AI Search config if valid ---
if "azure-ai-search" in secrets:
    search_block = secrets["azure-ai-search"]
    required_keys = ["api_key", "azure_endpoint", "index_name"]
    if all(k in search_block for k in required_keys):
        output_config["azure-ai-search"] = search_block
        print("✅ Included [azure-ai-search]")
    else:
        print("⚠️ Incomplete [azure-ai-search] config. Missing keys.")

# --- Optional INPUT block (e.g. model params) ---
if "INPUT" in secrets:
    output_config["INPUT"] = secrets["INPUT"]
    print("✅ Included [INPUT] block")

# --- Write final merged config ---
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(UNIFIED_CONFIG_PATH), exist_ok=True)
with open(UNIFIED_CONFIG_PATH, "w") as f:
    json.dump(output_config, f, indent=2)

print(f"\n🎉 All configs written to {UNIFIED_CONFIG_PATH}")
