#!/bin/bash
# key_importer.sh

# Define the path to your env file
KEY_FILE="${ISAACLAB_PATH}/_isaaclab_eureka/api_keys/.env.api_keys"

if [ -f "$KEY_FILE" ]; then
    echo "Loading environment variables from ${KEY_FILE}"
    while IFS= read -r line || [ -n "$line" ]; do
        # Remove leading/trailing whitespace
        trimmed=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        # Skip blank lines or lines starting with #
        if [[ -z "$trimmed" || "$trimmed" == \#* ]]; then
            continue
        fi

        # Expecting a line in the form key=value
        key=$(echo "$trimmed" | cut -d '=' -f1)
        value=$(echo "$trimmed" | cut -d '=' -f2-)

        # Export the variable. The syntax below is valid in bash.
        export "$key=$value"
    done < "$KEY_FILE"
else
    echo "WARNING: ${KEY_FILE} not found. No API keys are loaded."
fi