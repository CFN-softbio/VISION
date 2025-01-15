import json

def analyze_commands(json_file_path):
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Initialize counters
        total_commands = len(data)
        default_commands = 0
        cog_counts = {}

        # Analyze each command
        for command in data:
            # Count default commands
            if command.get('default', False):
                default_commands += 1
            
            # Count cog types
            cog_type = command.get('cog', '')
            if cog_type:
                cog_counts[cog_type] = cog_counts.get(cog_type, 0) + 1

        # Print results
        print(f"\nCommand Analysis:")
        print("=" * 50)
        print(f"Total commands: {total_commands}")
        print(f"Default commands: {default_commands}")
        print("\nCog Distribution:")
        print("-" * 20)
        for cog, count in sorted(cog_counts.items()):
            print(f"{cog}: {count}")

        return {
            "total_commands": total_commands,
            "default_commands": default_commands,
            "cog_distribution": cog_counts
        }

    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Use the function
# file_path = 'beamline_prompts/11BM/command_examples.json'
beamline = '11BM'
file_path = f'{beamline}/command_examples.json'
stats = analyze_commands(file_path)
