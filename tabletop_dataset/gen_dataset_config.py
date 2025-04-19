#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import requests
from openai import OpenAI

def initial_client():
    api_key = os.getenv("OPENAI_API_KEY")  # You need to replace this with your actual API key
    client = OpenAI(
        base_url="https://api.openai-proxy.org/v1",
        api_key=api_key)
    return client

def generate_diverse_instructions(client,env_name, original_instruction, num_variations=2):
    """
    Generate diverse instructions with the same meaning using OpenAI API.
    
    Args:
        env_name: The name of the environment
        original_instruction: The original instruction text
        num_variations: Number of alternative instructions to generate
        
    Returns:
        List of alternative instructions
    """
    
    # Craft a prompt that asks for alternative ways to express the instruction
    # The prompt explicitly asks for comma-separated responses for easy parsing
    prompt = f"""
    I need you to generate {num_variations} alternative ways to express a robotics task instruction, 
    keeping the same meaning but using different wording.
    
    Environment: {env_name}
    Original instruction: "{original_instruction}"
    
    Important requirements:
    1. Each alternative should clearly convey the same task as the original instruction
    2. Use natural, concise language appropriate for instructing a robot
    3. Maintain the same level of specificity as the original
    4. Vary the vocabulary and sentence structure
    5. Return ONLY the alternative instructions separated by commas, with no additional text, numbering, or explanation
    
    Example:
    If the original is "put the ball into the container", your response might be:
    place the ball inside the container,move the ball into the container
    """
    
    try:
        print(f"Generating alternatives for: {original_instruction} (env: {env_name})")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates alternative phrasings for robot instructions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=150,
            timeout=30
        )
        
        # Parse the response to extract the generated alternatives
        result = response.choices[0].message.content
        alternatives = [alt.strip() for alt in result.strip().split(',') if alt.strip()]
        
        # Ensure we have the requested number of alternatives
        if len(alternatives) < num_variations:
            print(f"Warning: Only generated {len(alternatives)} alternatives instead of {num_variations}")
        
        return alternatives[:num_variations]
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Fallback to simple variations if API call fails
        return []

def generate_dataset_json(args):
    # Base directory
    base_dir = "/home/wangxianhao/data/project/reasoning/Datasets"
    output_dir = "/home/wangxianhao/data/project/reasoning/rlds_dataset_builder/tabletop_dataset/"
    datasets_dir = os.path.join(base_dir, "datasets_2/e21754a84f6e80ad7af87f54d432a1a4e4683538")
    env_ins_path = os.path.join(base_dir, "env_ins.json")
    
    # Load environment instructions
    with open(env_ins_path, 'r') as f:
        env_instructions = json.load(f)
    
    # Initialize result structure
    result = {"datasets": []}
    
    # Get all environment directories under the datasets directory
    env_dirs = [d for d in os.listdir(datasets_dir) 
               if os.path.isdir(os.path.join(datasets_dir, d)) and 
               not d.startswith('.') and 
               d not in [".gitattributes", "README.md"]]
    
    # Process each environment directory
    for env_name in env_dirs:
        env_path = os.path.join(datasets_dir, env_name)
        
        # Check if this environment has instructions in env_ins.json
        if env_name in env_instructions:
            original_instruction = env_instructions[env_name]
            
            # Add entry with the original instruction
            result["datasets"].append({
                "name": env_name,
                "instruction": original_instruction,
                "hdf5_dir": env_path
            })
            
            if args.diverse:
                client = initial_client()
                # Generate diverse instructions
                diverse_instructions = generate_diverse_instructions(client, env_name, original_instruction)
                
                # Add separate entries for each diverse instruction
                for instruction in diverse_instructions:
                    result["datasets"].append({
                        "name": env_name,
                        "instruction": instruction,
                        "hdf5_dir": env_path
                    })
        else:
            print(f"Error: No instruction found for environment: {env_name}")
    
    # Write the result to a JSON file
    output_path = os.path.join(output_dir, "dataset_config.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Dataset JSON generated at: {output_path}")
    print(f"Total environments processed: {len(result['datasets'])}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset JSON with diverse instructions")
    parser.add_argument("--diverse", action="store_true", help="Generate diverse instructions")
    args = parser.parse_args()
    
    generate_dataset_json(args)
