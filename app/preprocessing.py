import pandas as pd
import json
import os

def process_conversations(df):
    """
    Processes the conversation DataFrame to extract structured conversations into a
    dictionary where keys are unique conversation IDs.

    The input DataFrame must contain: user_id, session_id, seq_num, user_question,
    answer - as shown to user, question_routing_label, and status.

    Returns: A dictionary where keys are 'CONV_X' and values are the conversation details.
    """

    # --- 1. Rename the complex column for easier access ---
    # The original column name is long and contains special characters, making it awkward to use.
    # The new column name 'llm_answer' is used in the turn structure for consistency.
    df = df.rename(columns={'answer - as shown to user': 'llm_answer'})

    # --- 2. Sort and group by session_id to keep turns together ---
    df = df.sort_values(by=['session_id', 'seq_num'])
    final_conversations_dict = {}
    grouped = df.groupby('session_id', sort=False)

    # --- 3. Iterate, Transform, and Construct the Final Dictionary ---
    for session_id, group_df in grouped:

        # 3a. Prepare the list of turns/interactions
        turns = []
        # Iterate over each row in the current conversation group
        for index, row in group_df.iterrows():
            # Create a dictionary representing a single Question/Answer turn
            turn = {
                'seq_num': int(row['seq_num']),
                'user_question': row['user_question'],
                'llm_answer': row['llm_answer'],
                'routing_label': row['question_routing_label'],
                'status': row['status']
            }
            turns.append(turn)

        # 3b. Build the conversation data object (the value for the final dictionary)
        conversation_data = {
            # Take the first value for user_id and session_id as they are constant for the group
            'user_id': group_df['user_id'].iloc[0],
            'session_id': session_id,
            'turns': turns # The list of turns created above
        }

        # 3c. Build the key and add to the final dictionary
        # Keyed by session_id to retain upstream grouping identifier.
        final_conversations_dict[session_id] = conversation_data

    return final_conversations_dict

def save_json_to_file(data_dict, output_filepath):
    """
    Saves the given Python dictionary to a specified file path as well-formatted JSON.

    Args:
        data_dict (dict): The dictionary containing the conversation data (all grouped data).
        output_filepath (str): The full path where the JSON file should be saved.

    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_filepath, 'w') as json_file:
            # Use json.dump for direct writing, with indent=2 for readability
            # This saves the entire conversations_dict into one large JSON file
            json.dump(data_dict, json_file, indent=2)

        print(f"✅ Successfully saved {len(data_dict)} conversations.")
        print(f"File location: **{os.path.abspath(output_filepath)}**")
        return True
    except IOError as e:
        print(f"❌ An error occurred while writing the file to '{output_filepath}': {e}")
        return False

def save_individual_conversations(conversations_dict, output_dir):
    """
    Saves each conversation in the dictionary as a separate JSON file.

    Args:
        conversations_dict (dict): Dictionary where keys are conversation IDs and values are conversation data.
        output_dir (str): The directory where individual files will be saved.

    Returns:
        int: The number of files successfully saved.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    saved_count = 0
    total_count = len(conversations_dict)

    for conv_id, conv_data in conversations_dict.items():
        # The filename will be the conversation ID (e.g., CONV_1.json)
        filename = f"{conv_id}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w') as f:
                # We save the conversation data (the value) here
                json.dump(conv_data, f, indent=2)
            saved_count += 1
        except IOError as e:
            print(f"❌ Error saving conversation {conv_id} to {filepath}: {e}")

    print(f"\n✅ Successfully saved {saved_count} of {total_count} conversations to individual files.")
    print(f"Output folder: **{os.path.abspath(output_dir)}**")

    return saved_count

def process_and_save_data(raw_data_dir, grouped_json_base_dir, individual_json_base_dir):
    """
    Processes all CSV files in a raw directory, groups conversations,
    and saves the results to specified JSON output directories.

    Args:
        raw_data_dir (str): Path to the directory containing raw CSV files.
        grouped_json_base_dir (str): Path to the directory for saving
                                     the full, grouped JSON file for each CSV.
        individual_json_base_dir (str): Path to the directory for saving
                                        individual conversation JSON files.
    """
    # Ensure output directories exist before starting any file operations
    os.makedirs(grouped_json_base_dir, exist_ok=True)
    os.makedirs(individual_json_base_dir, exist_ok=True)

    try:
        # Get a list of all files in the raw directory
        file_paths = os.listdir(raw_data_dir)
    except FileNotFoundError:
        print(f"Error: Raw data directory not found at '{raw_data_dir}'")
        return

    # Filter for only CSV files to avoid processing unrelated files
    csv_files = [f for f in file_paths if f.endswith('.csv')]

    print(f"Found {len(csv_files)} CSV files to process.")

    for file in csv_files:
        # 1. Construct the full path to the raw CSV file
        raw_filepath = os.path.join(raw_data_dir, file)

        # 2. Extract the unique identifier from the filename
        # Assumes the ID is the segment after the last hyphen and before '.csv'.
        # e.g., 'data-20231026.csv' -> splits to ['data', '20231026.csv'] -> takes '20231026.csv' -> splits to ['20231026'] -> takes '20231026'
        file_id = file.split(".csv")[0].split('-')[-1].strip()

        # --- Load CSV and Process Conversations ---
        print(f"\nProcessing file: {file}...")
        try:
            # Read the CSV into a pandas DataFrame
            df = pd.read_csv(raw_filepath)
            # Call the main processing function
            json_data = process_conversations(df)
        except Exception as e:
            print(f"An error occurred while loading or processing {file}: {e}")
            continue

        # --- Save the grouped conversations to a single JSON file ---
        # The file will be named using the extracted ID (e.g., '20231026.json')
        output_filepath = os.path.join(grouped_json_base_dir, f"{file_id}.json")
        save_json_to_file(json_data, output_filepath)

        # --- Save each conversation to individual JSON files ---
        # A new subdirectory is created for each CSV's individual conversations (e.g., 'individual_20231026')
        individual_output_dir = os.path.join(individual_json_base_dir, f"individual_{file_id}")
        save_individual_conversations(json_data, individual_output_dir)
        print(f"Saved individual conversations for {file} in: {individual_output_dir}")

    print("\nProcessing complete.")

def main(raw_dir, grouped_output_dir, individual_output_dir):
    process_and_save_data(
        raw_data_dir=raw_dir,
        grouped_json_base_dir=grouped_output_dir,
        individual_json_base_dir=individual_output_dir
    )
    return


if __name__ == "__main__":

    raw_deadend_dir               = 'path/to/raw'
    grouped_output_dir            = 'path/to/processed/json/grouped_conversations'
    individual_output_dir         = 'path/to/processed/json/individual_conversations'
    main(raw_deadend_dir, grouped_output_dir, individual_output_dir)
