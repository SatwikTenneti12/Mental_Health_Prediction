




import pandas as pd
import gradio as gr
import os
from requests import get



import numpy as np
import pandas as pd
import platform
import joblib


import pandas as pd
import os
from requests import get

# Questions
dass_questions = [
    f"Q{i}A: {desc}"
    for i, desc in enumerate(
        [
            "I found myself getting upset by quite trivial things.",
            "I was aware of dryness of my mouth.",
            "I couldn't seem to experience any positive feeling at all.",
            "I experienced breathing difficulty (e.g., excessively rapid breathing).",
            "I just couldn't seem to get going.",
            "I tended to over-react to situations.",
            "I had a feeling of shakiness.",
            "I found it difficult to relax.",
            "I found myself in situations that made me so anxious I was most relieved when they ended.",
            "I felt that I had nothing to look forward to.",
            "I found myself getting upset rather easily.",
            "I felt that I was using a lot of nervous energy.",
            "I felt sad and depressed.",
            "I found myself getting impatient when I was delayed.",
            "I had a feeling of faintness.",
            "I felt that I had lost interest in just about everything.",
            "I felt I wasn't worth much as a person.",
            "I felt that I was rather touchy.",
            "I perspired noticeably in the absence of high temperatures or physical exertion.",
            "I felt scared without any good reason.",
            "I felt that life wasn't worthwhile.",
            "I found it hard to wind down.",
            "I had difficulty in swallowing.",
            "I couldn't seem to get any enjoyment out of the things I did.",
            "I was aware of the action of my heart in the absence of physical exertion.",
            "I felt down-hearted and blue.",
            "I found that I was very irritable.",
            "I felt I was close to panic.",
            "I found it hard to calm down after something upset me.",
            "I feared that I would be 'thrown' by some trivial but unfamiliar task.",
            "I was unable to become enthusiastic about anything.",
            "I found it difficult to tolerate interruptions to what I was doing.",
            "I was in a state of nervous tension.",
            "I felt I was pretty worthless.",
            "I was intolerant of anything that kept me from getting on with what I was doing.",
            "I felt terrified.",
            "I could see nothing in the future to be hopeful about.",
            "I felt that life was meaningless.",
            "I found myself getting agitated.",
            "I was worried about situations in which I might panic and make a fool of myself.",
            "I experienced trembling.",
            "I found it difficult to work up the initiative to do things.",
        ],
        start=1,
    )
]

# Options for responses
dass_options = {
    "Did not apply to me at all": 1,
    "Applied to me to some degree, or some of the time": 2,
    "Applied to me to a considerable degree, or a good part of the time": 3,
    "Applied to me very much, or most of the time": 4,
}

# TIPI Questions
tipi_questions = [
    "Extraverted, enthusiastic.",
    "Critical, quarrelsome.",
    "Dependable, self-disciplined.",
    "Anxious, easily upset.",
    "Open to new experiences, complex.",
    "Reserved, quiet.",
    "Sympathetic, warm.",
    "Disorganized, careless.",
    "Calm, emotionally stable.",
    "Conventional, uncreative.",
]

# TIPI Options
tipi_options = {
    "Disagree strongly": 1,
    "Disagree moderately": 2,
    "Disagree a little": 3,
    "Neither agree nor disagree": 4,
    "Agree a little": 5,
    "Agree moderately": 6,
    "Agree strongly": 7,
}

# VCL Words
vcl_words = [
    "boat", "incoherent", "pallid", "robot", "audible", "cuivocal",
    "paucity", "epistemology", "florted", "decide", "pastiche",
    "verdid", "abysmal", "lucid", "betray", "funny"
]

# Define the demographics and metadata options
demographics = {
    "education": {"Less than high school": 1, "High school": 2, "University degree": 3, "Graduate degree": 4},
    "urban": {"Rural": 1, "Suburban": 2, "Urban": 3},
    "gender": {"Male": 1, "Female": 2, "Other": 3},
    "engnat": {"Yes": 1, "No": 2},
    "age": None,  # Numeric input
    "hand": {"Right": 1, "Left": 2, "Both": 3},
    "religion": {
        "Agnostic": 1, "Atheist": 2, "Buddhist": 3, "Christian (Catholic)": 4,
        "Christian (Mormon)": 5, "Christian (Protestant)": 6, "Christian (Other)": 7,
        "Hindu": 8, "Jewish": 9, "Muslim": 10, "Sikh": 11, "Other": 12
    },
    "orientation": {"Heterosexual": 1, "Bisexual": 2, "Homosexual": 3, "Asexual": 4, "Other": 5},
    "race": {"Asian": 10, "Arab": 20, "Black": 30, "Indigenous Australian": 40, "Native American": 50, "White": 60, "Other": 70},
    "voted": {"Yes": 1, "No": 2},
    "married": {"Never married": 1, "Currently married": 2, "Previously married": 3},
    "familysize": None,  # Numeric input
    "major": None,       # Text input
}


# Derived fields
def derive_information():
    try:
        ip_data = get("https://ipinfo.io").json()
        country = ip_data.get("country", "Unknown")
    except Exception:
        country = "Unknown"
    screensize = 2 if platform.system() in ["Linux", "Windows"] else 1
    uniquenetworklocation = 1
    source = 1
    return country, screensize, uniquenetworklocation, source

def save_user_inputs(inputs):
    """
    Save user inputs to the CSV file with appropriate column names and numeric answers.

    Parameters:
        inputs (list): List of responses collected from the user.
    """
    try:
        # Collect metadata
        country, screensize, uniquenetworklocation, source = derive_information()

        # Process inputs
        dass_responses = [dass_options.get(response, None) for response in inputs[:42]]
        tipi_responses = [tipi_options.get(response, None) for response in inputs[42:52]]
        vcl_responses = [1 if response else 0 for response in inputs[52:68]]
        demographics_responses = inputs[68:-1]  # Exclude the terms checkbox

        # Ensure all demographic responses are present and assign defaults if missing
        demographics_keys = [
            "education", "urban", "gender", "engnat", "age", "hand", "religion",
            "orientation", "race", "voted", "married", "familysize", "major"
        ]
        demographics_data = {key: demographics_responses[i] if i < len(demographics_responses) else None
                             for i, key in enumerate(demographics_keys)}

        # Prepare the row data
        data = {
            **{f"Q{i}A": dass_responses[i - 1] for i in range(1, 43)},
            **{f"TIPI{i}": tipi_responses[i - 1] for i in range(1, 11)},
            **{f"VCL{i}": vcl_responses[i - 1] for i in range(1, 17)},
            "country": country,
            "source": source,
            "screensize": screensize,
            "uniquenetworklocation": uniquenetworklocation,
            **demographics_data,
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])
        output_file = "survey_results.csv"
        # Save or append to the CSV file
        if not os.path.exists(output_file):
            # Add column headers if the file doesn't exist
            df.to_csv(output_file, index=False)
        else:
            # Append without headers if the file exists
            df.to_csv(output_file, mode="a", header=False, index=False)

        return "Inputs saved successfully!"
    except Exception as e:
        print(f"Error saving inputs: {e}")
        return f"An error occurred while saving the inputs: {e}"


def preprocess_data(data):
    """
    Preprocesses the input data to generate cleaned and structured datasets
    for Depression, Stress, and Anxiety.

    Parameters:
        data (pd.DataFrame): Input data containing survey responses.

    Returns:
        pd.DataFrame: Preprocessed Depression dataset.
        pd.DataFrame: Preprocessed Stress dataset.
        pd.DataFrame: Preprocessed Anxiety dataset.
    """
    try:
        # Step 1: Clean and transform the data
        data_1 = data.copy()
        data_1['major'] = data_1['major'].replace(np.nan, 'No Degree')
        def assign_age_group(age):
            if age <= 10:
                return 'Under 10'
            elif 10 <= age <= 16:
                return 'Primary Children'
            elif 17 <= age <= 21:
                return 'Secondary Children'
            elif 22 <= age <= 35:
                return 'Adults'
            elif 36 <= age <= 48:
                return 'Elder Adults'
            elif age >= 49:
                return 'Older People'
            return 'Unknown'

        # Create Age_Groups column if it doesn't exist
        if 'Age_Groups' not in data_1.columns:
            data_1['Age_Groups'] = data_1['age'].apply(assign_age_group)
        # Drop unnecessary columns
        # data_1 = data_1.drop(data_1.iloc[:, 43:44], axis=1)
        data_1 = data_1.drop(columns=['source'], errors='ignore')
        # Further cleaning and transformation
        data_2 = data_1.copy()
        # data_2 = data_2.drop(data_2.iloc[:, 51:69], axis=1)
        columns_to_drop = [f"VCL{i}" for i in range(1, 17)]
        data_2 = data_2.drop(columns=columns_to_drop, errors='ignore')
        data_2 = data_2.replace(to_replace=0, value=3)
        data_2 = data_2.rename(columns={
            'TIPI1': 'Extraverted-enthusiastic',
            'TIPI2': 'Critical-quarrelsome',
            'TIPI3': 'Dependable-self_disciplined',
            'TIPI4': 'Anxious-easily upset',
            'TIPI5': 'Open to new experiences-complex',
            'TIPI6': 'Reserved-quiet',
            'TIPI7': 'Sympathetic-warm',
            'TIPI8': 'Disorganized-careless',
            'TIPI9': 'Calm-emotionally_stable',
            'TIPI10': 'Conventional-uncreative'
        })

        # Replace inf/-inf and drop NaN values
        data_2 = data_2.replace([np.inf, -np.inf], np.nan)
        data_2 = data_2.dropna()

        # Step 2: Extract new_data and DASS-related data
        new_data = data_2.iloc[:, 42:]
        data_3 = data_2.filter(regex=r'Q\d{1,2}A')
        # breakpoint()
        # Adjust responses for DASS keys
        data_3 = data_3.subtract(1, axis=1)

        # Step 3: Create Depression, Stress, and Anxiety datasets
        DASS_keys = {
            'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
            'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
            'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
        }

        depression_cols = [f"Q{i}A" for i in DASS_keys["Depression"]]
        stress_cols = [f"Q{i}A" for i in DASS_keys["Stress"]]
        anxiety_cols = [f"Q{i}A" for i in DASS_keys["Anxiety"]]

        depression = data_3.filter(depression_cols)
        stress = data_3.filter(stress_cols)
        anxiety = data_3.filter(anxiety_cols)

        # Calculate scores for each category
        for dataset in [depression, stress, anxiety]:
            dataset['Total_Count'] = dataset.sum(axis=1)
        # breakpoint()
        # Merge with new_data for additional information
        Depression = pd.merge(depression, new_data, how='left', left_index=True, right_index=True)
        Stress = pd.merge(stress, new_data, how='inner', left_index=True, right_index=True)
        Anxiety = pd.merge(anxiety, new_data, how='inner', left_index=True, right_index=True)
        
        # Step 4: Transform Age_Groups
        def change_var(x):
            if x == 'Primary Children':
                return 0
            elif x == 'Secondary Children':
                return 1
            elif x == 'Adults':
                return 2
            elif x == 'Elder Adults':
                return 3
            elif x == 'Older People':
                return 4
            return np.nan  # Default if value is unexpected
        
        for dataset in [Depression, Stress, Anxiety]:
            if 'Age_Groups' in dataset.columns:
                dataset['Age_Groups'] = dataset['Age_Groups'].apply(change_var)
        # Drop NaN values
        
        Depression = Depression.dropna()
        Stress = Stress.dropna()
        Anxiety = Anxiety.dropna()

        # Drop unnecessary columns
        for dataset in [Depression, Stress, Anxiety]:
            dataset.drop(columns=['Total_Count', 'country', 'age', 'major'], inplace=True, errors='ignore')
        
        return Depression, Stress, Anxiety
       
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def encode_demographics(data):
    encoding_map = {
        "education": {"Less than high school": 1, "High school": 2, "University degree": 3, "Graduate degree": 4},
        "urban": {"Rural": 1, "Suburban": 2, "Urban": 3},
        "gender": {"Male": 1, "Female": 2, "Other": 3},
        "engnat": {"Yes": 1, "No": 2},
        "age": None,  # Numeric input
        "hand": {"Right": 1, "Left": 2, "Both": 3},
        "religion": {
            "Agnostic": 1, "Atheist": 2, "Buddhist": 3, "Christian (Catholic)": 4,
            "Christian (Mormon)": 5, "Christian (Protestant)": 6, "Christian (Other)": 7,
            "Hindu": 8, "Jewish": 9, "Muslim": 10, "Sikh": 11, "Other": 12
        },
        "orientation": {"Heterosexual": 1, "Bisexual": 2, "Homosexual": 3, "Asexual": 4, "Other": 5},
        "race": {"Asian": 10, "Arab": 20, "Black": 30, "Indigenous Australian": 40, "Native American": 50, "White": 60, "Other": 70},
        "voted": {"Yes": 1, "No": 2},
        "married": {"Never married": 1, "Currently married": 2, "Previously married": 3},
        "familysize": None,  # Numeric input
        "major": None,       # Text input
    }
    for col, mapping in encoding_map.items():
        if mapping:  # If an encoding map is provided
            if col in data.columns:
                print(f"Encoding column: {col}")
                data[col] = data[col].replace(mapping)
        else:
            print(f"Skipping encoding for column: {col} (no mapping provided)")

    return data

def validate_inputs(inputs):
    # Check that all radio buttons are answered (DASS and TIPI)
    dass_responses = inputs[:42]  # First 42 inputs are DASS questions
    tipi_responses = inputs[42:52]  # Next 10 inputs are TIPI questions
    terms_agreed = inputs[-1]  # The last input is the terms checkbox

    # Check for missing responses
    if any(response is None for response in dass_responses + tipi_responses):
        return "Please answer all DASS and TIPI questions.", None, None, None

    # Check that the checkbox is checked
    if not terms_agreed:
        return "You must agree to the terms and conditions.", None, None, None

    # Validation passed
    return False

# Process and Predict Function
def process_and_predict(*inputs):
    """
    Process user input, update the CSV file, preprocess the data, and predict the latest input values.

    Parameters:
        inputs: User responses collected from the Gradio interface.

    Returns:
        tuple: Messages and predictions for Depression, Stress, and Anxiety.
    """
    try:
        print("Inputs received from Gradio:", inputs)  # Debugging: Log Gradio inputs
        validation_error = validate_inputs(inputs)
        if validation_error:
            # Check if validation_error is a tuple (indicating an error message is present)
            if isinstance(validation_error, tuple) and validation_error[0]:
                return validation_error  # Return the error message
            else:
                return "An unknown validation error occurred."  # Fallback case

        # Step 1: Save user inputs to the CSV file
        save_status = save_user_inputs(inputs)
        print("Save status:", save_status)  # Debugging: Log save status
        if "successfully" not in save_status:
            return save_status, None, None, None

        # Step 2: Load and preprocess the updated CSV file
        output_file = "survey_results.csv"
        if not os.path.exists(output_file):
            return "Error: survey_results.csv file not found.", None, None, None

        updated_data = pd.read_csv(output_file)
        print("Loaded data shape:", updated_data.shape)  # Debugging: Log CSV shape
        print("Loaded data columns:", updated_data.columns.tolist())  # Debugging: Log CSV columns

        if updated_data.empty:
            return "Error: CSV file is empty. No data to preprocess.", None, None, None
        invalid_columns = ['VCL6', 'VCL9', 'VCL12']
        if any((updated_data.loc[updated_data.index[-1], col] == 1) for col in invalid_columns if col in updated_data.columns):
            print("Invalid response")
            return "Error: Invalid responses detected in survey.", None, None, None   
        Depression, Stress, Anxiety = preprocess_data(updated_data)
        
        
        # Step 3: Ensure preprocessed data is not empty
        print("Depression dataset shape:", Depression.shape)
        print("Stress dataset shape:", Stress.shape)
        print("Anxiety dataset shape:", Anxiety.shape)

        if Depression.empty or Stress.empty or Anxiety.empty:
            print("Error: Preprocessed data is empty.")
            return (
                "Error: Preprocessed data is empty. Please check the input data or preprocessing logic.",
                None,
                None,
                None,
            )

        # Step 4: Load pre-trained models
        try:
            depression_model = joblib.load("random_forest_Depression.pkl")
            print("Depression model loaded successfully.")
            stress_model = joblib.load("random_forest_stress.pkl")
            print("Stress model loaded successfully.")
            anxiety_model = joblib.load("random_forest_anxiety.pkl")
            print("Anxiety model loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: Model file not found - {e}")
            return f"Error: Model file not found - {e}", None, None, None
        except Exception as e:
            print(f"Unexpected error while loading models: {e}")
            return f"Unexpected error: {e}", None, None, None

        # Step 5: Predict the latest user input
        try:
            print("Preparing input for Depression prediction...")
            print("Shape of input data for Depression prediction:", Depression.iloc[[-1], :-1].shape)
            print("Input data for Depression prediction:", Depression.iloc[[-1], :-1])
            Depression_encoded = encode_demographics(Depression)
            Anxiety_encoded= encode_demographics(Anxiety)
            Stress_encoded = encode_demographics(Stress)
            with open("feature_names.txt", "r") as f:
                trained_features = f.read().splitlines()
            Depression_encoded = Depression_encoded.reindex(columns=trained_features, fill_value=0)

            with open("feature_names1.txt", "r") as f:
                trained_features = f.read().splitlines()
            Anxiety_encoded = Anxiety_encoded.reindex(columns=trained_features, fill_value=0)

            with open("feature_names2.txt", "r") as f:
                trained_features = f.read().splitlines()
            Stress_encoded = Stress_encoded.reindex(columns=trained_features, fill_value=0)

            depression_prediction = depression_model.predict(Depression_encoded.iloc[[-1]])[0]
            print(f"Depression Prediction: {depression_prediction}")

            stress_prediction = stress_model.predict(Stress_encoded.iloc[[-1]])[0]
            print(f"Stress Prediction: {stress_prediction}")

            anxiety_prediction = anxiety_model.predict(Anxiety_encoded.iloc[[-1]])[0]
            print(f"Anxiety Prediction: {anxiety_prediction}")

            print("Responses saved and processed successfully!")
            print(f"Depression Level: {depression_prediction}")
            print(f"Stress Level: {stress_prediction}")
            print(f"Anxiety Level: {anxiety_prediction}")

            return (
                "Responses saved and processed successfully!",
                f"Depression Level: {depression_prediction}",
                f"Stress Level: {stress_prediction}",
                f"Anxiety Level: {anxiety_prediction}",
            )

        except Exception as e:
            print(f"Error during predictions: {e}")
            return f"Error during predictions: {e}", None, None, None


    except Exception as e:
        return f"An unexpected error occurred: {e}", None, None, None



import gradio as gr

dass_inputs = [gr.Radio(label=q, choices=list(dass_options.keys())) for q in dass_questions]
tipi_inputs = [gr.Radio(label=q, choices=list(tipi_options.keys())) for q in tipi_questions]
vcl_inputs = [gr.Checkbox(label=q) for q in vcl_words]

demographics_inputs = [
    gr.Radio(label=key.capitalize(), choices=list(options.keys())) if options else
    (gr.Number(label=key.capitalize()) if key in ["age", "familysize"] else gr.Textbox(label="Major"))
    for key, options in demographics.items()
]

terms_and_conditions = gr.Checkbox(label="I agree to the terms and conditions", value=False)

all_inputs = dass_inputs + tipi_inputs + vcl_inputs + demographics_inputs + [terms_and_conditions]

interface = gr.Interface(
    fn=process_and_predict,
    inputs=all_inputs,
    outputs=[
        gr.Textbox(label="Response Status"),
        gr.Textbox(label="Depression Level"),
        gr.Textbox(label="Stress Level"),
        gr.Textbox(label="Anxiety Level"),
    ],
    title="Survey with Predictions",
    description="Submit responses to predict Depression, Stress, and Anxiety levels.",)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)


# if __name__ == "__main__":
#     data=pd.read_csv("survey_results.csv")
#     process_and_predict(data)