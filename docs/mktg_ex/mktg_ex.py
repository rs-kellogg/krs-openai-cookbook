#######################
# Toy Marketing Study #
#######################

# libraries
from openai import OpenAI
import pandas as pd
import os
import base64
import csv
from PIL import Image, ImageEnhance


# Load your API key
with open('~/api_key.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

#########
# Inputs
#########

input_image = "select.png"
enhanced_image = "select_enhanced.png"

personas = "simulated_survey_personas.csv"
output_file = "results.csv"

############
# Functions
############cle

def preprocess_image(image_path, enhanced_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Enhance the image contrast without converting to grayscale
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        # Save the enhanced image
        enhanced_image_path = enhanced_path
        img.save(enhanced_image_path)
        return enhanced_image_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to get the response from the GPT-4 model
def get_response(persona, image_base64):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": f"You are a participant in a marketing study from Monkeys Elbow, Kentucky. Here is your persona: {persona}"}]},
            {"role": "user", "content": [{"type": "text", "text": "Please select the image you find the most appealing from these three and explain your choice.\nThe image on the left can be called \"kawaii\", the image in the middle \"cubs\", and the image on the right \"futuristic\"."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"}
    )
    #return response.choices[0].message['content']
    return response.choices[0].message.content

# save results to a df
def save_results(results_df, id, persona, response):
    # create df from current row
    row_df = pd.DataFrame({
        'id': [id],
        'persona': [persona],
        'response': [response]
    })
    # combine
    results_df = pd.concat([results_df, row_df], ignore_index=True)
    # return dataframe
    return results_df

#######
# Run
#######

def main():
    results_df = pd.DataFrame(columns=['id', 'persona', 'response'])
    personas_df = pd.read_csv(personas)  

    image = preprocess_image(input_image, enhanced_image)
    base64_image = encode_image(image)

    for index, row in personas_df.iterrows():
        id = row['ID']
        print("Processing ID:", id)
        persona = row['Description']
        response = get_response(persona, base64_image)
        print(response)
        results_df = save_results(results_df, id, persona, response)

        # save results after each persona is processed
        results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
