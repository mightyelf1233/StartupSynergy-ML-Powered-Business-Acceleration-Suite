import tensorflow as tf
from recom_f_promt import recommend_categ_for_prompt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = tf.keras.models.load_model('ann_recom.h5')

# Make predictions

def predict_ann(prompt,loc,size):
    df = pd.read_csv("buisness_data.csv")

    label_encoder = LabelEncoder()
    label_encoder.fit(df['Investors'])
    y_train_encoded = label_encoder.transform(df['Investors'])
    categ = recommend_categ_for_prompt(prompt)[0]
    # Assuming data is your new data represented as a dictionary
    data = {
        "Category": categ,
        "Location": loc,
        "Business Size": size
    }

    # Create a DataFrame from the new data
    new_df = pd.DataFrame(data, index=[0])  # Creating a DataFrame with a single row

    # Concatenate the original DataFrame with the new DataFrame
    combined_df = pd.concat([df, new_df], ignore_index=True)

    # One-hot encode the combined DataFrame
    combined_df_encoded = pd.get_dummies(combined_df, columns=['Category', 'Business Size', 'Location'])

    # Extract the last row (the new data) from the combined one-hot encoded DataFrame
    new_data_encoded = combined_df_encoded.iloc[[-1]]
    new_data_encoded.drop(columns = ["User Name","Investors"], inplace=True)


    predictions = model.predict(new_data_encoded)
    predicted_labels_encoded = predictions.argmax(axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
    return predicted_labels