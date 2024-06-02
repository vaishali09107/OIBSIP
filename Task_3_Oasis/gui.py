import tkinter as tk
from tkinter import ttk
import pandas as pd
import pickle

# Load the model and preprocessor
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

def predict():
    car_name = entry_car_name.get()
    year = int(entry_year.get())
    present_price = float(entry_present_price.get())
    driven_kms = int(entry_driven_kms.get())
    fuel_type = entry_fuel_type.get()
    selling_type = entry_selling_type.get()
    transmission = entry_transmission.get()
    owner = int(entry_owner.get())

    data = {
        "Car_Name": car_name,
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner
    }

    features = pd.DataFrame([data])
    features_preprocessed = preprocessor.transform(features)
    prediction = model.predict(features_preprocessed)
    label_result.config(text=f'Predicted Selling Price: {prediction[0]:.2f} Lakhs')

# Create the GUI
root = tk.Tk()
root.title("Car Price Prediction")

ttk.Label(root, text="Car Name:").grid(row=0, column=0, padx=10, pady=10)
entry_car_name = ttk.Entry(root)
entry_car_name.grid(row=0, column=1, padx=10, pady=10)

ttk.Label(root, text="Year:").grid(row=1, column=0, padx=10, pady=10)
entry_year = ttk.Entry(root)
entry_year.grid(row=1, column=1, padx=10, pady=10)

ttk.Label(root, text="Present Price (in Lakhs):").grid(row=2, column=0, padx=10, pady=10)
entry_present_price = ttk.Entry(root)
entry_present_price.grid(row=2, column=1, padx=10, pady=10)

ttk.Label(root, text="Driven Kms:").grid(row=3, column=0, padx=10, pady=10)
entry_driven_kms = ttk.Entry(root)
entry_driven_kms.grid(row=3, column=1, padx=10, pady=10)

ttk.Label(root, text="Fuel Type:").grid(row=4, column=0, padx=10, pady=10)
entry_fuel_type = ttk.Entry(root)
entry_fuel_type.grid(row=4, column=1, padx=10, pady=10)

ttk.Label(root, text="Selling Type:").grid(row=5, column=0, padx=10, pady=10)
entry_selling_type = ttk.Entry(root)
entry_selling_type.grid(row=5, column=1, padx=10, pady=10)

ttk.Label(root, text="Transmission:").grid(row=6, column=0, padx=10, pady=10)
entry_transmission = ttk.Entry(root)
entry_transmission.grid(row=6, column=1, padx=10, pady=10)

ttk.Label(root, text="Owner:").grid(row=7, column=0, padx=10, pady=10)
entry_owner = ttk.Entry(root)
entry_owner.grid(row=7, column=1, padx=10, pady=10)

button_predict = ttk.Button(root, text="Predict", command=predict)
button_predict.grid(row=8, column=0, columnspan=2, pady=10)

label_result = ttk.Label(root, text="")
label_result.grid(row=9, column=0, columnspan=2, pady=10)

root.mainloop()
