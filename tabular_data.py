import pandas as pd
from ast import literal_eval

def remove_rows_with_missing_ratings(df):
    
    missing_rating_mask = df[[i for i in df.columns if "rating" in i]].notna().any(axis=1)
    return df[missing_rating_mask]

def remove_rows_with_string(df):

    not_numeric_mask = df["guests"].astype(str).str.isnumeric()

    return df[not_numeric_mask]

def combine_description_strings(df):
    
    def parse_string_to_list(dataframe_element):
        try:
            return literal_eval(dataframe_element)
        except:
            return dataframe_element

    missing_description_mask = df["Description"].notna()
    df = df[missing_description_mask]

    description_list = []
    for i in df["Description"]:
        parsed_list = parse_string_to_list(i)
        if isinstance(parsed_list, list):
            parsed_list = parsed_list[1:]
            parsed_list = " ".join([j for j in parsed_list if j !=""])
        else:
            parsed_list = str(parsed_list)

        description_list.append(parsed_list)
    
    df["Description"] = pd.Series(description_list)

    return df

def set_default_features_values(df):

    field_list = ["guests","beds","bathrooms","bedrooms"]
    for i in field_list:
        df[i].fillna(1, inplace=True)
    
    return df

def clean_tabular_data(df):
    
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_features_values(df)
    df = remove_rows_with_string(df)

    return df

def load_airbnb(df, label_fieldname):
    if label_fieldname in df.columns:

        numerical_features_fieldname_list = ['guests', 'beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating',
                                          'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating',
                                          'Value_rating', 'amenities_count', 'bedrooms']
        if label_fieldname != "Category":
            numerical_features_fieldname_list.remove(label_fieldname)
        return(df[numerical_features_fieldname_list], df[label_fieldname])
    else:
        print("Specified fieldname does not exist in the dataframe")
        return False

def main():
    df = pd.read_csv("tabular_data/listing.csv")
    df = clean_tabular_data(df)
    df = df.astype({"guests": "int32", "bedrooms": "int32"})
    print(df.describe())
    df.to_csv("tabular_data/clean_tabular_data.csv", index=False)

if __name__ == "__main__":
    main()
    
    
    
"""   
    df = pd.read_csv("tabular_data/listing.csv")
    df = clean_tabular_data(df)
    df.to_csv("tabular_data/clean_tabular_data.csv", index=False)
"""