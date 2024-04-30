import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Function to load data
def load_data(volume_file, products_file, members_file, zip_data_file):
    volume = pd.read_csv(volume_file)
    products = pd.read_csv(products_file)
    members = pd.read_csv(members_file)
    zip_data = pd.read_csv(zip_data_file)
    volume.fillna(0, inplace=True)
    products.fillna(0, inplace=True)
    members['number_of_locations'] = members['number_of_locations'].fillna(1)
    return volume, products, members, zip_data

# Function to prepare data
def prepare_function_data(volume, products, input_date, members, zip_data):
    volume['date'] = pd.to_datetime(volume['date'])
    products['date'] = pd.to_datetime(products['date'])

    # start dates for trailing 12 month and 12 months previous
    t12_start = input_date - pd.DateOffset(months=12)
    p12_start = t12_start - pd.DateOffset(months=12)

    volume_t12 = volume.query("date > @t12_start & date <= @input_date").groupby('nmg_id')['volume'].sum().reset_index(name='t12_volume')
    volume_p12 = volume.query("date > @p12_start & date <= @t12_start").groupby('nmg_id')['volume'].sum().reset_index(name='p12_volume')

    # filter for T12 in products and sum numeric columns
    products_t12 = products.query("date > @t12_start & date <= @input_date")
    products_t12_sum = products_t12.groupby('nmg_id').agg({col: 'sum' for col in products_t12 if col not in ['date', 'nmg_id']}).reset_index()

    final_df = volume_t12.set_index('nmg_id').join(volume_p12.set_index('nmg_id'), on='nmg_id', how='outer', rsuffix='_p12')
    final_df = final_df.join(products_t12_sum.set_index('nmg_id'), on='nmg_id', how='outer')

    #  volume growth calculation
    final_df['t12_volume_growth'] = (final_df['t12_volume'] - final_df.get('p12_volume', 0))

    # filter out useless data points that will hurt knn model
    final_df_filtered = final_df[
        (final_df['t12_volume'] > 0) &
        (final_df['p12_volume'] > 0) &
        (final_df['t12_volume_growth'] / final_df['p12_volume'] <= 0.5)
    ].reset_index()

    # columns to include 't12_' prefix as necessary
    for col in ['credit_card_processing_total', 'inventory_finance_total', 'lease_to_own_total', 'product_protection_total', 'retail_credit_total']:
        if col in final_df_filtered.columns:
            final_df_filtered.rename(columns={col: f't12_{col}'}, inplace=True)
    final_df_filtered['volume_growth']= (final_df_filtered['t12_volume']-final_df_filtered['p12_volume'])/final_df_filtered['p12_volume']
    final_df_filtered = final_df_filtered.fillna(0)
    final_df_filtered = final_df_filtered.drop('t12_volume_growth',axis=1)
    # joining everything
    final_with_members = final_df_filtered.reset_index().merge(members, on='nmg_id', how='outer')

    members['zip_code'] = members['zip_code'].astype(str)
    zip_data['geo_id'] = zip_data['geo_id'].astype(str)

    final_with_zip = final_with_members.merge(zip_data, left_on='zip_code', right_on='geo_id', how='outer')

    final_with_zip = final_with_zip.fillna(0)
    final_with_zip = final_with_zip.drop(['index', 'level_0', 'geo_id'], axis=1, errors='ignore')
    def categorize_locations(value):
      if value == 1:
          return "Single Retailer"
      elif value < 5:
          return "Medium-sized Retailer"
      else:
         return "Large Retailer"

    final_with_zip['number_of_locations'] = final_with_zip['number_of_locations'].apply(categorize_locations)

    return final_with_zip



# Function to find top growing neighbors
# def find_top_growing_neighbors(df, input_nmg_id, n_neighbors=50, top_n=5, volume_weight=10):
#     if 'nmg_id' not in df.columns:
#         raise KeyError("'nmg_id' column not found in DataFrame.")
#     df['nmg_id'] = df['nmg_id'].astype(str)

#     # Convert specified categorical columns to strings
#     categorical_features = ['industry_name', 'number_of_locations']
#     for feature in categorical_features:
#         df[feature] = df[feature].astype(str)

#     numeric_features = ['p12_volume', 'median_income', 'households', 'poverty_rate', 'total_pop', 'vacant_housing_rate', 'income_per_capita']
#     features_to_use = numeric_features + categorical_features

#     # Adjusting preprocessing pipelines
#     numeric_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('scaler', StandardScaler())])

#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#         ('encoder', OneHotEncoder(handle_unknown='ignore'))])

#     # Preprocessing for numeric and categorical data
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)])

#     # Apply preprocessing
#     X = preprocessor.fit_transform(df[features_to_use])

#     # Adjust the weight of t12_volume
#     t12_volume_index = features_to_use.index('p12_volume')  # Get the correct column index for t12_volume
#     X[:, t12_volume_index] *= volume_weight

#     # Fit Nearest Neighbors
#     nn = NearestNeighbors(n_neighbors=n_neighbors)
#     nn.fit(X)

#     input_index = df.index[df['nmg_id'] == input_nmg_id].tolist()
#     if not input_index:
#         raise ValueError(f"No company found with nmg_id: {input_nmg_id}")

#     # Find nearest neighbors
#     distances, indices = nn.kneighbors([X[input_index[0]]])
#     nearest_indices = indices[0][1:]  # Exclude the first index since it's the input itself

#     neighbors_df = df.iloc[nearest_indices].copy()
#     top_growing_neighbors = neighbors_df.sort_values(by='volume_growth', ascending=False).head(top_n)

#     columns_to_display = ['nmg_id', 'company_name', 't12_volume', 'industry_name', 'number_of_locations', 'median_income', 'total_pop', 'volume_growth']
#     return top_growing_neighbors[columns_to_display].reset_index(drop=True)



def evaluate_product_usage(df):
    product_columns = [
        't12_credit_card_processing_total',
        't12_inventory_finance_total',
        't12_lease_to_own_total',
        't12_product_protection_total',
        't12_retail_credit_total'
    ]
    usage_columns = [
        'credit_card_processing',
        'inventory_finance',
        'lease_to_own',
        'product_protection',
        'retail_credit'
    ]
    for usage_col in usage_columns:
        df[usage_col] = 'No'
    for product_col, usage_col in zip(product_columns, usage_columns):
        df[usage_col] = df[product_col].apply(lambda x: 'Yes' if x > 0 else 'No')
    reordered_columns = ['nmg_id']
    for product_col, usage_col in zip(product_columns, usage_columns):
        reordered_columns.extend([usage_col, product_col])
    additional_columns = [col for col in df.columns if col not in reordered_columns and col not in product_columns]
    reordered_columns.extend(additional_columns)
    return df[reordered_columns]

def find_top_growing_neighbors_comp(df, input_nmg_id, n_neighbors=50, top_n=5, volume_weight=10):
    if 'nmg_id' not in df.columns:
        raise KeyError("'nmg_id' column not found in DataFrame.")
    df['nmg_id'] = df['nmg_id'].astype(str)

    # cat to string
    categorical_features = ['industry_name', 'number_of_locations']
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)

    numeric_features = ['p12_volume', 'median_income', 'households', 'poverty_rate', 'total_pop', 'vacant_housing_rate', 'income_per_capita']
    features_to_use = numeric_features + categorical_features

    #  preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    #  apply
    X = preprocessor.fit_transform(df[features_to_use])

    # adjsuting the weight of t12_volume so that volumes are more comprable
    t12_volume_index = features_to_use.index('p12_volume')
    X[:, t12_volume_index] *= volume_weight

    # fitting KNN model
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)

    input_index = df.index[df['nmg_id'] == input_nmg_id].tolist()
    if not input_index:
        raise ValueError(f"No company found with nmg_id: {input_nmg_id}")

    # Find nearest neighbors
    distances, indices = nn.kneighbors([X[input_index[0]]])
    nearest_indices = indices[0][1:]  # Exclude the first index since it's the input itself

    neighbors_df = df.iloc[nearest_indices].copy()
    top_growing_neighbors = neighbors_df.sort_values(by='volume_growth', ascending=False).head(top_n)

    # apply product usage evaluation to the top growing comprable retailers
    top_growing_neighbors_with_product_usage = evaluate_product_usage(top_growing_neighbors)

    # colums
    columns_to_display = [
        'nmg_id',
        'credit_card_processing',
        't12_credit_card_processing_total',
        'inventory_finance',
        't12_inventory_finance_total',
        'lease_to_own',
        't12_lease_to_own_total',
        'product_protection',
        't12_product_protection_total',
        'retail_credit',
        't12_retail_credit_total'
    ]

    return top_growing_neighbors_with_product_usage[columns_to_display].reset_index(drop=True)
def find_services_used_by_nmg(df, input_nmg_id):
    input_nmg_info = df[df['nmg_id'] == input_nmg_id].copy()
    if input_nmg_info.empty:
        raise ValueError(f"No company found with nmg_id: {input_nmg_id}")

    # Determine service usage for the input NMG
    services_used = {
        'credit_card_processing': 'Yes' if input_nmg_info['t12_credit_card_processing_total'].iloc[0] > 0 else 'No',
        'inventory_finance': 'Yes' if input_nmg_info['t12_inventory_finance_total'].iloc[0] > 0 else 'No',
        'lease_to_own': 'Yes' if input_nmg_info['t12_lease_to_own_total'].iloc[0] > 0 else 'No',
        'product_protection': 'Yes' if input_nmg_info['t12_product_protection_total'].iloc[0] > 0 else 'No',
        'retail_credit': 'Yes' if input_nmg_info['t12_retail_credit_total'].iloc[0] > 0 else 'No'
    }

    return pd.DataFrame.from_dict(services_used, orient='index', columns=[f'{input_nmg_id}'])
def find_top_growing_neighbors_same_industry(df, input_nmg_id, n_neighbors=50, top_n=5, volume_weight=10):
    if 'nmg_id' not in df.columns:
        raise KeyError("'nmg_id' column not found in DataFrame.")
    df['nmg_id'] = df['nmg_id'].astype(str)

    # Convert specified categorical columns to strings
    categorical_features = ['industry_name', 'number_of_locations']
    for feature in categorical_features:
        df[feature] = df[feature].astype(str)

    numeric_features = ['p12_volume', 'median_income', 'households', 'poverty_rate', 'total_pop', 'vacant_housing_rate', 'income_per_capita']
    features_to_use = numeric_features + categorical_features

    # Adjusting preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    # Preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Apply preprocessing
    X = preprocessor.fit_transform(df[features_to_use])

    # Adjust the weight of t12_volume
    t12_volume_index = features_to_use.index('p12_volume')  # Get the correct column index for t12_volume
    X[:, t12_volume_index] *= volume_weight

    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)

    input_index = df.index[df['nmg_id'] == input_nmg_id].tolist()
    if not input_index:
        raise ValueError(f"No company found with nmg_id: {input_nmg_id}")

    input_industry = df.loc[input_index[0], 'industry_name']

    # Find nearest neighbors
    distances, indices = nn.kneighbors([X[input_index[0]]])
    nearest_indices = indices[0][1:]  # Exclude the first index since it's the input itself

    # Filter neighbors by industry
    neighbors_df = df.iloc[nearest_indices].copy()
    same_industry_neighbors = neighbors_df[neighbors_df['industry_name'] == input_industry]

    if len(same_industry_neighbors) < top_n:
        return "Not enough neighbors in the same industry."

    top_growing_neighbors = same_industry_neighbors.sort_values(by='volume_growth', ascending=False).head(top_n)

    columns_to_display = ['nmg_id','poverty_rate', 'company_name', 't12_volume', 'industry_name', 'number_of_locations', 'median_income', 'total_pop', 'volume_growth']
    return top_growing_neighbors[columns_to_display].reset_index(drop=True)

# function_data = None
def main():
   
    st.set_page_config(page_title="Retailer Growth Analyzer", page_icon=":chart_with_upwards_trend:")

    # Logo and title
    st.image('nationwide_marketing_group_logo.svg')
    st.title('Retailer Growth Analyzer')
    # Sidebar
    st.sidebar.title('Upload CSV Files')
    # File uploader for CSV files with improved styling
    with st.sidebar.expander("Upload CSV Files", expanded=True):
        volume_file = st.file_uploader('Volume Info CSV', type=['csv'])
        products_file = st.file_uploader('Product Usage CSV', type=['csv'])
        members_file = st.file_uploader('Membership Info CSV', type=['csv'])
        zip_data_file = st.file_uploader('Zip Info CSV', type=['csv'])

    # Input fields
    nmg_id_int = st.number_input('Enter the first nmg_id:', value=0, step=1, format='%d')
    nmg_id = str(nmg_id_int)
    
    input_date_str = st.text_input('Enter the date (YYYY-MM-DD):', '2024-01-01')
    input_date = pd.to_datetime(input_date_str, errors='coerce')  # Convert input date string to datetime

    # # Checkbox for enabling comparison
    # comparison= st.sidebar.checkbox('Enable Comparison')
    # services_used_by_n= st.sidebar.checkbox('Enable Service Usage by Neighbors')
    # customer = st.sidebar.checkbox('Enable Cutomer Information')
    # same_ind=st.sidebar.checkbox('Growing neighbors with same industry')
    st.sidebar.caption('Neighbor Service Utilization Threshold')
    threshold = st.sidebar.number_input(label="Enter Threshold", min_value=0.0,step=1.0)
    
    # Run Analysis button
    if st.sidebar.button('Run Analysis'):
        if volume_file is not None and products_file is not None and members_file is not None and zip_data_file is not None:
            volume, products, members, zip_data = load_data(volume_file, products_file, members_file, zip_data_file)
            function_data = prepare_function_data(volume, products, input_date, members, zip_data)
            function_data['nmg_id']=function_data['nmg_id'].astype('i')
            # Ensure 'industry_name' column is present in function_data
            if 'industry_name' not in function_data.columns:
                st.warning("The 'industry_name' column is missing in the prepared data.")
                return

            # top_neighbors_with_product_usage_1 = find_top_growing_neighbors(function_data, nmg_id)
            find_top_growing_neighbors_same_ind= find_top_growing_neighbors_same_industry(function_data, input_nmg_id=nmg_id)
            # Display customer information based on nmg_id
            
            st.subheader('Customer Information ')
            customer_info_1 = function_data[function_data['nmg_id'] == nmg_id][['nmg_id', 'company_name', 'industry_name', 'median_income', 'volume_growth']]
            st.write(customer_info_1)

               
               
            # Display top growing neighbors for first nmg_id
          
            st.subheader('Additional input Demographic and Volume Information')
            columns_to_keep = [
                'nmg_id',
                'company_name',
                't12_volume',
                'industry_name',
                'number_of_locations',
                'median_income',
                'total_pop',
                'volume_growth']


            filtered_data = function_data[function_data['nmg_id'] == nmg_id][columns_to_keep]

            # the initial nmg_id
            st.write(filtered_data, index=False)
            # st.subheader('Top Growing Neighbors')
            # st.write(top_neighbors_with_product_usage_1.head(10))
            st.subheader('Top Growing Compareable Retailors')
            
            st.write(find_top_growing_neighbors_same_ind)
            
            
                
            top_neighbors_with_product_usage = find_top_growing_neighbors_comp(function_data, nmg_id)

            # Filter the comparison DataFrame to include only the specified services
            services_to_compare = ['credit_card_processing', 'inventory_finance', 'lease_to_own', 
                                'product_protection', 'retail_credit']
            comparison_df = top_neighbors_with_product_usage[['nmg_id'] + services_to_compare] 

            # Get services used by the input NMG ID
            services_used_by_input = find_services_used_by_nmg(function_data, nmg_id).T.squeeze()

            # Filter out services already used by the input NMG ID
            services_not_used_by_input = services_used_by_input[services_used_by_input != 'Yes'].index.tolist()

            # Calculate counts of neighbors using each service that the input NMG ID isn't using
            filtered_comparison_df = comparison_df[comparison_df[services_not_used_by_input] == 'Yes']
            filtered_comparison_df_numeric = filtered_comparison_df.replace({'Yes': 1, 'No': 0}).select_dtypes(include='number')

            # Calculate counts of neighbors using each service that the input NMG ID isn't using
            service_counts = filtered_comparison_df_numeric.sum()

            # Get recommended services where count >= 2
            recommended_services = service_counts[service_counts >= threshold].index.tolist()

            # Create DataFrame for recommended services
            recommendations_df = pd.DataFrame(recommended_services, columns=['Recommended Services'])
            
                # Specify the services to check
            services_to_check = ['credit_card_processing', 'inventory_finance', 'lease_to_own', 'product_protection', 'retail_credit']

                # Create a list of dictionaries to store neighbor information
            neighbor_info = []

                # Iterate over each neighbor and construct the DataFrame
            for idx, neighbor in top_neighbors_with_product_usage.iterrows():
                neighbor_row = {'Neighbor nmg_id': neighbor['nmg_id']}
                for service in services_to_check:
                    neighbor_row[service] = neighbor[service]
                neighbor_info.append(neighbor_row)

            # Convert neighbor_info into a DataFrame
            neighbors_df = pd.DataFrame(neighbor_info)
            # Display service usage by neighbors in a table
            services_used_by_nmg = find_services_used_by_nmg(function_data,nmg_id)

            # Convert to table form
            services_table = services_used_by_nmg.T.rename(columns={0: 'Services'})

            # Print the result
            st.subheader("Services Used by queried Customer:")
            st.write(services_table)
            st.header('Service Usage by Top Growing Compareable Retailors')
            st.table(neighbors_df)
            # Display recommendations
                
            st.subheader('Recommendation')
            if not recommendations_df.empty:
                st.write("Additional Recommended services for nmg_id", nmg_id)
                st.write(recommendations_df)
            else:
                st.write('No services recommended for nmg_id',nmg_id)
                
        else:
            st.warning('Please upload all')


if __name__ == '__main__':
    main()