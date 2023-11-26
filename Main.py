#!/usr/bin/env python
# coding: utf-8

# # 1. Data Collection

# ## Importing necessary libraries

# In[1]:


import bs4
from bs4 import BeautifulSoup
import requests
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import time
import os
from tqdm.notebook import tqdm
import hashlib
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
import pickle
from collections import Counter
from functools import reduce
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from forex_python.converter import CurrencyRates
import folium
from geopy.geocoders import Nominatim
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt


# ## Making the List of Masters degree URLs
# To avoid getting timedout by the host website, used a time delay of 2s per request.after going through the first 400 pages, saved to URLs in master_urls.txt file.

# In[2]:


def get_master_urls(num_pages=400, use_cached_data=True):
    base_url = "https://www.findamasters.com/masters-degrees/msc-degrees/"
    master_urls = []

    if use_cached_data and os.path.exists("master_urls.txt"):
        # Load existing data from the saved file
        with open("master_urls.txt", 'r') as file:
            master_urls = [line.strip() for line in file]

    else:
        for page in range(1, num_pages + 1):
            page_url = f"{base_url}?PG={page}"
            print(page_url)
            time.sleep(2)
            response = requests.get(page_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                course_links = soup.select('.courseLink')

                for link in course_links:
                    relative_url = link['href']
                    course_url = f"https://www.findamasters.com{relative_url}"
                    master_urls.append(course_url)

    return master_urls

def save_to_txt(master_urls, output_file="master_urls.txt"):
    with open(output_file, 'w') as file:
        for url in master_urls:
            file.write(url + '\n')


# In[3]:


master_urls = get_master_urls()
save_to_txt(master_urls)


# ## Crawling the website to get the corresponding html
# 

# The code sends one request at a time and downloads the html file. If there is no such directory, the directory will be created as the code gets executed.The program sends multiple requests at a time using ThreadPoolExecutor which speeds up the downloading process. Used tqdm to monitor the progress.

# In[ ]:


def download_and_save_html(url, folder_path):
    try:
        response = requests.get(url, timeout=10)  # Set a timeout value
        response.raise_for_status()  # Check for HTTP errors
    except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
        print(f"Error: {e}. Retrying after a delay...")
        time.sleep(5)
        # Retry the request
        return download_and_save_html(url, folder_path)

    if response.status_code == 200:
        # Generate a unique filename using the hash of the URL wanted to genated back the url from the hash, didnt workout, so this is UNNECESSARY
        file_hash = hashlib.md5(url.encode()).hexdigest()
        file_path = os.path.join(folder_path, f"{file_hash}.html")

        # Create folder if doesn't exists
        os.makedirs(folder_path, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)

def save_html_for_master_urls(file_name="master_urls.txt"):
    with open(file_name, 'r') as file:
        master_urls = [line.strip() for line in file]

    main_folder_path = "html_files"
    os.makedirs(main_folder_path, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        futures = []
        for page_number, url in enumerate(master_urls, start=1):
            subfolder_path = os.path.join(main_folder_path, f"page_{page_number}_html")
            futures.append(executor.submit(download_and_save_html, url, subfolder_path))

        # Wait for all threads to complete
        for future in tqdm(futures, desc="Downloading HTML", total=len(futures)):
            future.result()

# Specify the filename for your master_urls.txt file (assuming it's in the current directory)
master_urls_filename = "master_urls.txt"

save_html_for_master_urls(master_urls_filename)


# ## Parsing Downloaded pages

# First extracted all the desired data from each of the pages one by one created a tsv file for each of the pages. used try and except to extarct data nd in case of missing data stored empty sting. Saved all the tsv files in a sub folder called tsv_files inside the main folder html_files. Again used tqdm to monitor progress.

# In[4]:


def extract_data_from_html(html_content, url):
    # Initialize a dictionary to store the extracted data
    extracted_data = {
        "courseName": "",
        "universityName": "",
        "faculty_name": "",
        "IsItFullTime": "",
        "Description": "",
        "StartDate": "",
        "fees": "",
        "modality": "",
        "duration": "",
        "city": "",
        "country": "",
        "administration": "",
        "url":url,
    }

    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract data using appropriate CSS selectors
    extracted_data["courseName"] = soup.find('h1', class_='course-header__course-title').get_text(strip=True)
    extracted_data["universityName"] = soup.find('a', class_='course-header__institution').get_text(strip=True)
    try:
        extracted_data['Description'] = soup.find("div", class_ = "course-sections__description").find("div", {"class": "course-sections__content"}).get_text(strip = True)
    except:
        extracted_data['Description'] = ''
    try:    
        extracted_data["faculty_name"] = soup.find('a', class_ = 'course-header__department').get_text(strip=True)
    except:
        extracted_data["faculty_name"] =''
    try:    
        extracted_data["IsItFullTime"] = soup.find('span', class_='key-info__content').get_text(strip=True)
    except:
        extracted_data["IsItFullTime"] = ''
    try:
        extracted_data['fees'] = soup.find("div", class_ = "course-sections__fees").get_text(strip=True)
    except:
        extracted_data['fees'] = ''
    try:    
        extracted_data['StartDate'] = soup.find('span', class_='key-info__start-date').get_text(strip=True)
    except:
        extracted_data['StartDate'] = ''
    try:
        extracted_data['modality'] = soup.find('span', class_='key-info__qualification').get_text(strip=True)
    except:
        extracted_data['modality'] = ''
    try:
        extracted_data['duration'] = soup.find('span', class_='key-info__duration').get_text(strip=True)
    except:
        extracted_data['duration'] = ''
    try:    
        extracted_data["city"] = soup.find('a', class_='course-data__city').get_text(strip=True)
    except:
        extracted_data["city"] = ''
    try:    
        extracted_data["country"] = soup.find('a', class_='course-data__country').get_text(strip=True)
    except:
        extracted_data["country"] = ''
    try:    
        extracted_data["administration"] = soup.find('a', class_='course-data__on-campus').get_text(strip=True)
    except:
        extracted_data["administration"] = ''
    return extracted_data


def process_html_files(main_folder_path, output_folder="tsv_files", subfolder_pattern="page_{}_html"):
    # Initialize a list to store the extracted data from all pages
    all_extracted_data = []

    # Iterate through possible page numbers
    for page_number in tqdm(range(1, 6001), desc="Processing HTML files", unit="page"):  # Adjust the range based on your actual data
        subfolder_name = subfolder_pattern.format(page_number)
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        output_path = os.path.join(main_folder_path, output_folder)
        os.makedirs(output_path, exist_ok=True)
        master_urls_filename = "master_urls.txt"
                
        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Get the list of HTML files in the subfolder
            html_files = [file for file in os.listdir(subfolder_path) if file.endswith(".html")]
           
        # Read the master_urls.txt file line by line
            with open(master_urls_filename, 'r') as urls_file:
                url = urls_file.readlines()[page_number - 1].strip()

                    
        # Iterate through each HTML file
                for html_filename in html_files:
                    html_filepath = os.path.join(subfolder_path, html_filename)


                    # Read the HTML content from the file
                    with open(html_filepath, 'r', encoding='utf-8') as html_file:
                        html_content = html_file.read()
                    
                    # Extract data from the HTML content
                    extracted_data = extract_data_from_html(html_content, url)

                # Append the extracted data to the list
                all_extracted_data.append(extracted_data)
    # Create a DataFrame from the list of dictionaries
    coursedf = pd.DataFrame(all_extracted_data)

    # Save the DataFrame to a TSV file in the output folder
    tsv_filename = os.path.join(output_path, "coursedf.tsv")
    coursedf.to_csv(tsv_filename, sep='\t', index=False)

    return coursedf


# In[5]:


# Specify the main folder path containing subfolders with HTML files
main_folder_path = "html_files"

# Call the function to process HTML files and extract data
coursedf = process_html_files(main_folder_path)
coursedf[:9]


# # 2. Search Engine
# ## Preprocessing

# ### Preprocessing The text

# Downloading NLTK packages for stemming and removing stopwords

# In[ ]:


# Download NLTK resources (run this once)
nltk.download('stopwords')
nltk.download('punkt')


# In[6]:


# Initialize NLTK's Porter Stemmer and English stopwords
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation, and apply stemming
    filtered_tokens = [porter.stem(token.lower()) for token in tokens if (token.lower() not in stop_words) and (token.lower() not in string.punctuation)]
    return filtered_tokens

# Apply text preprocessing to the 'Description' column
coursedf['formatted_Description'] = coursedf['Description'].apply(preprocess_text)
coursedf


# ### Preprocessing the Fees

# Using Forex-python to get realtime exchange rates and convert the currencies to USD.

# In[25]:


# Set display options for floating-point numbers
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Function to fetch real-time exchange rates
def get_exchange_rate(currency):
    c = CurrencyRates()
    try:
        rate = c.get_rate(currency, 'USD')  # Corrected order of arguments
        return rate
    except:
        return 1.0  # Default to 1.0 if rate not available

# Define a function to convert fee to USD
def convert_to_usd(row):
    if pd.notnull(row['fee']) and pd.notnull(row['currency']):
        # Convert fee to USD using real-time exchange rate
        usd_fee = row['fee'] * get_exchange_rate(row['currency']) * 100
        return usd_fee
    else:
        return np.nan

# Define a function to extract fee values and currencies
def extract_fees(row):
    # Regular expression to find currency symbols and amounts
    pattern = r'([£$€]{1})\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    
    # Find all matches in the 'fees' column
    matches = re.findall(pattern, row['fees'])
    
    # If no matches, return NaN
    if not matches:
        return pd.Series([np.nan, np.nan], index=['fee', 'currency'])
    
    # Extract fee values and currencies
    fees = [float(match[1].replace(',', '')) for match in matches]
    currencies = [match[0] for match in matches]
    
    # Return the highest fee and its currency
    max_fee_index = fees.index(max(fees))
    return pd.Series([fees[max_fee_index], currencies[max_fee_index]], index=['fee', 'currency'])

# Apply the extract_fees function to the DataFrame
coursedf[['fee', 'currency']] = coursedf.apply(extract_fees, axis=1)

# Convert the 'fee' column to USD
coursedf['fee_usd'] = coursedf.apply(convert_to_usd, axis=1)

# Convert the 'fee_usd' column to string and replace NaN values with 'Not Available'
coursedf['fee_usd'] = coursedf['fee_usd'].astype(str).replace('nan', 'Not Available')
# Display the updated DataFrame
coursedf['fee_usd']


# ### Vocabolary

# Created a vocabolary of unique words from the descriptions and asssigned a term ID for each word.

# In[8]:


# Assuming 'description_list' is a list of tokenized descriptions
all_words = [word for description in coursedf['formatted_Description'] for word in description]

# Remove duplicates to get unique words
unique_words = list(set(all_words))

# Build vocabulary mapping each unique word to a term_id
vocabulary = {word: term_id for term_id, word in enumerate(unique_words)}

# Print the first few entries in the vocabulary for verification
for word, term_id in list(vocabulary.items())[:]:
    print(f"Word: {word}, Term ID: {term_id}")


# ### Inverted Index
# created inverted index for each term ID.

# In[9]:


# Create an inverted index
inverted_index = defaultdict(list)  # Use a list to store document indices

# Iterate through the columns (terms) in the DataFrame and update the inverted index
for term_id, term in enumerate(tqdm(unique_words, desc="Building Inverted Index")):
    for index, row in coursedf.iterrows():
        if term in row['formatted_Description']:
            inverted_index[term_id].append(index)

# Convert the inverted index to a regular dictionary
inverted_index = dict(inverted_index)

# Ensure all term IDs are present, even if they have no associated documents
all_term_ids = set(range(len(unique_words)))
for term_id in all_term_ids:
    if term_id not in inverted_index:
        inverted_index[term_id] = []

# Print the inverted index in the desired format
formatted_inverted_index = {term_id: documents for term_id, documents in inverted_index.items()}

# Print the formatted inverted index for verification
print(formatted_inverted_index)


# ### Saving Inverted Index
# Using pikle to to save our created inverted Index.

# In[10]:


# Save the inverted index to a file
with open('inverted_index.pkl', 'wb') as f:
    pickle.dump(formatted_inverted_index, f)


# In[11]:


# Load the inverted index from the file
with open('inverted_index.pkl', 'rb') as f:
    inverted_index = pickle.load(f)


# ## 1st search engine!
#  Here we are using query words to search the description of the courses to find matches. It is searching whether the query words are present in the description of the course.  When there are multiple query words, does individual word searches and then shows the common ones! rejects stop words.

# In[16]:


def search(query):
    # Tokenize the search query
    query_terms = query.split()
    query_terms = preprocess_text(query)

    # Initialize a list to store individual search results
    individual_results = []

    # Iterate through each query term
    for term in query_terms:
        term_id = vocabulary.get(term)
        if term_id is not None:
            # Retrieve the documents containing the term from the inverted index
            documents = inverted_index.get(term_id, [])
            individual_results.append(set(documents))

    # Find the common document indices
    common_document_indices = set.intersection(*individual_results) if individual_results else set()

    # Retrieve the corresponding rows from the DataFrame
    search_results = coursedf.loc[list(common_document_indices)]

    return search_results

# Example search query
user_query = input("Enter your search query: ")

# Perform the search
search_results = search(user_query)

# Display the search results
print("\nSearch Results:")
search_results[['courseName', 'universityName', 'Description', 'url']]


# ### New Inverted Index using tfidf.

# In[12]:


# Use TfidfVectorizer directly on the 'formatted_Description' column
tfidf_vectorizer = TfidfVectorizer(input='content', lowercase=False, tokenizer=lambda text: text)
tfidf_matrix = tfidf_vectorizer.fit_transform(tqdm(coursedf['formatted_Description'], desc="Computing TF-IDF"))

# Convert the TF-IDF matrix to a dense DataFrame
tfidf_data = pd.DataFrame(tfidf_matrix.todense(), index=coursedf.index, columns=tfidf_vectorizer.get_feature_names_out())

# Initialize an inverted index with TF-IDF values
inverted_index_with_tfidf = {}

# Iterate through the columns (terms) and populate the inverted index
for term_id, term in enumerate(tqdm(tfidf_data.columns, desc="Building Inverted Index")):
    for index, tfidf_value in tfidf_data[term].iteritems():
        # Consider only non-zero TF-IDF values
        if tfidf_value > 0:
            # Append (document, TF-IDF) tuple to the inverted index
            if term_id not in inverted_index_with_tfidf:
                inverted_index_with_tfidf[term_id] = []
            inverted_index_with_tfidf[term_id].append((index, tfidf_value))

# Convert the inverted index to the desired format
inverted_index_with_tfidf = {term_id: documents for term_id, documents in inverted_index_with_tfidf.items()}

# Print the first few entries in the inverted index for verification
for term_id, documents in list(inverted_index_with_tfidf.items())[:]:
    print(f"Term ID: {term_id}, Documents: {documents}")


# ### saving Inverted Index with tfidf

# In[13]:


# Save the inverted index to a file
with open('inverted_index_tfidf.pkl', 'wb') as f:
    pickle.dump(inverted_index_with_tfidf, f)


# In[14]:


# Load the inverted index from the file
with open('inverted_index_tfidf.pkl', 'rb') as f:
    inverted_index_with_tfidf = pickle.load(f)


# ## 2nd Search engine!
# Here we convert our query into a tfidf vector and use cosine similarity to find best matches.

# In[15]:


# Sample query
query = input("What is your query?:")
# Tokenize and preprocess the search query
query_terms = preprocess_text(query)

# Use the TfidfVectorizer directly on the 'formatted_Description' column
tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda text: text)
tfidf_matrix = tfidf_vectorizer.fit_transform(coursedf['formatted_Description'])

# Convert the query to TF-IDF representation
query_tfidf_vector = tfidf_vectorizer.transform([query_terms])

# Calculate cosine similarity between the query vector and all document vectors
cosine_similarities = cosine_similarity(query_tfidf_vector, tfidf_matrix)

# Get the indices of the sorted similarity scores in descending order
sorted_indices = np.argsort(cosine_similarities[0])[::-1]

# Display the details of the top matching documents and their similarity scores
top_matches = coursedf.iloc[sorted_indices[:5]]
top_matches['similarity'] = cosine_similarities[0][sorted_indices[:5]]
# Filter out rows with similarity score of 0
filtered_top_matches = top_matches[top_matches['similarity'] != 0]

filtered_top_matches[['courseName', 'universityName', 'Description', 'url', 'similarity']]


# ## 3. Define a new score!

# The first scoring function might capture more varied results based on keyword presence, while the second might provide more nuanced results based on semantic similarity. The choice between the two could depend on the specific requirements of the search application and the nature of the dataset.
# 

# In[26]:


# Define a scoring function
def calculate_score(document):
    # Initialize score
    score = 0
    
    # Example weights
    weight_tfidf_similarity = 0.6
    weight_fees = 0.1
    weight_duration = 0.1
    weight_modality = 0.1
    weight_location = 0.1
    
    # Calculate TF-IDF similarity score (assuming 'tfidf_similarity' is a pre-calculated value)
    tfidf_similarity_score = document['tfidf_similarity'] if 'tfidf_similarity' in document else 0
    
    # Consider other attributes (fees, duration, modality, location) and adjust the score
    fees_score = document['fees_score'] if 'fees_score' in document else 0
    duration_score = document['duration_score'] if 'duration_score' in document else 0
    modality_score = document['modality_score'] if 'modality_score' in document else 0
    location_score = document['location_score'] if 'location_score' in document else 0
    
    # Calculate the overall score using weighted sum
    score = (
        weight_tfidf_similarity * tfidf_similarity_score +
        weight_fees * fees_score +
        weight_duration * duration_score +
        weight_modality * modality_score +
        weight_location * location_score
    )
    
    return score


# In[29]:


# Define a scoring function based on description length and fees
def calculate_score(row):
    # Scoring based on description length
    description_score = len(row['formatted_Description'])
    
    # Scoring based on fees (if available)
    fees = row['fee_usd']
    if fees != 'Not Available':
        # Extract numeric value from fees (considering only USD)
        fee_amount = float(fees[1:])  # Remove currency symbol
        # Higher fees will have lower scores, inversely proportional
        fees_score = 1 / fee_amount if fee_amount > 0 else 0
    else:
        # Assign a default score for unavailable fees
        fees_score = 0.5  # Adjust as needed
    
    # Total score - combine both scores
    total_score = (description_score + fees_score) / 2  # Normalize to get an average score
    
    # Normalize score between 0 and 1 using min-max normalization
    min_score = 0
    max_score = max(description_score, 1)  # To avoid division by zero
    normalized_score = (total_score - min_score) / (max_score - min_score)
    
    return normalized_score


query = input("Enter your search query: ")
query_documents = search(query)  # Assume this retrieves the relevant documents

# Calculate scores for each document
tqdm.pandas(desc='Calculating Scores')  # Use tqdm with pandas for progress bar
query_documents['score'] = query_documents.progress_apply(calculate_score, axis=1)

# Get the top-k documents based on the score
top_k = 5  # Specify the number of top documents needed
top_documents = query_documents.nlargest(top_k, 'score')

# Display the top-k documents
print("\nTop Documents:")
top_documents[['courseName', 'universityName', 'Description', 'url', 'score']]


# # 4.  Visualizing the most relevant MSc degrees

# In[2]:


pip install geopy


# In[30]:


# Function to get coordinates using geopy
def get_coordinates_geopy(city, country):
    try:
        geolocator = Nominatim(user_agent="my_geocoder")
        location = geolocator.geocode(f"{city}, {country}")
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except Exception as e:
        print(f"Error geocoding {city}, {country}: {e}")
        return None
    
query = input("Enter your search query: ")
query_documents = search(query)  # Assume this retrieves the relevant documents

# Calculate scores for each document
tqdm.pandas(desc='Calculating Scores')  # Use tqdm with pandas for progress bar
query_documents['score'] = query_documents.progress_apply(calculate_score, axis=1)

geodf = pd.merge(query_documents, coursedf[['courseName', 'universityName', 'Description', 'url']],
                     on=['courseName', 'universityName', 'Description', 'url'], how='left')

# Get coordinates for each course using geopy
geodf['coordinates'] = geodf.apply(lambda row: get_coordinates_geopy(row['city'], row['country']), axis=1)
geodf = geodf.dropna(subset=['coordinates']) # Drop rows with missing coordinates


# Convert 'fees_extracted' to numeric and define a color scale based on fees
geodf['fee_usd'] = pd.to_numeric(geodf['fee_usd'], errors='coerce')
min_fee = geodf['fee_usd'].min()
max_fee = geodf['fee_usd'].max()

# Function to get fee color based on a colormap
def get_fee_color(fee):
    normalized_fee = (fee - min_fee) / (max_fee - min_fee)
    color = plt.cm.autumn(normalized_fee)
    hex_color = "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    return hex_color

# Create a folium map centered around the first coordinate
map_center = geodf['coordinates'].iloc[0]
mymap = folium.Map(location=map_center, zoom_start=5)

# Add a MarkerCluster to group nearby markers
marker_cluster = MarkerCluster().add_to(mymap)

# Function to add markers to the map
def add_markers(row):
    fee_color = get_fee_color(row['fee_usd'])
    folium.Marker(
        location=row['coordinates'],
        popup=f"Course: {row['courseName']} - {row['universityName']}\nScore: {row['score']}\nFees: {row['fee_usd']}",
        icon=folium.Icon(color='white', icon_color=fee_color)
    ).add_to(marker_cluster)

# Add a legend for fees
fees_legend = folium.Element(
    """<div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; background-color:white; padding: 10px; border:2px solid grey;">
       <p><strong>Legend: Fees</strong></p>
       <p style="color:red">Low Fees</p>
       <p style="color:orange">Moderate Fees</p>
       <p style="color:yellow">High Fees</p>
   </div>"""
)

mymap.get_root().html.add_child(fees_legend)

# Apply the add_markers function to each row in geodf
geodf.apply(add_markers, axis=1)

mymap.save("msc_map.html")
print("Map saved as msc_map.html")


# The user starts by entering a search query to find information on relevant courses and then a scoring system is applied to evaluate the relevance of the documents related to the query, as done in the previous exercise. Then, after combining the coursedf dataset with the query results, the geographical coordinates for each course are obtained by using the get_coordinates_geopy function. 
# 
# For encoding the fees, a visual representation is used based on a color scale, where the courses with lowest fees take the redest colors and the ones with higher fees take the yellowest colors. A legend is plotted to interpret the color scale, making it easy for users to understand the fee range from low to high.
# 
# Another important step was using a MarkerCluster to prevent overcrowding, as lots of courses are taken at the same city, and this way nearby course markers are grouped for a more organized map view.
# 
# The final map is saved as an HTML file ("msc_map.html"). It visually represents MSc courses related to the user's search query, providing a geographical overview of the courses locations, in addition to a popup displaying some relevant information such as course name, university, score, and fees. Moreover, the marker color corresponds to the fees, providing a quick visual reference.
# 

# # Command Line Question

# In[ ]:


#!/bin/bash

# Initialize an empty file to store merged data
: > merged_courses.tsv

# Loop through each of the 6000 HTML files
for i in {1..6000}; do
  folder_path="HTML_folders/page_$i"
  tsv_file="html_$i.html.tsv"

  # Handle the first file separately to keep the header
  if [[ $i -eq 1 ]]; then
    cat "$folder_path/$tsv_file" > merged_courses.tsv
  else
    # Skip the header for the rest of the files
    sed 1d "$folder_path/$tsv_file" >> merged_courses.tsv
  fi
done

echo "Merged file creation complete."

# Analyze the data

# Question-1: Country Analysis
echo "# Question-1: Country Analysis"

# Process the data to count courses per country
awk -F'	' 'FNR > 1 { count[$11]++ } END { for (c in count) print c, count[c] }' merged_courses.tsv | sort -nrk2 | {
  read -r top_country top_count
  echo "Most frequent country: $top_country with $top_count courses"

  # Find the top cities in the most frequent country
  awk -F'	' -v country="$top_country" '$11 == country { city[$10]++ } END { for (c in city) print c, city[c] }' merged_courses.tsv | sort -nrk2 | head -n5
}

# Question-2: Part-Time Course Count
echo "# Question-2: Part-Time Course Count"

# Count the number of part-time courses
part_time_count=$(awk -F'	' '$4 == "Part time" { count++ } END { print count }' merged_courses.tsv)
echo "Part-time courses count: $part_time_count"

# Question-3: Engineering Course Analysis
echo "# Question-3: Engineering Course Analysis"

# Count courses with 'Engineer' in the name
engineering_count=$(grep -c "Engineer" merged_courses.tsv)
percentage=$(awk "BEGIN {printf "%.2f", $engineering_count/6000*100}")
echo "Engineering courses make up $percentage% of all courses."


# # Algorithmic Question

# **1- Implement a code to solve the above mentioned problem.**

# In[35]:


def generate_report(days, sum_hours, limits, current_schedule, current_day, possible_schedule):
    if current_day == days:
        if sum_hours == 0:
            print("YES")
            print(' '.join(map(str, current_schedule)))
            possible_schedule[0] = True
        return

    min_hours, max_hours = limits[current_day]

    for hours in range(min_hours, min(max_hours, sum_hours) + 1):
        remaining_hours = sum_hours - hours
        new_schedule = current_schedule.copy()
        new_schedule.append(hours)

        generate_report(days, remaining_hours, limits, new_schedule, current_day + 1, possible_schedule)

        if possible_schedule[0]:
            return

if __name__ == '__main__':
    days, sum_hours = map(int, input().split())
    limits =  []
    for i in range (days):
        min_time, max_time = map(int, input().split())
        limits.append((min_time, max_time))

    min_total = sum(limit[0] for limit in limits)
    max_total = sum(limit[1] for limit in limits)
    possible_schedule = [False]
    
    if sum_hours < min_total or sum_hours > max_total:
        print("NO")
    else:
        generate_report(days, sum_hours, limits, [], 0, possible_schedule)


# **2- What is the time complexity (the Big O notation) of your solution? Please provide a detailed explanation of how you calculated the time complexity.**

# Let's analyze the time complexity of the algorithm by parts:
# - The inizialization part takes constant time to read the first two integers and to inizialize the operators. Regarding the loop that reads and appends the limits, it takes linear time O(D) with D being the number of days, as well as calculating the *min_total* and the *max_total*, that also takes O(D) time.
# - The complexity of the function generate_reports will be determined by the number of recursive calls made by the function. If we denote D the total number of days and M the difference between the minimum and the maximum possible hours for each day, the worst case complexity will be exponential O((M+1)^D), where (M+1) represents the number of choices for hours each day and D is the number of days. If we break it down, we will see that:
# - - At each day, the function iterates from min_hours to min(max_hours, sum_hours) + 1. This is at most M + 1 choices.
# - - The function is called recursively for each choice, resulting in a branching factor of M + 1.
# - - This recursion happens for each day, and there are D days.
# - Therefore, the overall time complexity of the hole algorithm is O((M+1)^D), but the complexity might be lower in practice as when the algorithm finds an appropiate schedule, it stops the recursion. Moreover, if the schedule is not feasable, the algorithm won't call the generate_report function so the overall complexity will be O(N).

# **3- Ask ChatGPT or any other LLM chatbot tool to check your code's time complexity (the Big O notation). Compare your answer to theirs. Do you believe this is correct? If the two differ, which one is right? (why?)**

# This is the response obtained after asking ChatGPT:
# 
# Let's denote:
# 
# - `D` as the number of days.
# - `H` as the sum of hours.
# - `M` as the maximum difference between the minimum and maximum hours for each day (i.e., `M = max(max_hours - min_hours)`).
# 
# 1. **Reading Input:**
#    - Reading input takes O(D) time because you iterate over `days`.
# 
# 2. **Calculating Sums:**
#    - Calculating `min_total` and `max_total` takes O(D) time because you iterate over `limits`.
# 
# 3. **Recursive Function:**
#    - The `generate_report` function is recursive and explores all possible combinations of hours for each day.
#    - In the worst case, the function explores all possibilities for each day, leading to a branching factor of `(M+1)` (number of choices for hours each day).
#    - The depth of the recursion is `D` (number of days).
#    - Therefore, the overall time complexity of the recursive function is O((M+1)^D).
# 
# The dominant factor in the overall time complexity is the recursive function, which gives the overall time complexity of O((M+1)^D).
# 
# Keep in mind that this analysis assumes that the recursive function explores all possible combinations. If there are conditions or constraints that limit the number of recursive calls, the effective time complexity may be lower in practice.

# If we compare our answers, we can see that we have arrived at the same conclusion, so we believe that ChatGPT's answer is also correct.

# **4- What do you think of the optimality of your code? Do you believe it is optimal? Can you improve? Please elaborate on your response.**

# The recursion leads to an exponential time complexity which can lead to expensive time complexity for large input values. 
# However, there may be some strategies that can be considered to potentially improve the performance of the solution:
# - The efficiency might be improved by applying memoization techniques to store and reuse the results of previous computations and the same subproblems won't be recomputed multiple times. 
# - Maybe instead of using recursion, dynamic programming with a bottom-up approach could be used. This strategy constists on solving subproblems for smaller instances of the problem and then building up to the main problem.
# 
# Despite that, we have tried to apply this changes to the code or building another code using this strategies, but the time complexity hasn't improve much. Taking into consideration that the worst case won't happen frequently, and that when the schedule is not possible, the solution only takes linear time, it will only cost exponential time in few cases.
