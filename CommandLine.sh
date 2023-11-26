
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
