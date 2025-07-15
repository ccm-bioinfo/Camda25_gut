#!/bin/bash

#Este script crea un archivo samples que guarda los nombres de las carpetas de las muestras y luego ejecuta el archivo modify_faa.py para agragar el nombre del taxon a los prokka

rm -f samples.txt

# Set the directory to process
directory="/files2/camda2024/gut/prokka/"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory '$directory' does not exist."
  exit 1
fi

touch ./samples.txt
# Create or overwrite the output file
output_file=./samples.txt

# Get all folders in the directory
folders=$(find "$directory" -type d)

# Save folder names to the file
for folder in $folders; do
  # Get the folder name without the full path
  folder_name="${folder##*/}"
  echo "$folder_name" >> "$output_file"
done

echo "Folder names saved to '$output_file'."

tail -n +2 "$output_file" > "$output_file.tmp" && mv "$output_file.tmp" "$output_file"

#sed '1d' ./samples.txt

cat ./samples.txt | while read line; do
  python modify_faa.py -i /files2/camda2024/gut/prokka/$line/$line"_anotacion.gff" -k /files2/camda2024/gut/taxonomy/assembly/kraken/$line".kraken" -r /files2/camda2024/gut/taxonomy/assembly/reports/$line".report" -f /files2/camda2024/gut/prokka/$line/$line".faa"
done

echo "Listo"
