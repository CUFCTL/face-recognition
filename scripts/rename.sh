counter=0

for folder in $1/*; do
	for file in "$folder"/*; do
		counter=$((counter+1))
		mv "$file" "$folder/$counter.ppm"
	done
	counter=$((counter-counter))
done
