import csv

def text_to_csv(input_file, output_file):
    # Initialize lists to store titles and categories
    titles = []
    categories = []

    # Read data from input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

        # Process each line to extract titles and categories
        for line in lines:
            # Split the line into title and category
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue
            title, category = parts[0].strip(), parts[1].strip()

            # Append title and category to respective lists
            titles.append(title)
            categories.append(category)

    # Write data to CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['titles', 'categories'])  # Write header
        writer.writerows(zip(titles, categories))  # Write data

if __name__ == "__main__":
    # Prompt the user to enter input data
    text_filename = input("Enter the text filename: ")
    csv_filename = input("Enter the output csv filename: ")
    text_to_csv(text_filename, csv_filename)


