from bs4 import BeautifulSoup

with open("bncHdr.xml", "r", encoding="utf-8") as file:
    content = file.read()

soup = BeautifulSoup(content, "xml")

# 1. Extract the Title of the corpus/text
title = soup.find("title").text if soup.find("title") else "No title found"
print(f"Title: {title}")

# 2. Extract Author information
author = soup.find("author").text if soup.find("author") else "Unknown Author"
print(f"Author: {author}")

# 3. Extract the total number of words (Extent)
extent = soup.find("extent").text if soup.find("extent") else "N/A"
print(f"File Size/Words: {extent}")