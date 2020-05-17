text_file = open("tags.txt", "r")

lines = text_file.readlines()

namesTags = []

for i in range(0, len(lines), 1):
    strTagsFull = lines[i].split(",")[0]
    namesTags.append(strTagsFull[4:]) 

f = open("result.txt", "w")

for i in range(0, len(namesTags)):
    Tag = namesTags[i]
    f.write('<outtag rowset="CPAS" name="'+ Tag + '" />\n')

