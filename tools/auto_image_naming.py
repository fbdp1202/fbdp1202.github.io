import os

f = open("input.txt", "r")
lines = f.readlines()

original_name = []
revised_name = []
for line in lines:
    l = line.split(" ")
    original_name.append(str(l[0]))
    revised_name.append(str(l[1]).replace('\n', ''))

root_path = "../"
post_path = root_path + "_posts/"
post_category = "PaperToMath/"

folder_path = post_path + post_category

img_path = "../assets/img/"
img_category = "dev/papertomath/"

title = "summary"
chapter_num = ""

img_title = title + "/"
# img_subtitle = "lecture" + chapter_num + "/" +
img_subtitle = "SA_EEND/"

move_folder = img_path + img_category + img_title + img_subtitle


count = 1
for i in range(len(original_name)):
    move_name = move_folder + title + "-" + chapter_num + "-{:03d}".format(count) + "-" + revised_name[i] + ".png"
    print(move_name)
    for filename in os.listdir(folder_path):
        if original_name[i] == filename:
            os.rename(folder_path + filename, move_name)
            count += 1

print(len(original_name), count - 1)

# def ChangeName(path, cName):
#     i = 1
