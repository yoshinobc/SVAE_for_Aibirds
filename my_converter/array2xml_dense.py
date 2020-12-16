import numpy as np
import queue

#豚やTNTは取り出したら隣接するブロックは消去する
"""
rectFat

aaaaaaaa
aaaaaaaa
aaaaaaaa
aaaaaaaa

bbbb
bbbb
bbbb
bbbb
bbbb
bbbb
bbbb
bbbb


rectSmall

cccccccc
cccccccc

dd
dd
dd
dd
dd
dd
dd
dd

rectMedium

eeeeeeeeeeeeeeee
eeeeeeeeeeeeeeee

ff
ff
ff
ff
ff
ff
ff
ff
ff
ff
ff
ff
ff
ff
ff
ff

rectBig
gggggggggggggggggggg
gggggggggggggggggggg

hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh
hh

rectTiny
iiii
iiii

jj
jj
jj
jj

SquareSmall
kkkk
kkkk
kkkk
kkkk

SquareTiny
ll
ll

SquareHole
mmmmmmmm
mmmmmmmm
mmmmmmmm
mmmmmmmm
mmmmmmmm
mmmmmmmm
mmmmmmmm
mmmmmmmm

TriangleHole
nnnnnnnn
nnnnnnnn
nnnnnnnn
nnnnnnnn
nnnnnnnn
nnnnnnnn
nnnnnnnn
nnnnnnnn

TriangleHole90
oooooooo
oooooooo
oooooooo
oooooooo
oooooooo
oooooooo
oooooooo
oooooooo

Triangle
qqqqqqqq
qqqqqqqq
qqqqqqqq
qqqqqqqq
qqqqqqqq
qqqqqqqq
qqqqqqqq
qqqqqqqq

Triangle90
rrrrrrrr
rrrrrrrr
rrrrrrrr
rrrrrrrr
rrrrrrrr
rrrrrrrr
rrrrrrrr
rrrrrrrr

Circle
ssssssss
ssssssss
ssssssss
ssssssss
ssssssss
ssssssss
ssssssss
ssssssss

CircleSmall
tttt
tttt
tttt
tttt

"""
hosei_x = 30
hosei_y = 36
delta = 0.1


def search_match_object(map_array, x, y, x_lists, y_lists, index):
    count = 0
    for x_ in x_lists:
        for y_ in y_lists:
            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                continue
            if map_array[x + x_][y + y_] == index:
                count += 1
    if count >= len(x_lists) * len(y_lists) * delta:
        return (x + x_lists[-1] / 2, y + y_lists[-1] / 2)
    else:
        return None


def calc_object(count_objects):
    x_list, y_list = [], []
    for count_object in count_objects:
        x_list.append(count_object[0])
        y_list.append(count_object[1])
    return (sum(x_list) / len(x_list), sum(y_list) / len(y_list))


def dfs_pig(map_array, x, y):
    count_objects = []
    walk_lists = [0, 1, -1]
    q = queue.Queue()
    q.put((x, y))
    count = 0
    while not q.empty():
        point = q.get()
        x, y = point[0], point[1]
        for walk_x in walk_lists:
            for walk_y in walk_lists:
                if (x + walk_x >= len(map_array) or x + walk_x < 0) or (y + walk_y >= len(map_array) or y + walk_y < 0):
                    continue
                if map_array[x + walk_x][y + walk_y] == 3:
                    count_objects.append((x + walk_x, y + walk_y))
                    count += 1
                    map_array[x + walk_x][y + walk_y] = 0
                    q.put((x + walk_x, y + walk_y))
        select_object = calc_object(count_objects)
        #print("count_objects", count_objects)
        for count_object in count_objects:
            map_array[count_object[0]][count_object[1]] = 0
        return select_object, map_array


def conv_platform_core(map_array, x_lists, y_lists):
    platforms = []
    text = ""
    for x in range(len(map_array)):
        for y in range(len(map_array[0])):
            count_platforms = None
            index = 1
            if map_array[x][y] == index:
                count_platforms = search_match_object(
                    map_array, x, y, x_lists, y_lists, index)
            if count_platforms == None:
                continue
            # print("before", count_platforms)
            # render_map(map_array)
            platforms.append(
                ((count_platforms[1] - hosei_x) / 10, (count_platforms[0] - hosei_y) / 10))
            for x_ in x_lists:
                for y_ in y_lists:
                    if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                        continue
                    map_array[x + x_][y + y_] = 0
            # print("after")
            # render_map(map_array)
    for platform in platforms:
        text += '<Platform type="' + "Platform" + '" material="' + "" + '" x="' + \
            str(platform[0]) + '" y="' + str(platform[1]) + \
            '" rotation="' + str(0) + '" />\n'
    return text, map_array


def conv_platform(map_array):
    y_lists = [0, 1, 2, 3, 4, 5]
    texts = ""
    for i in reversed([1, 2, 3, 4, 5]):
        y_lists = [k for k in range(i)]
        for j in reversed([1, 2, 3, 4, 5]):
            x_lists = [k for k in range(j)]
            text, map_array = conv_platform_core(map_array, x_lists, y_lists)
            texts += text
    return texts


def conv_pig(map_array):
    x_lists = [0, 1, 2, 3, 4]
    y_lists = [0, 1, 2, 3, 4]
    pigs = []
    text = ""
    for x in range(len(map_array)):
        for y in range(len(map_array[0])):
            count_pigs = None
            index = 3
            if map_array[x][y] == index:
                count_pigs = search_match_object(map_array, x, y, x_lists, y_lists,index)
                #count_pigs, map_array = dfs_pig(map_array, x, y)
                if count_pigs == None:
                    # print(count_rects)
                    continue
                print(count_pigs)
                pigs.append(
                    ((count_pigs[1] - hosei_x) / 10, (count_pigs[0] - hosei_y) / 10))

                for x_ in x_lists:
                    for y_ in y_lists:
                        if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                            continue
                        map_array[x + x_][y + y_] = 0


    for pig in pigs:
        text += '<Pig type="' + "BasicSmall" + '" material="' + "" + '" x="' + \
            str(pig[0]) + '" y="' + str(pig[1]) + \
            '" rotation="' + str(0) + '" />\n'
    return text


def conv_tnt(map_array):
    x_lists = [0, 1, 2, 3, 4]
    y_lists = [0, 1, 2, 3, 4]
    tnts = []
    text = ""
    for x in range(len(map_array)):
        for y in range(len(map_array[0])):
            count_tnts = None
            index = 2
            if map_array[x][y] == index:
                count_tnts = search_match_object(
                    map_array, x, y, x_lists, y_lists, index)
                if count_tnts == None:
                    # print(count_rects)
                    continue
                tnts.append(
                    ((count_tnts[1] - hosei_x) / 10, (count_tnts[0] - hosei_y) / 10))
                for x_ in x_lists:
                    for y_ in y_lists:
                        if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                            continue
                        map_array[x + x_][y + y_] = 0
    for tnt in tnts:
        #text += '<TNT type="' + "BasicSmall" + '" material="' + "" + '" x="' + \
        #    str(tnt[0]) + '" y="' + str(tnt[1]) + \
        #    '" rotation="' + str(0) + '" />\n'
        text += '<TNT type="' + "" + '" x="' + \
            str(tnt[0]) + '" y="' + str(tnt[1]) + \
            '" rotation="' + str(0) + '" />\n'
    return text


def check_rect(map_array, x, y, x_lists, y_lists, code):
    material = map_array[x][y]
    if material not in code:
        return None, None
    if material == 0 or material == 1 or material == 2 or material == 3:
        return None, None
    count = 0
    for x_ in x_lists:
        for y_ in y_lists:
            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                continue
            if map_array[x+x_][y+y_] == material:
                count += 1
    #print(code,material,count,len(x_lists)*len(y_lists))
    if count >= len(x_lists) * len(y_lists) * delta:
        if (material - 4) % 3 == 0:
            material = 0
        elif (material - 4) % 3 == 1:
            material = 1
        else:
            material = 2
        return (x + x_lists[-1] / 2, y + y_lists[-1] / 2), material
    else:
        return None, None


def conv_rect(map_array, x_lists, y_lists, code):
    rects = []
    for x in range(len(map_array)):
        for y in range(len(map_array[0])):
            count_rects, material = check_rect(
                map_array, x, y, x_lists, y_lists, code)
            if count_rects != None:
                # print(count_rects)
                rects.append(
                    ((count_rects[1] - hosei_x) / 10, (count_rects[0] - hosei_y) / 10, material))
                for x_ in x_lists:
                    for y_ in y_lists:
                        if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                            continue
                        map_array[x + x_][y + y_] = 0
    return map_array, rects


def conv_block(map_array):
    text = ""
    material_key = {0: "wood", 1: "stone", 2: "ice"}
    map_array, square_hole = conv_rect(
        map_array, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [16, 16+19, 16+19*2])
    for rect in square_hole:
        text += '<Block type="' + "SquareHole" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'

    map_array, rect_big_90 = conv_rect(map_array, [
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [0, 1], [11, 11+19, 11+19*2])
    for rect in rect_big_90:
        text += '<Block type="' + "RectBig" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(90) + '" />\n'

    map_array, rect_big = conv_rect(map_array,  [0, 1], [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [10, 10+19, 10+19*2])
    for rect in rect_big:
        text += '<Block type="' + "RectBig" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'

    map_array, rect_fat_90 = conv_rect(
        map_array, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3], [5, 5+19, 5+19*2])
    for rect in rect_fat_90:
        text += '<Block type="' + "RectFat" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(90) + '" />\n'

    map_array, rect_fat = conv_rect(
        map_array, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [4, 4+19, 4+19*2])
    for rect in rect_fat:
        text += '<Block type="' + "RectFat" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'

    map_array, rect_medium_90 = conv_rect(map_array, [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [0, 1], [9, 9+19, 9+19*2])
    for rect in rect_medium_90:
        text += '<Block type="' + "RectMedium" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(90) + '" />\n'

    map_array, rect_medium = conv_rect(map_array, [0, 1], [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [8, 8+19, 8+19*2])
    for rect in rect_medium:
        text += '<Block type="' + "RectMedium" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'

    map_array, rect_small_90 = conv_rect(
        map_array, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1], [7, 7+19, 7+19*2])
    for rect in rect_small_90:
        text += '<Block type="' + "RectSmall" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(90) + '" />\n'

    map_array, rect_small = conv_rect(
        map_array, [0, 1], [0, 1, 2, 3, 4, 5, 6, 7], [6, 6+19, 6+19*2])  # small
    for rect in rect_small:
        text += '<Block type="' + "RectSmall" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'

    map_array, square_small = conv_rect(
        map_array, [0, 1, 2, 3], [0, 1, 2, 3], [14, 14+19, 14+19*2])
    for rect in square_small:
        text += '<Block type="' + "SquareSmall" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'

    map_array, rect_tiny_90 = conv_rect(
        map_array, [0, 1, 2, 3], [0, 1], [13, 13+19, 13+19*2])
    for rect in rect_tiny_90:
        text += '<Block type="' + "RectTiny" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(90) + '" />\n'

    map_array, rect_tiny = conv_rect(
        map_array, [0, 1], [0, 1, 2, 3], [12, 12+19, 12+19*2])
    for rect in rect_tiny:
        text += '<Block type="' + "RectTiny" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'

    map_array, square_tiny = conv_rect(
        map_array, [0, 1], [0, 1], [15, 15+19, 15+19*2])
    for rect in square_tiny:
        text += '<Block type="' + "SquareTiny" + '" material="' + material_key[int(rect[2])] + '" x="' + \
            str(rect[0]) + '" y="' + str(rect[1]) + \
            '" rotation="' + str(0) + '" />\n'
    return text


def conv_trianglehole(map_array):
    x_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    y_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    materials = {0: "wood", 1: "stone", 2: "ice"}
    text = ""
    for material in materials.keys():
        triangleholes = []
        for x in range(len(map_array)):
            for y in range(len(map_array[0])):
                count_triangleholes = None
                index = 17 + material * 19
                if map_array[x][y] == index:
                    count_triangleholes = search_match_object(
                        map_array, x, y, x_lists, y_lists, index)
                    if count_triangleholes == None:
                        # print(count_rects)
                        continue
                    triangleholes.append(
                        ((count_triangleholes[1] - hosei_x) / 10, (count_triangleholes[0] - hosei_y) / 10))
                    for x_ in x_lists:
                        for y_ in y_lists:
                            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                                continue
                            map_array[x + x_][y + y_] = 0
        for trianglehole in triangleholes:
            text += '<Block type="' + "TriangleHole" + '" material="' + materials[material] + '" x="' + \
                str(trianglehole[0]) + '" y="' + str(trianglehole[1]) + \
                '" rotation="' + str(0) + '" />\n'
    return text


def conv_trianglehole90(map_array):
    x_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    y_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    materials = {0: "wood", 1: "stone", 2: "ice"}
    text = ""
    for material in materials.keys():
        trianglehole90s = []
        for x in range(len(map_array)):
            for y in range(len(map_array[0])):
                count_trianglehole90s = None
                index = 18 + material * 19
                if map_array[x][y] == index:
                    count_trianglehole90s = search_match_object(
                        map_array, x, y, x_lists, y_lists, index)
                    if count_trianglehole90s == None:
                        # print(count_rects)
                        continue
                    trianglehole90s.append(
                        ((count_trianglehole90s[1] - hosei_x) / 10, (count_trianglehole90s[0] - hosei_y) / 10))
                    for x_ in x_lists:
                        for y_ in y_lists:
                            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                                continue
                            map_array[x + x_][y + y_] = 0
        for trianglehole90 in trianglehole90s:
            text += '<Block type="' + "TriangleHole" + '" material="' + materials[material] + '" x="' + \
                str(trianglehole90[0]) + '" y="' + str(trianglehole90[1]) + \
                '" rotation="' + str(90) + '" />\n'
    return text


def conv_triangle(map_array):
    x_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    y_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    materials = {0: "wood", 1: "stone", 2: "ice"}
    text = ""
    for material in materials.keys():
        triangles = []
        for x in range(len(map_array)):
            for y in range(len(map_array[0])):
                count_triangles = None
                index = 19 + material * 19
                if map_array[x][y] == index:
                    count_triangles = search_match_object(
                        map_array, x, y, x_lists, y_lists, index)
                    if count_triangles == None:
                        # print(count_rects)
                        continue
                    triangles.append(
                        ((count_triangles[1] - hosei_x) / 10, (count_triangles[0] - hosei_y) / 10))
                    for x_ in x_lists:
                        for y_ in y_lists:
                            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                                continue
                            map_array[x + x_][y + y_] = 0
        for triangle in triangles:
            text += '<Block type="' + "Triangle" + '" material="' + materials[material] + '" x="' + \
                str(triangle[0]) + '" y="' + str(triangle[1]) + \
                '" rotation="' + str(0) + '" />\n'
    return text


def conv_triangle90(map_array):
    x_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    y_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    materials = {0: "wood", 1: "stone", 2: "ice"}
    text = ""
    for material in materials.keys():
        triangle90s = []
        for x in range(len(map_array)):
            for y in range(len(map_array[0])):
                count_triangle90s = None
                index = 20 + material * 19
                if map_array[x][y] == index:
                    count_triangle90s = search_match_object(
                        map_array, x, y, x_lists, y_lists, index)
                    if count_triangle90s == None:
                        # print(count_rects)
                        continue
                    triangle90s.append(
                        ((count_triangle90s[1] - hosei_x) / 10, (count_triangle90s[0] - hosei_y) / 10))
                    for x_ in x_lists:
                        for y_ in y_lists:
                            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                                continue
                            map_array[x + x_][y + y_] = 0
        for triangle90 in triangle90s:
            text += '<Block type="' + "Triangle" + '" material="' + materials[material] + '" x="' + \
                str(triangle90[0]) + '" y="' + str(triangle90[1]) + \
                '" rotation="' + str(90) + '" />\n'
    return text


def conv_circle(map_array):
    x_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    y_lists = [0, 1, 2, 3, 4, 5, 6, 7]
    materials = {0: "wood", 1: "stone", 2: "ice"}
    text = ""
    for material in materials.keys():
        circles = []
        for x in range(len(map_array)):
            for y in range(len(map_array[0])):
                count_circles = None
                index = 21 + material * 19
                if map_array[x][y] == index:
                    count_circles = search_match_object(
                        map_array, x, y, x_lists, y_lists, index)
                    if count_circles == None:
                        # print(count_rects)
                        continue
                    circles.append(
                        ((count_circles[1] - hosei_x) / 10, (count_circles[0] - hosei_y) / 10))
                    for x_ in x_lists:
                        for y_ in y_lists:
                            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                                continue
                            map_array[x + x_][y + y_] = 0
        for circle in circles:
            text += '<Block type="' + "Circle" + '" material="' + materials[material] + '" x="' + \
                str(circle[0]) + '" y="' + str(circle[1]) + \
                '" rotation="' + str(0) + '" />\n'
    return text


def conv_circlesmall(map_array):
    x_lists = [0, 1, 2, 3]
    y_lists = [0, 1, 2, 3]
    materials = {0: "wood", 1: "stone", 2: "ice"}
    text = ""
    for material in materials.keys():
        circlesmalls = []
        for x in range(len(map_array)):
            for y in range(len(map_array[0])):
                count_circlesmalls = None
                index = 22 + material * 19
                if map_array[x][y] == index:
                    count_circlesmalls = search_match_object(
                        map_array, x, y, x_lists, y_lists, index)
                    if count_circlesmalls == None:
                        continue
                    circlesmalls.append(
                        ((count_circlesmalls[1] - hosei_x) / 10, (count_circlesmalls[0] - hosei_y) / 10))
                    for x_ in x_lists:
                        for y_ in y_lists:
                            if (x+x_ >= len(map_array) or x+x_ < 0) or (y+y_ >= len(map_array) or y+y_ < 0):
                                continue
                            map_array[x + x_][y + y_] = 0
        for circlesmall in circlesmalls:
            text += '<Block type="' + "CircleSmall" + '" material="' + materials[material] + '" x="' + \
                str(circlesmall[0]) + '" y="' + str(circlesmall[1]) + \
                '" rotation="' + str(0) + '" />\n'
    return text


def render_map(map_array):
    text = ""
    for x in range(len(map_array)-1, 0, -1):
        for y in range(len(map_array[0])):
            num = map_array[x][y]
            if num == 0:
                num = "."
            elif num == 1:
                num = "#"
            elif num == 2:
                num = "T"
            elif num == 3:
                num = "P"
            elif num == 4:
                num = "W"
            elif num == 5:
                num = "S"
            elif num == 6:
                num = "I"
            elif num <= 9:
                num = "a"
            elif num <= 12:
                num = "b"
            elif num <= 15:
                num = "c"
            elif num <= 18:
                num = "d"
            elif num <= 21:
                num = "e"
            elif num <= 24:
                num = "f"
            text += num
        text += "\n"
    print(text)


def array2xml(map_array_sub):
    map_array = np.zeros((96, 96))
    for x in range(len(map_array_sub)):
        for y in range(len(map_array_sub[0])):
            num = np.argmax(map_array_sub[x][y])
            map_array[x][y] = num
    text = '<?xml version="1.0" encoding="utf-16"?>\n'
    text += '<Level width ="2">\n'
    text += '<Camera x="0" y="2" minWidth="20" maxWidth="30">\n'
    text += '<Birds>\n'
    text += '<Bird type="BirdBlue"/>\n'
    text += '<Bird type="BirdRed"/>\n'
    text += '<Bird type="BirdYellow"/>\n'
    text += '<Bird type="BirdBlue"/>\n'
    text += '<Bird type="BirdBlue"/>\n'
    text += '<Bird type="BirdBlue"/>\n'
    text += '<Bird type="BirdRed"/>\n'
    text += '</Birds>\n'
    text += '<Slingshot x="-8" y="-2.5">\n'
    text += '<GameObjects>\n'
    text += conv_platform(map_array)
    text += conv_tnt(map_array)
    text += conv_pig(map_array)
    text += conv_block(map_array)
    text += conv_trianglehole(map_array)
    text += conv_trianglehole90(map_array)
    text += conv_triangle(map_array)
    text += conv_triangle90(map_array)
    text += conv_circle(map_array)
    text += conv_circlesmall(map_array)

    text += "</GameObjects>\n"
    text += "</Level>\n"

    return text
