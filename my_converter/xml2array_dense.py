import numpy as np

def convertPosition(x_l, x_r, y_l, y_r):
    return np.clip(x_l, 0, 96), np.clip(x_r, 0, 96), np.clip(y_l, 0, 96), np.clip(y_r, 0, 96)


material_code = {"wood": 0, "stone": 1, "ice": 2}
type_code = {"empty": 0,
             "Platform": 1,
             "TNT": 2,
             "Pig": 3,
             "RectFat": 4,
             "RectFat90": 5,
             "RectSmall": 6,
             "RectSmall90": 7,
             "RectMedium": 8,
             "RectMedium90": 9,
             "RectBig": 10,
             "RectBig90": 11,
             "RectTiny": 12,
             "RectTiny90": 13,
             "SquareSmall": 14,
             "SquareTiny": 15,
             "SquareHole": 16,
             "TriangleHole": 17,
             "TriangleHole90": 18,
             "Triangle": 19,
             "Triangle90": 20,
             "Circle": 21,
             "CircleSmall": 22
             }

size_code = {
    "Platform": [0.62, 0.62],
    "Pig": [0.5, 0.45],
    "TNT": [0.55, 0.55],
    "SquareHole": [0.84, 0.84],
    "RectFat": [0.85, 0.43],
    "RectFat90": [0.43, 0.85],
    "SquareSmall": [0.43, 0.43],
    "SquareTiny": [0.22, 0.22],
    "RectTiny": [0.43, 0.22],
    "RectTiny90": [0.22, 0.43],
    "RectSmall": [0.85, 0.22],
    "RectSmall90": [0.22, 0.85],
    "RectMedium": [1.68, 0.22],
    "RectMedium90": [0.22, 1.68],
    "RectBig": [2.06, 0.22],
    "RectBig90": [0.22, 2.06],
    "CircleSmall": [0.45, 0.45],
    "Circle": [0.8, 0.8],
    "Triangle": [0.82, 0.82],
    "Triangle90": [0.82, 0.82],
    "TriangleHole": [0.82, 0.82],
    "TriangleHole90": [0.82, 0.82]
}


def preprocess(text):
    return text[1:-3]


def label(content, type, material):
    if content == "empty" or content == "Platform" or content == "TNT" or content == "PigMedium":
        return type_code[type]
    else:
        return type_code[type] + material_code[material] * 19

def print_map(map_array):
    np.set_printoptions(threshold=np.inf)
    for r in reversed(range(len(map_array))):
        for c in reversed(range(len(map_array[0]))):
            print(np.argmax(map_array[r][c]), end="")
        print()

def load_xml(xml_txt):
    contents = []
    for text in xml_txt:
        block_name = ""
        x, y = 0, 0
        text = preprocess(text)
        content = text.split(" ")
        material = ""
        if len(content) >= 2:
            if content[0] == "Block":
                material = content[2][10:-1]
                if str(content[5][10:-1]) != "0" and str(content[5][10:-1]) != "0.0":
                    if content[5][10:-1] == "90.0" or content[5][10:-1] == "135.0":
                        block_name = content[1][6:-1] + content[5][10:-3]
                    else:
                        block_name = content[1][6:-1] + content[5][10:-1]
                else:
                    block_name = content[1][6:-1]
                x, y = float(content[3][3:-1]), float(content[4][3:-1])
            elif content[0] == "Platform" or content[0] == "Pig":
                block_name = content[0]
                x, y = float(content[3][3:-1]), float(content[4][3:-1])
            elif content[0] == "TNT":
                block_name = content[0]
                x, y = float(content[2][3:-1]), float(content[3][3:-1])
            else:
                continue
            x = round(x, 1)
            contents.append((block_name, x, y, material))
    return contents


def xml2array(xml_data):
    map_array = np.full((96, 96, 61), 0)
    for x in range(96):
        for y in range(96):
            map_array[y][x][0] = 1
    tmp_line = []
    for line in xml_data:
        if "Block" in line:
            tmp_line.append(line)
    for t_line in tmp_line:
        xml_data.remove(t_line)
        xml_data.append(t_line)
    contents = load_xml(xml_data)
    for content in contents:
        x, y = float(content[1]), float(content[2])
        x_l, x_r = x - size_code[content[0]][0] / 2, x + size_code[content[0]][0]/2
        y_l, y_r = y - size_code[content[0]][1] / 2, y + size_code[content[0]][1] / 2
        if y < -3.5:
            # print(text)
            continue
        x_l *= 10
        x_r *= 10
        y_l *= 10
        y_r *= 10
        x_l += 30
        x_r += 30
        y_l += 36
        y_r += 36
        x_l, x_r, y_l, y_r = int(x_l), int(x_r), int(y_l), int(y_r)
        x_l, x_r, y_l, y_r = convertPosition(x_l, x_r, y_l, y_r)
        if content[3] == "":
            ob = type_code[content[0]]
        else:
            ob = type_code[content[0]] + material_code[content[3]]*19
        for x in range(x_l, x_r):
            for y in range(y_l, y_r):
                try:
                    map_array[y][x][0] = 0
                    map_array[y][x][ob] = 1
                except:
                    print("a", x, y)
    #print_map(map_array)
    return map_array
