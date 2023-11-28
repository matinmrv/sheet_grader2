import cv2
import numpy as np
import random

def show_images(titles, images, wait=True):

    for title, image in zip(titles, images):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def find_image_contours(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edge_img = cv2.Canny(blur_img, 10, 70)
    contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours

def get_rect_cnts(contours):

    rect_cnts = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            rect_cnts.append(approx)
    rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)

    return rect_cnts

def get_question_mask(document, doc_copy, biggest_cnt):

    x, y = biggest_cnt[0][0][0] + 4, biggest_cnt[0][0][1] + 4
    x_W, y_H = biggest_cnt[2][0][0] + 4, biggest_cnt[2][0][1] + 4 
    mask = np.zeros((document.shape[0], document.shape[1]), np.uint8)
    cv2.rectangle(mask, (x, y), (x_W, y_H), (255, 255, 255), -1) 
    masked = cv2.bitwise_and(doc_copy, doc_copy, mask=mask)
    masked = masked[y:y_H, x:x_W]

    return masked

def thresholding(masked):

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

    return thresh

def split_image(image, questions, answers):

    r = len(image) // questions * questions
    c = len(image[0]) // answers * answers
    image = image[:r, :c]
    
    rows = np.vsplit(image, questions)
    boxes = []

    for row in rows:
        cols = np.hsplit(row, answers)
        for box in cols:
            boxes.append(box)

    return boxes

def evaluation(boxes, ans_dict, questions, answers):
    score = 0
    correct_ones = {}

    pixel_values = {} 

    for i in range(0, questions):
        user_answer = None
        no_answer = True  

        for j in range(answers):
            pixels = cv2.countNonZero(boxes[j + i * 5])

            if pixels > 100:
                no_answer = False

            if user_answer is None or pixels > user_answer[1]:
                user_answer = (j, pixels)

        if no_answer:
            correct_ones[i] = None  
        else:
            correct_ones[i] = user_answer[0]

        if ans_dict[i] == user_answer[0]:
            score += 1

    return score, correct_ones



def create_dict():

    ans_dict = {}
    for i in range(26, 50):
        ans_dict[i] = random.randint(0, 4)

    return ans_dict


def coordin(ques_coords, gray_x_template):
        
    highest_y_center = max(ques_coords, key=lambda x: x[1])
    lowest_y_center = min(ques_coords, key=lambda x: x[1])

    xl, xr = highest_y_center[0] + 62, highest_y_center[0] + 240
    yu, yd = lowest_y_center[1] - 10, highest_y_center[1] + 13

    len_ques = len(ques_coords)

    return yu, yd, xl, xr, len_ques

def forward(image, yu, yd, xl, xr, ans_dict, questions, answers=5):

    canvas = np.zeros_like(image)
    canvas[yu:yd, xl:xr] = image[yu:yd, xl:xr]
    canvas_copy = canvas.copy()

    image_contours = find_image_contours(canvas)
    cv2.drawContours(canvas, image_contours, -1, (0, 255, 0), 3)
    rect_cnts = get_rect_cnts(image_contours)
    biggest_cnt = rect_cnts[0]

    masked = get_question_mask(canvas, canvas_copy, biggest_cnt)
    thresh = thresholding(masked)
    boxes = split_image(thresh, questions, answers)
    score, correct_ones = evaluation(boxes, ans_dict, questions, answers)
    
    return score, correct_ones

def answer_dict(correct_p1, correct_p2):

    len_ques = len(correct_p1) + len(correct_p2)
    ans_dict = {}

    for i in range(0, len_ques):
        if i < 25:
            ans_dict[i] = correct_p1[i]
        else:
            ans_dict[i] = correct_p2[i - 25]

    return ans_dict
    

def one_main(image_path, rec_mark_path, ques_mark_path):

    ans_dict_p1 = {0: 0, 1: 2, 2: 2, 3: 1, 4: 2, 5: 4, 6: 0, 7: 1, 8: 0, 9: 2, 10: 2, 11: 1, 12: 2, 13: 4, 14: 0, 15: 1, 16: 4, 17: 2, 18: 2, 19: 3, 20: 2, 21: 0, 22: 4, 23: 3, 24: 1}
    ans_dict_p2 = {0: 4, 1: 2, 2: 2, 3: 3, 4: 2, 5: 0, 6: 4, 7: 3, 8: 0, 9: 2, 10: 2, 11: 1, 12: 2, 13: 4, 14: 0, 15: 1, 16: 4, 17: 2, 18: 2, 19: 3, 20: 2, 21: 0, 22: 4, 23: 3, 24: 3}

    image = cv2.imread(image_path)
    image = cv2.resize(image, (1000, 800))
    rec_template = cv2.imread(rec_mark_path)
    ques_template = cv2.imread(ques_mark_path)

    gray_rec_template = cv2.cvtColor(rec_template, cv2.COLOR_BGR2GRAY)
    gray_ques_template = cv2.cvtColor(ques_template, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rec_result = cv2.matchTemplate(gray_image, gray_rec_template, cv2.TM_CCOEFF_NORMED)
    ques_result = cv2.matchTemplate(gray_image, gray_ques_template, cv2.TM_CCOEFF_NORMED)

    top_n = 4 
    indices = np.argsort(rec_result.flatten())[::-1][:top_n]
    centers = []

    for index in indices:
        pt = np.unravel_index(index, rec_result.shape)
        center = (pt[1] + gray_rec_template.shape[1] // 2, pt[0] + gray_rec_template.shape[0] // 2)
        centers.append(center)
        cv2.rectangle(image, (pt[1], pt[0]), (pt[1] + gray_rec_template.shape[1], pt[0] + gray_rec_template.shape[0]), (0, 255, 0), 2)

    xl, xr,  = centers[0][0], centers[2][0]
    yu, yd = centers[1][1], centers[0][1]

    top_n_ques = 200
    indices_ques = np.argsort(ques_result.flatten())[::-1][:top_n_ques]
    ques_coords_part1 = []
    ques_coords_part2 = []

    min_distance = 5
    detected_centers_part1 = []
    detected_centers_part2 = []

    for index in indices_ques:
        pt = np.unravel_index(index, ques_result.shape)
        coord = (pt[1], pt[0])

        # part2
        if coord[0] > xl and coord[1] > yu and coord[1] < yd:
            if all(np.linalg.norm(np.array(coord) - np.array(center)) > min_distance for center in detected_centers_part2):
                ques_coords_part2.append(coord)
                detected_centers_part2.append(coord)

        # part1
        if coord[0] < xl and coord[1] > yu and coord[1] < yd:
            if all(np.linalg.norm(np.array(coord) - np.array(center)) > min_distance for center in detected_centers_part1):
                ques_coords_part1.append(coord)
                detected_centers_part1.append(coord)


    
    yu1, yd1, xl1, xr1, len_ques_p1 = coordin(ques_coords_part1, gray_rec_template)
    yu2, yd2, xl2, xr2, len_ques_p2 = coordin(ques_coords_part2, gray_ques_template)

    score_p1, correct_p1 = forward(image, yu1, yd1, xl1, xr1, ans_dict_p1, len_ques_p1, answers=5)
    score_p2, correct_p2 = forward(image, yu2, yd2, xl2, xr2, ans_dict_p2, len_ques_p2, answers=5)
    
    full_score = score_p1 + score_p2

    ans_dict = answer_dict(correct_p1, correct_p2)


 
    print("Final Answers Dictionary:")
    print(ans_dict)
    print("Score Part 1:", score_p1)
    print("Score Part 2:", score_p2)
    print("Full Score:", full_score, "/ 50")

if __name__ == "__main__":
    one_main('/home/matin/Desktop/answer-sheet-marker.jpg', '/home/matin/Desktop/data/temp.png', '/home/matin/Desktop/data/temp3.png')


