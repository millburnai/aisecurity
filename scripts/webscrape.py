# IMPORTANT: IMAGES ARE DOWNLOADED TO THE SCRIPTS FOLDER THAT THIS PROGRAM IS IN


import sys
import os
import time

# import requests
import shutil

# from requests.auth import HTTPBasicAuth
import os  # file editing
from PIL import Image  # image editing
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys  # import ability to type
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://halbrown.photoreflect.com/store/Photos.aspx?e=10785352"
PASSWORD_INPUT_NAME = "ctl01$CPCenter$PhotoSetAccess$EventChallengeAnswer"
SUBMIT_BUTTON_CLASS_NAME = "btn psa-access-event psaEventPassBtn"
ANSWER_REQUIRED_CLASS_NAME = "error psaAnswerRequired"
INCORRECT_PASSWORD_CLASS_NAME = "error psaPassError"
LIST_ITEMS_CLASS_NAME = "ps-photo-container all-rounded ps-photo-spaced ps-photo-box-shadow ps-photo-transparent-border photo-set-viewer-list-item"
STUDENT_NAME_CLASS_NAME = "ps-caption-event-name"
OUTER_PICTURE_CLASS_NAME = "ps-photo lazy"
BIG_PICTURE_CLASS_NAME = "ImageProtectionOverlay"
PASSWORD = "millers2022"  # password that's entered

try:
    dump = sys.argv[1]
except IndexError:
    print("usage: python3 webscrape.py DUMP_DIR")
    sys.exit(1)

chrome_options = webdriver.ChromeOptions()
# Prevents user's search history becoming sus
chrome_options.add_argument("--incognito")
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window()

driver.implicitly_wait(3)
ignored_exceptions = (
    NoSuchElementException,
    StaleElementReferenceException,
)
wait = WebDriverWait(
    driver, 10, ignored_exceptions=ignored_exceptions
)  # wait for max 10 seconds while trying to find elements

driver.get(URL)

# enter password
password_input = driver.find_element_by_xpath(
    "//input[@name='" + PASSWORD_INPUT_NAME + "']"
)
password_input.clear()
password_input.send_keys(PASSWORD)
submit_button = driver.find_element_by_xpath(
    "//input[@class='" + SUBMIT_BUTTON_CLASS_NAME + "']"
)  # invalid xpath
submit_button.click()

# time.sleep(1)
# try:
#     answer_required = driver.find_element_by_xpath("//span[@class='" + ANSWER_REQUIRED_CLASS_NAME + "']")
#     incorrect_password = driver.find_element_by_xpath("//span[@class='" + INCORRECT_PASSWORD_CLASS_NAME + "']")
#
#     if answer_required.get_attribute("style") != "display: none":
#         sys.exit("Couldn't enter the password in")
#     elif incorrect_password.get_attribute("style") != "display: none":
#         sys.exit("Incorrect password: " + PASSWORD)
# except NoSuchElementException:
#     pass


# Scrape pictures
a = ActionChains(driver)
students = driver.find_elements_by_xpath("//li[@class='" + LIST_ITEMS_CLASS_NAME + "']")
num_each_pic = []
names = []
for i in range(0, len(students)):
    s = students[i]
    # Get the only span from all of <item>'s descendants, then get its text
    names = driver.find_elements_by_xpath(
        "//span[@class='" + STUDENT_NAME_CLASS_NAME + "']"
    )
    name = names[i].text
    name = name.replace(", ", "_")  # make the name last_first
    try:
        name = name.split("_")[1] + "_" + name.split("_")[0]  # actaully First_Last
    except IndexError:
        print(name, i, " - SKIPPING")
        continue

    # make image dir
    try:
        os.makedirs(f"{dump}/{name}")
    except FileExistsError:
        continue

    s.click()
    time.sleep(0.6)

    list_items = driver.find_elements_by_xpath(
        "//div[@class='" + OUTER_PICTURE_CLASS_NAME + "']"
        )
    num_each_pic.append(len(list_items))
    for j in range(0, len(list_items)):

        p = list_items[j]
        p.click()
        time.sleep(0.6)
        driver.execute_script("window.scrollTo(0, 350)")

        cur_pic = driver.find_element_by_xpath(
            "//div[@class='" + BIG_PICTURE_CLASS_NAME + "']"
        )

        # take screenshot
        location = cur_pic.location
        size = cur_pic.size
        file_name = os.path.join(f"{dump}/{name}", name + "-" + str(j) + ".png")

        driver.save_screenshot(file_name)
        # change the rest to this
        # crop image; These sizes are good enough
        im = Image.open(file_name)
        im = im.crop((600, 25, 1605, 1530))
        im.save(file_name)
        print("Downloaded " + file_name)

        driver.back()

        list_items = driver.find_elements_by_xpath(
            "//div[@class='" + OUTER_PICTURE_CLASS_NAME + "']"
            )

    print("Downloaded {} images of {}".format(len(list_items), name))

    driver.back()  # go back to main page
    students = driver.find_elements_by_xpath(
        "//li[@class='" + LIST_ITEMS_CLASS_NAME + "']"
    )
    # list_items = driver.find_elements_by_xpath("//div[@class='"+ OUTER_PICTURE_CLASS_NAME + "']") #need to re-find the links to each student's page of pictures


print("Done downloading images")
print("Finished!")
driver.close()
