import sys
import os
import time
import requests
import shutil
from selenium import webdriver
from selenium.webdriver.common.keys import Keys #import ability to type
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://halbrown.photoreflect.com/store/Photos.aspx?e=10785352"
PASSWORD_INPUT_NAME = "ctl01$CPCenter$PhotoSetAccess$EventChallengeAnswer"
SUBMIT_BUTTON_CLASS_NAME = "btn psa-access-event psaEventPassBtn"
ANSWER_REQUIRED_CLASS_NAME = "error psaAnswerRequired"
INCORRECT_PASSWORD_CLASS_NAME = "error psaPassError"
LIST_ITEMS_CLASS_NAME = "ps-photo-container all-rounded ps-photo-spaced ps-photo-box-shadow ps-photo-transparent-border photo-set-viewer-list-item"
STUDENT_NAME_CLASS_NAME = "ps-caption-event-name"
PICTURE_CLASS_NAME = "ps-photo lazy"
PASSWORD = "millers2022" #password that's entered

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito") #makes Chrome open incognito so your history isn't flooded :P
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window()

driver.implicitly_wait(3)
ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
wait = WebDriverWait(driver, 10, ignored_exceptions=ignored_exceptions) #wait for max 10 seconds while trying to find elements

driver.get(URL)

#enter password
password_input = driver.find_element_by_xpath("//input[@name='" + PASSWORD_INPUT_NAME + "']")
password_input.clear()
password_input.send_keys(PASSWORD)
submit_button = driver.find_element_by_xpath("//input[@class='" + SUBMIT_BUTTON_CLASS_NAME + "']") #invalid xpath
submit_button.click()

try:
    answer_required = driver.find_element_by_xpath("//span[@class='" + ANSWER_REQUIRED_CLASS_NAME + "']")
    incorrect_password = driver.find_element_by_xpath("//span[@class='" + INCORRECT_PASSWORD_CLASS_NAME + "']")

    if answer_required.get_attribute("style") != "display: none":
        sys.exit("Couldn't enter the password in")
    elif incorrect_password.get_attribute("style") != "display: none":
        sys.exit("Incorrect password: " + PASSWORD)
except NoSuchElementException:
    pass


#scrape pictures

students = driver.find_elements_by_xpath("//li[@class='"+ LIST_ITEMS_CLASS_NAME + "']")
for i in range(0, len(students)):
    s = students[i]
    names = driver.find_elements_by_xpath("//span[@class='" + STUDENT_NAME_CLASS_NAME + "']") #get the only span from all of <item>'s descendants, then get its text
    name = names[i].text
    name = name.replace(", ", "_") #make the name last_first
    s.click()
    time.sleep(2)

    list_items = driver.find_elements_by_xpath("//div[@class='"+ PICTURE_CLASS_NAME + "']")
    for j in range(0, len(list_items)):
        url = list_items[j].get_attribute("data-src")
        path = os.path.abspath(os.path.join(os.pardir, "student_pictures"))
        if not os.path.exists(path):
            os.makedirs(path)
            print("created folders")

        response = requests.get(url, stream=True)
        with open(os.path.join(path, "{}_{}".format(name, j+1)), "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

    print("Downloaded {} images of {}".format(len(list_items), name))

    driver.back() #go back to main page
    students = driver.find_elements_by_xpath("//li[@class='"+ LIST_ITEMS_CLASS_NAME + "']")
    list_items = driver.find_elements_by_xpath("//div[@class='"+ PICTURE_CLASS_NAME + "']") #need to re-find the links to each student's page of pictures

print("Finished!")
driver.close()
