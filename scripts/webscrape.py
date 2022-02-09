import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys #import ability to type

URL = "https://halbrown.photoreflect.com/store/Photos.aspx?e=10785352"
PASSWORD_INPUT_NAME = "ctl01$CPCenter$PhotoSetAccess$EventChallengeAnswer"
PASSWORD_INPUT_CLASS = "psaEventAnswer" #unused, just in case we want to find by class instead of name
SUBMIT_BUTTON_CLASS = "btn psa-access-event psaEventPassBtn" #doesn't have a name
ANSWER_REQUIRED_CLASS = "error psaAnswerRequired"
INCORRECT_PASSWORD_CLASS = "error psaPassError"
PASSWORD = "millers2022" #password that's entered
SCROLL_PAUSE_TIME = 0.2 #maybe unnecessary?

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito") #makes Chrome open incognito so your history isn't flooded :P
driver = webdriver.Chrome(chrome_options=chrome_options)
driver.maximize_window()

driver.get(URL)

#enter password
answer_required = driver.find_element_by_class(ANSWER_REQUIRED_CLASS)
incorrect_password = driver.find_element_by_class(INCORRECT_PASSWORD_CLASS)

password_input = driver.find_element_by_name(PASSWORD_INPUT_NAME)
password_input.clear()
password_input.send_keys(PASSWORD)
submit_button = driver.find_element_by_class(SUBMIT_BUTTON_CLASS)
submit_button.click()

if answer_required.get_attribute("style") != "display: none":
    sys.exit("Couldn't enter the password in")
elif incorrect_password.get_attribute("style") != "display: none":
    sys.exit("Incorrect password: " + PASSWORD)

#scrape pictures

list_items = driver.find_elements_by_class("ps-photo-container all-rounded ps-photo-spaced ps-photo-box-shadow ps-photo-transparent-border photo-set-viewer-list-item")
for item in list_items:
    name = item.find_element_by_xpath(".//descendants::span").get_text() #get the only span from all of <item>'s descendants, then get its text
    #TODO: format name to be LAST_FIRST
    #TODO: click <item> then download images
    print(name)
